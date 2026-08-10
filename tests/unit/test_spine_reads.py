"""Unit tests for the spine read endpoints: concepts/by-study, observations/by-ids, observations/rollup."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-00000000bbbb"
OTHER = "00000000-0000-0000-0000-00000000dddd"
STUDY = "00000000-0000-0000-0000-0000000000a1"
CONCEPT = "00000000-0000-0000-0000-0000000000c1"


class _FakeSupabase:
    def __init__(self):
        self.rpc_calls: List[tuple] = []
        self._rpc_next: Dict[str, Any] = {}

    def set_rpc(self, name, data): self._rpc_next[name] = data

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        ret = self._rpc_next.get(name, [])
        class _R: pass
        r = _R(); r.data = ret; r.count = None
        class _E:
            def __init__(s, r): s._r = r
            def execute(s): return s._r
        return _E(r)

    def last(self, name):
        for n, p in reversed(self.rpc_calls):
            if n == name:
                return p
        raise AssertionError(f"{name} not called")


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import spine_router as spine_mod
    from src.middleware import auth as auth_mod
    fake = _FakeSupabase()
    getter = lambda: fake
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(spine_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)

    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT, "scopes": ["read", "write"], "status": "active", "expired": False})

    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_concepts_by_study(client):
    c, fake = client
    fake.set_rpc("concepts_by_study", [
        {"concept_id": CONCEPT, "label": "Onboarding friction"},
        {"concept_id": "00000000-0000-0000-0000-0000000000c2", "label": "Pricing confusion"},
    ])
    r = c.get(f"/concepts/by-study?tenant_id={TENANT}&study_ids={STUDY}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["label"] for x in body["concepts"]] == ["Onboarding friction", "Pricing confusion"]
    assert fake.last("concepts_by_study")["p_study_ids"] == [STUDY]


def test_observations_by_ids_returns_keyed_map(client):
    c, fake = client
    fake.set_rpc("observations_by_ids", [
        {"node_id": "n1", "observation_id": "obs-1", "value": {"number": 0.62, "unit": "pct"},
         "prevalence": {"pct": 0.62, "n": 31}, "reliability": {"method": "thematic"},
         "source": {"evidence_ref": "chunk-7"}, "study_id": STUDY, "evidence_chunk_ids": ["e7"]},
        {"node_id": "n2", "observation_id": "obs-2", "value": {"number": 8, "unit": "NPS"}, "study_id": STUDY},
    ])
    r = c.post("/observations/by-ids", json={"tenant_id": TENANT, "ids": ["obs-1", "obs-2"], "study_ids": [STUDY]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"obs-1", "obs-2"}                      # keyed by observation_id
    assert body["obs-1"]["value"] == {"number": 0.62, "unit": "pct"}    # verbatim
    assert body["obs-1"]["reliability"]["method"] == "thematic"
    assert body["obs-1"]["evidence_chunk_ids"] == ["e7"]
    p = fake.last("observations_by_ids")
    assert p["p_observation_ids"] == ["obs-1", "obs-2"]
    assert p["p_study_ids"] == [STUDY]


def test_observations_rollup(client):
    c, fake = client
    fake.set_rpc("observations_rollup", {
        "cube": [
            {"concept_id": CONCEPT, "label": "Onboarding friction", "study_id": STUDY,
             "modality": "interview", "persona": "SMB", "period": "2026-06", "direction": "negative",
             "obs_count": 4, "n_sum": 120},
        ],
        "evidence": {CONCEPT: ["obs-1", "obs-2"]},
    })
    r = c.post("/observations/rollup", json={"tenant_id": TENANT, "study_ids": [STUDY],
                                             "range": {"from": "2026-01", "to": "2026-12"}, "top_evidence": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cube"][0]["obs_count"] == 4 and body["cube"][0]["n_sum"] == 120
    assert body["evidence"][CONCEPT] == ["obs-1", "obs-2"]
    p = fake.last("observations_rollup")
    assert p["p_study_ids"] == [STUDY]
    assert p["p_range_from"] == "2026-01" and p["p_range_to"] == "2026-12"
    assert p["p_top_evidence"] == 3


def test_reads_tenant_mismatch_403(client):
    c, _ = client
    # auth on (harness) → body tenant must match key tenant
    assert c.post("/observations/by-ids", json={"tenant_id": OTHER, "ids": []}).status_code == 403
    assert c.post("/observations/rollup", json={"tenant_id": OTHER, "study_ids": []}).status_code == 403
    assert c.get(f"/concepts/by-study?tenant_id={OTHER}").status_code == 403

"""Unit tests: observation evidence persistence — upsert forwards the verbatim
quote + chunk links; by-concept returns the evidence quote."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-0000000000ee"
CONCEPT = "00000000-0000-0000-0000-0000000000c2"
CHUNK = "00000000-0000-0000-0000-0000000000c1"


class _FakeSB:
    def __init__(self):
        self._rpc: Dict[str, Any] = {}
        self._calls: Dict[str, List[dict]] = {}
    def set_rpc(self, n, d): self._rpc[n] = d
    def calls(self, n): return self._calls.get(n, [])
    def rpc(self, name, params):
        self._calls.setdefault(name, []).append(params)
        ret = self._rpc.get(name)
        class R: pass
        r = R(); r.data = ret
        class E:
            def __init__(s, r): s._r = r
            def execute(s): return s._r
        return E(r)


@pytest.fixture
def client(monkeypatch):
    from src.supabase import supabase_client as sbmod
    from src.routers import spine_router as spine_mod
    from src.middleware import auth as auth_mod
    from src.routers import ingest_router as ingest_mod
    fake = _FakeSB(); getter = lambda: fake
    for m in (sbmod, spine_mod, auth_mod):
        monkeypatch.setattr(m, "get_supabase", getter)
    monkeypatch.setattr(ingest_mod, "_embed_in_batches",
                        lambda texts, tenant_id=None: [[0.01] * 1536 for _ in texts])
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT,
                                         "scopes": ["read", "write"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_upsert_forwards_evidence_quote_and_chunk_ids(client):
    c, fake = client
    fake.set_rpc("upsert_observation", {
        "observation_id": "obs-1", "node_id": "n1", "evidence_linked": True, "concept_linked": True,
    })
    r = c.post("/observations/upsert", json={
        "tenant_id": TENANT,
        "observation_id": "obs-1",
        "nl_text": "Respondents couldn't find the sign-out button",
        "value": {},
        "concept_id": CONCEPT,
        "evidence": {"text": "I couldn't find where to sign out", "speaker": "P3", "offset_ms": 125000},
        "evidence_chunk_ids": [CHUNK],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["evidence_linked"] is True and body["concept_linked"] is True
    # the verbatim quote + chunk links reached the RPC
    call = fake.calls("upsert_observation")[0]
    assert call["p_evidence"] == {"text": "I couldn't find where to sign out", "speaker": "P3", "offset_ms": 125000}
    assert call["p_evidence_chunk_ids"] == [CHUNK]


def test_upsert_without_evidence_sends_nulls(client):
    c, fake = client
    fake.set_rpc("upsert_observation", {"observation_id": "obs-2", "node_id": "n2",
                                        "evidence_linked": False, "concept_linked": False})
    r = c.post("/observations/upsert", json={
        "tenant_id": TENANT, "observation_id": "obs-2", "nl_text": "plain", "value": {},
    })
    assert r.status_code == 200, r.text
    call = fake.calls("upsert_observation")[0]
    assert call["p_evidence"] is None
    assert call["p_evidence_chunk_ids"] is None


def test_by_concept_returns_evidence_quote(client):
    c, fake = client
    fake.set_rpc("observations_by_concept", [
        {"node_id": "n1", "observation_id": "obs-1", "nl_text": "sign-out hard to find",
         "value": {}, "study_id": None,
         "evidence_chunk_ids": [CHUNK],
         "evidence": {"text": "I couldn't find where to sign out", "speaker": "P3", "offset_ms": 125000}},
        {"node_id": "n2", "observation_id": "obs-2", "nl_text": "no evidence one",
         "value": {}, "study_id": None, "evidence_chunk_ids": [], "evidence": None},
    ])
    r = c.get(f"/observations/by-concept?tenant_id={TENANT}&concept_id={CONCEPT}")
    assert r.status_code == 200, r.text
    obs = r.json()["observations"]
    assert obs[0]["evidence"]["text"] == "I couldn't find where to sign out"
    assert obs[0]["evidence"]["speaker"] == "P3"
    assert obs[0]["evidence_chunk_ids"] == [CHUNK]
    assert obs[1]["evidence"] is None   # absent, not crashing

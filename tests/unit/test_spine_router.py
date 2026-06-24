"""Unit tests for the observation + concept spine endpoints (migration 43).

Mounts the FastAPI app with a fake Supabase whose RPCs return preset payloads,
and stubs embeddings, so the routes are exercised end-to-end without external
services. The RPCs themselves (SQL idempotency, tenant isolation, verbatim
storage) are covered by the integration checklist in the plan; here we assert the
HTTP layer plumbs the right params and faithfully carries value + provenance.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-00000000bbbb"
OTHER_TENANT = "00000000-0000-0000-0000-00000000dddd"
CLIENT = "00000000-0000-0000-0000-00000000cccc"
CONCEPT_NODE = "00000000-0000-0000-0000-0000000000c1"
STUDY = "00000000-0000-0000-0000-0000000000a1"


class _FakeSupabase:
    """Records rpc calls; returns preset payloads keyed by rpc name."""

    def __init__(self):
        self.rpc_calls: List[tuple] = []
        self._rpc_next: Dict[str, Any] = {}

    def set_rpc(self, name: str, data: Any):
        self._rpc_next[name] = data

    def table(self, name):  # not used by these routes, kept for parity
        raise AssertionError(f"unexpected table access: {name}")

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        ret = self._rpc_next.get(name, {})

        class _R:
            pass

        r = _R()
        r.data = ret
        r.count = None

        class _Exec:
            def __init__(self, r):
                self._r = r

            def execute(self):
                return self._r

        return _Exec(r)

    def last_params(self, name: str) -> Dict[str, Any]:
        for n, p in reversed(self.rpc_calls):
            if n == name:
                return p
        raise AssertionError(f"rpc {name} was never called")


@pytest.fixture
def client(monkeypatch):
    from src.supabase import supabase_client as sbmod
    from src.routers import spine_router as spine_mod
    from src.middleware import auth as auth_mod

    fake_sb = _FakeSupabase()
    getter = lambda: fake_sb
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(spine_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)

    # Deterministic embeddings (bound into spine_router's namespace at import).
    monkeypatch.setattr(spine_mod, "_embed_in_batches", lambda texts, **kw: [[0.1] * 1536 for _ in texts])

    from src.services import api_key_service

    def _verify(self, raw_key: str):
        return {
            "key_id": "00000000-0000-0000-0000-00000000aaaa",
            "tenant_id": TENANT,
            "scopes": ["read", "write", "admin"],
            "status": "active",
            "expired": False,
        }

    monkeypatch.setattr(api_key_service.ApiKeyService, "verify", _verify)

    from src.main import app
    c = TestClient(app)
    c.headers.update({"X-API-Key": "dp_test_key"})
    return c, fake_sb


# ── observations:upsert ─────────────────────────────────────────────────────


def test_observation_upsert_stores_value_verbatim_and_keys_by_id(client):
    c, fake = client
    fake.set_rpc("upsert_observation", {
        "observation_id": "obs-123",
        "node_id": "node-1",
        "evidence_linked": True,
        "concept_linked": True,
    })

    value = {"number": 4.27, "unit": "NPS"}
    resp = c.post("/observations/upsert", json={
        "tenant_id": TENANT,
        "client_id": CLIENT,
        "observation_id": "obs-123",
        "nl_text": "Detractors cite onboarding friction.",
        "value": value,
        "modality": "interview",
        "signal_type": "sentiment",
        "direction": "negative",
        "prevalence": {"pct": 0.62, "n": 31},
        "confidence": 0.8,
        "reliability": {"sample_n": 31, "method": "thematic", "quality_flags": ["low_diarization"]},
        "segment": {"persona": "SMB admin", "variant_key": "v2"},
        "occurred_at": "2026-06-01T00:00:00Z",
        "source": {"aggregate_id": "agg-9", "input_hash": "h1", "agent_version": "2.1", "evidence_ref": "chunk-7"},
        "study_id": STUDY,
        "concept_id": CONCEPT_NODE,
        "evidence_chunk_id": "00000000-0000-0000-0000-0000000000e7",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "observation_id": "obs-123",
        "node_id": "node-1",
        "evidence_linked": True,
        "concept_linked": True,
    }

    params = fake.last_params("upsert_observation")
    assert params["p_observation_id"] == "obs-123"          # idempotency key
    # value carried byte-for-byte, never reshaped
    assert params["p_properties"]["value"] == value
    assert params["p_properties"]["prevalence"] == {"pct": 0.62, "n": 31}
    assert params["p_properties"]["source"]["agent_version"] == "2.1"
    assert params["p_study_id"] == STUDY
    assert params["p_concept_id"] == CONCEPT_NODE
    assert params["p_embedding"] is not None and len(params["p_embedding"]) == 1536


def test_observation_upsert_tenant_mismatch_403(client):
    c, _ = client
    resp = c.post("/observations/upsert", json={
        "tenant_id": OTHER_TENANT,
        "observation_id": "obs-x",
        "nl_text": "irrelevant",
        "value": {"number": 1, "unit": "x"},
    })
    assert resp.status_code == 403


# ── observations:by-concept ─────────────────────────────────────────────────


def test_observations_by_concept_carries_value_and_provenance(client):
    c, fake = client
    fake.set_rpc("observations_by_concept", [
        {
            "node_id": "node-1",
            "observation_id": "obs-123",
            "nl_text": "Detractors cite onboarding friction.",
            "value": {"number": 4.27, "unit": "NPS"},
            "modality": "interview",
            "signal_type": "sentiment",
            "direction": "negative",
            "prevalence": {"pct": 0.62, "n": 31},
            "confidence": 0.8,
            "reliability": {"sample_n": 31, "method": "thematic"},
            "segment": {"persona": "SMB admin", "variant_key": "v2"},
            "occurred_at": "2026-06-01T00:00:00Z",
            "source": {"aggregate_id": "agg-9", "evidence_ref": "chunk-7"},
            "study_id": STUDY,
            "evidence_chunk_ids": ["00000000-0000-0000-0000-0000000000e7"],
        },
    ])

    resp = c.get(
        f"/observations/by-concept?tenant_id={TENANT}&concept_id={CONCEPT_NODE}"
        f"&study_ids={STUDY}&modality=interview"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["concept_id"] == CONCEPT_NODE
    obs = body["observations"][0]
    assert obs["value"] == {"number": 4.27, "unit": "NPS"}          # verbatim
    assert obs["prevalence"] == {"pct": 0.62, "n": 31}
    assert obs["reliability"]["sample_n"] == 31                      # reliability returned
    assert obs["source"]["evidence_ref"] == "chunk-7"               # provenance returned
    assert obs["evidence_chunk_ids"] == ["00000000-0000-0000-0000-0000000000e7"]

    params = fake.last_params("observations_by_concept")
    assert params["p_concept_id"] == CONCEPT_NODE
    assert params["p_study_ids"] == [STUDY]
    assert params["p_modality"] == "interview"


def test_observations_by_concept_tenant_mismatch_403(client):
    c, _ = client
    resp = c.get(f"/observations/by-concept?tenant_id={OTHER_TENANT}&concept_id={CONCEPT_NODE}")
    assert resp.status_code == 403


# ── concepts:create ─────────────────────────────────────────────────────────


def test_concept_create_mints_id_when_not_supplied(client):
    c, fake = client
    fake.set_rpc("create_concept", {
        "concept_id": "node-c1",
        "canonical_id": "11111111-2222-3333-4444-555555555555",
        "node_key": "concept:11111111-2222-3333-4444-555555555555",
        "created": True,
    })

    resp = c.post("/concepts/create", json={
        "tenant_id": TENANT,
        "client_id": CLIENT,
        "canonical_label": "Onboarding friction",
        "aliases": ["setup pain", "activation drop-off"],
    })
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["created"] is True
    assert body["canonical_id"] == "11111111-2222-3333-4444-555555555555"
    # No canonical_id supplied → server mints (passes None to the RPC, not the label)
    params = fake.last_params("create_concept")
    assert params["p_canonical_id"] is None
    assert params["p_canonical_label"] == "Onboarding friction"
    assert params["p_aliases"] == ["setup pain", "activation drop-off"]


def test_concept_create_idempotent_when_id_supplied(client):
    c, fake = client
    fake.set_rpc("create_concept", {
        "concept_id": "node-c1", "canonical_id": "fixed-id",
        "node_key": "concept:fixed-id", "created": False,
    })
    resp = c.post("/concepts/create", json={
        "tenant_id": TENANT,
        "canonical_label": "Onboarding friction",
        "canonical_id": "fixed-id",
    })
    assert resp.status_code == 201
    assert resp.json()["created"] is False
    assert fake.last_params("create_concept")["p_canonical_id"] == "fixed-id"


# ── concepts:nearest ────────────────────────────────────────────────────────


def test_concept_nearest_requires_embedding_or_text(client):
    c, _ = client
    resp = c.post("/concepts/nearest", json={"tenant_id": TENANT, "hints": ["friction"]})
    assert resp.status_code == 400


def test_concept_nearest_returns_candidates(client):
    c, fake = client
    fake.set_rpc("nearest_concepts", [
        {
            "id": "node-c1",
            "canonical_id": "cid-1",
            "canonical_label": "Onboarding friction",
            "alias_set": {"canonical": "Onboarding friction", "members": ["setup pain"]},
            "merge_confidence": 0.9,
            "similarity": 0.82,
            "final_score": 0.87,
        },
    ])
    resp = c.post("/concepts/nearest", json={
        "tenant_id": TENANT,
        "query_text": "users struggle to set things up",
        "hints": ["friction"],
        "top_k": 5,
    })
    assert resp.status_code == 200, resp.text
    cand = resp.json()["candidates"][0]
    assert cand["canonical_label"] == "Onboarding friction"
    assert cand["similarity"] == 0.82
    assert cand["score"] == 0.87
    params = fake.last_params("nearest_concepts")
    assert params["p_hints"] == ["friction"]
    assert len(params["p_embedding"]) == 1536


# ── concepts:merge ──────────────────────────────────────────────────────────


def test_concept_merge_returns_counts(client):
    c, fake = client
    fake.set_rpc("merge_concepts", {
        "merged": True, "rewired_count": 3, "surviving_member_count": 4,
    })
    resp = c.post("/concepts/merge", json={
        "tenant_id": TENANT,
        "surviving_concept_id": "node-c1",
        "source_concept_id": "node-c2",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"merged": True, "rewired_count": 3, "surviving_member_count": 4}
    params = fake.last_params("merge_concepts")
    assert params["p_surviving_concept_id"] == "node-c1"
    assert params["p_source_concept_id"] == "node-c2"

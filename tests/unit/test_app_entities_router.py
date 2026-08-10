"""Unit tests for the app-entity recall endpoints (migration 47).

Mocked Supabase RPCs + embeddings, same harness as test_spine_router. The SQL
(idempotency, study/kind scoping, status='active' exclusion, cosine ordering,
synonym recall) is covered by the integration checklist; here we assert the HTTP
layer plumbs the right params and shapes the response.
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
STUDY = "00000000-0000-0000-0000-0000000000a1"


class _Chain:
    """Minimal kg_nodes read stub for the text-change check; records filters."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def select(self, *_a, **_kw): return self
    def eq(self, *_a, **_kw): return self
    def is_(self, *_a, **_kw): return self
    def limit(self, *_a, **_kw): return self

    def execute(self):
        class _R: pass
        r = _R(); r.data = self._rows; r.count = None
        return r


class _FakeSupabase:
    def __init__(self):
        self.rpc_calls: List[tuple] = []
        self._rpc_next: Dict[str, Any] = {}
        self._table_rows: Dict[str, List[Dict[str, Any]]] = {}

    def set_rpc(self, name: str, data: Any):
        self._rpc_next[name] = data

    def set_table_rows(self, name: str, rows: List[Dict[str, Any]]):
        self._table_rows[name] = rows

    def table(self, name):
        return _Chain(self._table_rows.get(name, []))

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        ret = self._rpc_next.get(name, {})

        class _R: pass
        r = _R(); r.data = ret; r.count = None

        class _Exec:
            def __init__(self, r): self._r = r
            def execute(self): return self._r
        return _Exec(r)

    def last_params(self, name: str) -> Dict[str, Any]:
        for n, p in reversed(self.rpc_calls):
            if n == name:
                return p
        raise AssertionError(f"rpc {name} never called")

    def called(self, name: str) -> bool:
        return any(n == name for n, _ in self.rpc_calls)


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import spine_router as spine_mod
    from src.services import spine_service as spine_svc_mod
    from src.middleware import auth as auth_mod

    fake_sb = _FakeSupabase()
    getter = lambda: fake_sb
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(spine_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)
    monkeypatch.setattr(spine_svc_mod, "_embed_in_batches", lambda texts, **kw: [[0.2] * 1536 for _ in texts])

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


# ── upsert ──────────────────────────────────────────────────────────────────


def test_app_entity_upsert_created_then_idempotent(client):
    c, fake = client
    # First upsert: no existing row → embeds, RPC reports created.
    fake.set_rpc("upsert_app_entity", {
        "node_id": "node-1", "external_id": "ins-42", "kind": "insight", "created": True,
    })
    r1 = c.post("/app-entities/upsert", json={
        "tenant_id": TENANT, "study_id": STUDY,
        "kind": "insight", "external_id": "ins-42",
        "text": "Users abandon onboarding at the SSO step.",
    })
    assert r1.status_code == 200, r1.text
    assert r1.json()["created"] is True
    p1 = fake.last_params("upsert_app_entity")
    assert p1["p_external_id"] == "ins-42" and p1["p_kind"] == "insight"
    assert p1["p_study_id"] == STUDY
    assert p1["p_embedding"] is not None and len(p1["p_embedding"]) == 1536  # embedded (new)
    assert p1["p_status"] == "active"

    # Second upsert, SAME text already stored → no re-embed (p_embedding None),
    # RPC reports created=false.
    fake.set_table_rows("kg_nodes", [{"name": "Users abandon onboarding at the SSO step."}])
    fake.set_rpc("upsert_app_entity", {
        "node_id": "node-1", "external_id": "ins-42", "kind": "insight", "created": False,
    })
    r2 = c.post("/app-entities/upsert", json={
        "tenant_id": TENANT, "study_id": STUDY,
        "kind": "insight", "external_id": "ins-42",
        "text": "Users abandon onboarding at the SSO step.",
    })
    assert r2.status_code == 200
    assert r2.json()["created"] is False
    assert fake.last_params("upsert_app_entity")["p_embedding"] is None  # skipped re-embed


def test_app_entity_upsert_reembeds_on_text_change(client):
    c, fake = client
    fake.set_table_rows("kg_nodes", [{"name": "old text"}])
    fake.set_rpc("upsert_app_entity", {
        "node_id": "node-1", "external_id": "ins-42", "kind": "insight", "created": False,
    })
    c.post("/app-entities/upsert", json={
        "tenant_id": TENANT, "study_id": STUDY,
        "kind": "insight", "external_id": "ins-42", "text": "brand new wording",
    }).raise_for_status()
    assert fake.last_params("upsert_app_entity")["p_embedding"] is not None  # re-embedded


def test_app_entity_upsert_status_inactive_passed_through(client):
    c, fake = client
    fake.set_rpc("upsert_app_entity", {
        "node_id": "n", "external_id": "q1", "kind": "quote", "created": False,
    })
    c.post("/app-entities/upsert", json={
        "tenant_id": TENANT, "study_id": STUDY,
        "kind": "quote", "external_id": "q1", "text": "…", "status": "inactive",
    }).raise_for_status()
    assert fake.last_params("upsert_app_entity")["p_status"] == "inactive"


def test_app_entity_upsert_tenant_mismatch_403(client):
    c, _ = client
    r = c.post("/app-entities/upsert", json={
        "tenant_id": OTHER_TENANT, "study_id": STUDY,
        "kind": "insight", "external_id": "x", "text": "y",
    })
    assert r.status_code == 403


# ── nearest ─────────────────────────────────────────────────────────────────


def test_app_entity_nearest_returns_ids_scoped(client):
    c, fake = client
    fake.set_rpc("nearest_app_entities", [
        {"external_id": "ins-42", "kind": "insight", "study_id": STUDY, "similarity": 0.81},
        {"external_id": "call-7", "kind": "session", "study_id": STUDY, "similarity": 0.77},
    ])
    r = c.post("/app-entities/nearest", json={
        "tenant_id": TENANT,
        "query_text": "people give up before finishing setup",
        "study_ids": [STUDY],
        "kinds": ["insight", "session"],
        "top_k": 20,
    })
    assert r.status_code == 200, r.text
    matches = r.json()["matches"]
    assert [m["external_id"] for m in matches] == ["ins-42", "call-7"]   # app ids, not text
    assert matches[0]["similarity"] == 0.81
    assert all("text" not in m for m in matches)                        # never returns node text
    p = fake.last_params("nearest_app_entities")
    assert p["p_study_ids"] == [STUDY]
    assert p["p_kinds"] == ["insight", "session"]
    assert len(p["p_embedding"]) == 1536


def test_app_entity_nearest_requires_query_or_embedding(client):
    c, _ = client
    r = c.post("/app-entities/nearest", json={"tenant_id": TENANT, "study_ids": [STUDY]})
    assert r.status_code == 400


def test_app_entity_nearest_tenant_mismatch_403(client):
    c, _ = client
    r = c.post("/app-entities/nearest", json={
        "tenant_id": OTHER_TENANT, "query_text": "q", "study_ids": [STUDY],
    })
    assert r.status_code == 403

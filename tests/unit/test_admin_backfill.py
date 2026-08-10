"""Unit tests: /admin/mirror-taxonomy (batch mirror seeded tags) + /admin/reembed
(re-embed null-embedding rows). Both are system ops — embedder is mocked."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-00000000cccc"
TAG1 = "00000000-0000-0000-0000-0000000000f1"
TAG2 = "00000000-0000-0000-0000-0000000000f2"


class _Chain:
    """Minimal PostgREST table chain supporting the reembed remaining-count query."""
    def __init__(self, rows, count):
        self.rows = rows
        self._count = count
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def is_(self, *a, **k): return self
    def in_(self, *a, **k): return self
    def execute(self):
        class R: pass
        r = R(); r.data = self.rows; r.count = self._count; return r


class _FakeSB:
    def __init__(self):
        self._rpc: Dict[str, Any] = {}
        self._calls: Dict[str, List[dict]] = {}
        self._table_count = 0
    def set_rpc(self, n, d): self._rpc[n] = d
    def set_table_count(self, c): self._table_count = c
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
    def table(self, name): return _Chain([], self._table_count)


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import admin_router as admin_mod
    from src.middleware import auth as auth_mod
    from src.routers import ingest_router as ingest_mod
    fake = _FakeSB(); getter = lambda: fake
    for m in (sbmod, admin_mod, auth_mod):
        monkeypatch.setattr(m, "get_supabase", getter)
    # Deterministic embedder: one 1536-vector per input text, no OpenAI call.
    monkeypatch.setattr(ingest_mod, "_embed_in_batches",
                        lambda texts, tenant_id=None: [[0.01] * 1536 for _ in texts])
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT,
                                         "scopes": ["read", "admin"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


# ── /admin/mirror-taxonomy ──────────────────────────────────────────────────


def test_mirror_taxonomy_mirrors_each_tag_with_embedding(client):
    c, fake = client
    fake.set_rpc("mirror_tag_concept", {})   # RPC echoes nothing → external_ref falls back to tag_id
    r = c.post("/admin/mirror-taxonomy", json={
        "tenant_id": TENANT,
        "tags": [
            {"tag_id": TAG1, "label": "Pricing", "description": "pricing feedback"},
            {"tag_id": TAG2, "label": "Onboarding"},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mirrored"] == 2 and body["embedded"] == 2 and body["failed"] == 0
    # each tag mirrored → external_ref stamped with its own tag id
    refs = {item["tag_id"]: item["external_ref"] for item in body["results"]}
    assert refs[TAG1] == TAG1 and refs[TAG2] == TAG2
    assert all(item["embedded"] for item in body["results"])
    # the RPC was called once per tag, each carrying a 1536-dim embedding
    calls = fake.calls("mirror_tag_concept")
    assert len(calls) == 2
    assert len(calls[0]["p_embedding"]) == 1536
    assert calls[0]["p_tag_id"] == TAG1


def test_mirror_taxonomy_reports_invalid_tag_id_without_aborting(client):
    c, fake = client
    fake.set_rpc("mirror_tag_concept", {})
    r = c.post("/admin/mirror-taxonomy", json={
        "tenant_id": TENANT,
        "tags": [
            {"tag_id": "not-a-uuid", "label": "Bad"},
            {"tag_id": TAG1, "label": "Pricing"},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mirrored"] == 1 and body["failed"] == 1
    bad = next(i for i in body["results"] if i["tag_id"] == "not-a-uuid")
    assert "invalid tag_id" in (bad["error"] or "")


def test_mirror_taxonomy_empty_is_noop(client):
    c, fake = client
    r = c.post("/admin/mirror-taxonomy", json={"tenant_id": TENANT, "tags": []})
    assert r.status_code == 200
    assert r.json()["mirrored"] == 0
    assert fake.calls("mirror_tag_concept") == []


def test_mirror_taxonomy_bad_tenant_400(client):
    c, _ = client
    r = c.post("/admin/mirror-taxonomy", json={"tenant_id": "nope", "tags": []})
    assert r.status_code == 400


# ── /admin/reembed ──────────────────────────────────────────────────────────


def test_reembed_fills_null_embedding_rows(client):
    c, fake = client
    fake.set_rpc("reembed_candidates", [
        {"id": "11111111-0000-0000-0000-000000000001", "node_type": "Observation", "embed_text": "users found pricing confusing"},
        {"id": "11111111-0000-0000-0000-000000000002", "node_type": "Concept", "embed_text": "Pricing"},
    ])
    fake.set_rpc("reembed_apply_batch", 2)
    fake.set_table_count(0)   # nothing left after this sweep
    r = c.post("/admin/reembed", json={"tenant_id": TENANT, "types": ["Observation", "Concept"], "limit": 100})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scanned"] == 2 and body["reembedded"] == 2 and body["remaining"] == 0
    # apply batch received aligned ids + one 1536-vec per id
    call = fake.calls("reembed_apply_batch")[0]
    assert len(call["p_ids"]) == 2
    assert len(call["p_embeddings"]) == 2 and len(call["p_embeddings"][0]) == 1536


def test_reembed_no_candidates_is_noop(client):
    c, fake = client
    fake.set_rpc("reembed_candidates", [])
    fake.set_table_count(0)
    r = c.post("/admin/reembed", json={"tenant_id": TENANT})
    assert r.status_code == 200
    body = r.json()
    assert body["scanned"] == 0 and body["reembedded"] == 0
    assert fake.calls("reembed_apply_batch") == []   # never embed/apply when nothing to do


def test_reembed_reports_remaining_for_chunked_backfill(client):
    c, fake = client
    fake.set_rpc("reembed_candidates", [
        {"id": "11111111-0000-0000-0000-000000000003", "node_type": "Observation", "embed_text": "x"},
    ])
    fake.set_rpc("reembed_apply_batch", 1)
    fake.set_table_count(42)   # more remain → caller re-runs
    r = c.post("/admin/reembed", json={"tenant_id": TENANT, "limit": 1})
    assert r.status_code == 200
    assert r.json()["remaining"] == 42


def test_reembed_bad_tenant_400(client):
    c, _ = client
    r = c.post("/admin/reembed", json={"tenant_id": "nope"})
    assert r.status_code == 400

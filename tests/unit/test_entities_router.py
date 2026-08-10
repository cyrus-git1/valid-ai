"""Characterization tests for the /entities + /kg/entities endpoints.

Written BEFORE extracting EntityService so the extraction can be proven
behavior-preserving. Mocked Supabase (tables + RPCs) + embeddings, same harness
style as test_app_entities_router. Asserts the HTTP layer plumbs the right
params to the RPCs/queries and shapes responses correctly.
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
DOC = "00000000-0000-0000-0000-0000000000d1"
SHA = "a" * 64


class _Chain:
    """Records the table's filter/write calls; returns canned rows on execute."""

    def __init__(self, name: str, rows: List[Dict[str, Any]], sink: Dict[str, Any]):
        self._name = name
        self._rows = rows
        self._sink = sink

    def select(self, *_a, **_kw): return self
    def eq(self, *_a, **_kw): return self
    def filter(self, *_a, **_kw): return self
    def in_(self, *_a, **_kw): return self
    def order(self, *_a, **_kw): return self
    def limit(self, *_a, **_kw): return self

    def upsert(self, payload, **kw):
        self._sink.setdefault("upserts", []).append((self._name, payload, kw))
        return self

    def execute(self):
        class _R: pass
        r = _R(); r.data = self._rows; r.count = None
        return r


class _FakeSupabase:
    def __init__(self):
        self.rpc_calls: List[tuple] = []
        self._rpc_next: Dict[str, Any] = {}
        self._table_rows: Dict[str, List[Dict[str, Any]]] = {}
        self.sink: Dict[str, Any] = {}

    def set_rpc(self, name: str, data: Any):
        self._rpc_next[name] = data

    def set_table_rows(self, name: str, rows: List[Dict[str, Any]]):
        self._table_rows[name] = rows

    def table(self, name):
        return _Chain(name, self._table_rows.get(name, []), self.sink)

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
    from src.routers import entities_router as ent_mod
    from src.services import entity_service as ent_svc_mod
    from src.middleware import auth as auth_mod

    fake_sb = _FakeSupabase()
    getter = lambda: fake_sb
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(ent_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)
    # Embedding orchestration moved into EntityService — patch it there.
    monkeypatch.setattr(ent_svc_mod, "_embed_in_batches", lambda texts, **kw: [[0.2] * 1536 for _ in texts])

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


# ── Tier 1: extraction cache ──────────────────────────────────────────────────


def test_cache_get_bad_length_400(client):
    c, _ = client
    r = c.get("/entities/cache/tooshort")
    assert r.status_code == 400


def test_cache_get_miss_404(client):
    c, fake = client
    fake.set_table_rows("entity_extraction_cache", [])
    r = c.get(f"/entities/cache/{SHA}")
    assert r.status_code == 404


def test_cache_get_hit_shapes_response(client):
    c, fake = client
    fake.set_table_rows("entity_extraction_cache", [{
        "transcript_sha256": SHA,
        "entities": [{"name": "Acme"}],
        "model_name": "gpt-4o-mini",
        "prompt_version": "v1",
        "created_at": "2026-01-01T00:00:00Z",
        "expires_at": "2999-01-01T00:00:00Z",
        "tenant_id": TENANT,
    }])
    r = c.get(f"/entities/cache/{SHA}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript_sha256"] == SHA
    assert body["entities"] == [{"name": "Acme"}]
    assert body["model_name"] == "gpt-4o-mini"


def test_cache_get_expired_404(client):
    c, fake = client
    fake.set_table_rows("entity_extraction_cache", [{
        "transcript_sha256": SHA, "entities": [], "model_name": "m",
        "prompt_version": "v1", "created_at": "2020-01-01T00:00:00Z",
        "expires_at": "2020-01-02T00:00:00Z", "tenant_id": TENANT,
    }])
    r = c.get(f"/entities/cache/{SHA}")
    assert r.status_code == 404


def test_cache_get_prompt_version_mismatch_404(client):
    c, fake = client
    fake.set_table_rows("entity_extraction_cache", [{
        "transcript_sha256": SHA, "entities": [], "model_name": "m",
        "prompt_version": "v1", "created_at": "2026-01-01T00:00:00Z",
        "expires_at": "2999-01-01T00:00:00Z", "tenant_id": TENANT,
    }])
    r = c.get(f"/entities/cache/{SHA}?prompt_version=v2")
    assert r.status_code == 404


def test_cache_upsert_tenant_mismatch_403(client):
    c, _ = client
    r = c.post("/entities/cache", json={
        "transcript_sha256": SHA, "tenant_id": OTHER_TENANT,
        "entities": [], "model_name": "m",
    })
    assert r.status_code == 403


def test_cache_upsert_success_writes_payload(client):
    c, fake = client
    r = c.post("/entities/cache", json={
        "transcript_sha256": SHA, "tenant_id": TENANT,
        "entities": [{"name": "Acme"}], "model_name": "gpt-4o-mini",
    })
    assert r.status_code == 201, r.text
    assert r.json()["transcript_sha256"] == SHA
    upserts = fake.sink.get("upserts", [])
    assert upserts, "expected an upsert to entity_extraction_cache"
    name, payload, kw = upserts[-1]
    assert name == "entity_extraction_cache"
    assert payload["transcript_sha256"] == SHA
    assert payload["tenant_id"] == TENANT
    assert kw.get("on_conflict") == "transcript_sha256"


# ── Tier 2: entity KG upsert + merge ─────────────────────────────────────────


def test_kg_upsert_new_entity_embeds_and_calls_rpc(client):
    c, fake = client
    fake.set_table_rows("kg_nodes", [])  # no existing node → treated as new → embed
    fake.set_rpc("upsert_entity_with_mention", {"entity_id": "e1"})
    r = c.post("/kg/entities/upsert", json={
        "tenant_id": TENANT, "document_id": DOC,
        "entities": [{
            "canonical_id": "abcdefabcdef", "canonical_text": "Acme Corporation",
            "label": "ORG", "source": "spacy",
        }],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["upserted_entities"] == 1
    assert body["upserted_mentions"] == 1
    p = fake.last_params("upsert_entity_with_mention")
    assert p["p_tenant_id"] == TENANT
    assert p["p_canonical_id"] == "abcdefabcdef"
    assert p["p_label"] == "ORG"
    assert p["p_embedding"] is not None and len(p["p_embedding"]) == 1536


def test_kg_upsert_tenant_mismatch_403(client):
    c, _ = client
    r = c.post("/kg/entities/upsert", json={
        "tenant_id": OTHER_TENANT, "document_id": DOC,
        "entities": [{"canonical_id": "abcdefabcdef", "canonical_text": "x", "label": "ORG"}],
    })
    assert r.status_code == 403


def test_kg_merge_source_not_found_404(client):
    c, fake = client
    fake.set_table_rows("kg_nodes", [])  # _resolve_canonical → no rows
    r = c.post("/kg/entities/abcdefabcdef/merge", json={
        "tenant_id": TENANT, "surviving_canonical_id": "111111111111",
        "surviving_label": "ORG",
    })
    assert r.status_code == 404


# ── Tier 3: search + rollup ──────────────────────────────────────────────────


def test_search_list_path_filters_and_sorts(client):
    c, fake = client
    fake.set_table_rows("kg_nodes", [
        {"id": "n1", "name": "Acme", "status": "active",
         "properties": {"canonical_id": "c1", "label": "ORG", "total_mentions": 2}},
        {"id": "n2", "name": "Globex", "status": "active",
         "properties": {"canonical_id": "c2", "label": "ORG", "total_mentions": 9}},
    ])
    r = c.get(f"/kg/entities/search?tenant_id={TENANT}&min_mentions=1")
    assert r.status_code == 200, r.text
    ents = r.json()["entities"]
    assert [e["canonical_id"] for e in ents] == ["c2", "c1"]  # sorted by mentions desc


def test_search_semantic_path_calls_search_rpc(client):
    c, fake = client
    fake.set_rpc("search_kg_nodes", [
        {"properties": {"canonical_id": "c1", "label": "ORG", "total_mentions": 3},
         "name": "Acme", "final_score": 0.9},
    ])
    r = c.get(f"/kg/entities/search?tenant_id={TENANT}&q=acme&top_k=5")
    assert r.status_code == 200, r.text
    assert fake.called("search_kg_nodes")
    p = fake.last_params("search_kg_nodes")
    assert p["p_types"] == ["Entity"]
    assert p["p_top_k"] == 5
    assert len(p["p_embedding"]) == 1536


def test_rollup_not_found_404(client):
    c, fake = client
    fake.set_table_rows("kg_nodes", [])
    r = c.get(f"/kg/entities/abcdefabcdef?tenant_id={TENANT}")
    assert r.status_code == 404


def test_rollup_success_shapes_entity(client):
    c, fake = client
    fake.set_table_rows("kg_nodes", [
        {"id": "n1", "name": "Acme", "status": "active",
         "properties": {"canonical_id": "abcdefabcdef", "label": "ORG",
                        "canonical_text": "Acme Corporation", "total_mentions": 4}},
    ])
    fake.set_table_rows("kg_edges", [])  # no mentions
    r = c.get(f"/kg/entities/abcdefabcdef?tenant_id={TENANT}&label=ORG")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["entity"]["canonical_id"] == "abcdefabcdef"
    assert body["entity"]["total_mentions"] == 4
    assert body["mentions"] == []

"""Unit tests for the new summary endpoints + source_types plumbing.

These tests mount the FastAPI app with monkeypatched supabase + openai so
the routes can be exercised end-to-end without external services.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"   # exercises the AuthMiddleware path
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")


class _Chain:
    """Supabase-py query chain stub that records every call."""

    def __init__(self, table_name: str, store: Dict[str, Any]):
        self.table_name = table_name
        self.store = store
        self._pending_insert: Any = None
        self._pending_update: Any = None
        self._filters: List[tuple] = []

    def select(self, *_a, **_kw): return self
    def eq(self, col, val): self._filters.append(("eq", col, val)); return self
    def in_(self, col, val): self._filters.append(("in", col, val)); return self
    def order(self, *_a, **_kw): return self
    def limit(self, *_a, **_kw): return self

    def insert(self, row):
        self._pending_insert = row
        return self

    def update(self, row):
        self._pending_update = row
        return self

    def delete(self):
        return self

    def execute(self):
        rows = list(self.store.get(self.table_name, []))

        # Apply filters for reads
        def _match(r: dict) -> bool:
            for op, col, val in self._filters:
                if op == "eq" and r.get(col) != val:
                    return False
                if op == "in" and r.get(col) not in val:
                    return False
            return True

        if self._pending_insert is not None:
            rec = dict(self._pending_insert)
            rec.setdefault("id", f"id-{len(rows) + 1}")
            self.store.setdefault(self.table_name, []).append(rec)
            class _R: pass
            r = _R(); r.data = [rec]; r.count = None; return r

        if self._pending_update is not None:
            matched = [r for r in rows if _match(r)]
            for r in matched:
                r.update(self._pending_update)
            class _R: pass
            r = _R(); r.data = matched; r.count = None; return r

        filtered = [r for r in rows if _match(r)]
        class _R: pass
        r = _R(); r.data = filtered; r.count = len(filtered); return r


class _FakeSupabase:
    def __init__(self):
        self.store: Dict[str, List[Dict[str, Any]]] = {
            "documents": [],
            "chunks": [],
            "kg_nodes": [],
            "memory_state": [],
        }
        self.rpc_calls: List[tuple] = []
        self._rpc_next: Dict[str, Any] = {}

    def table(self, name):
        return _Chain(name, self.store)

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        ret = self._rpc_next.get(name)
        class _R: pass
        r = _R()
        r.data = ret if ret is not None else f"rpc-{name}-id"
        r.count = None
        class _Exec:
            def __init__(self, r): self._r = r
            def execute(self): return self._r
        return _Exec(r)


@pytest.fixture
def client(monkeypatch):
    # Stub out supabase + embeddings before the app imports anything.
    # Python's `from X import Y` binds Y at import time; patch each consumer module.
    from src.db import supabase_client as sbmod
    from src.routers import data_router as data_mod
    from src.routers import ingest_router as ing_mod
    from src.middleware import auth as auth_mod
    fake_sb = _FakeSupabase()
    getter = lambda: fake_sb
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(data_mod, "get_supabase", getter)
    monkeypatch.setattr(ing_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)

    # Auth bypass via monkeypatched verify (same pattern as test_auth_middleware)
    from src.services import api_key_service

    def _verify(self, raw_key: str):
        return {
            "key_id": "00000000-0000-0000-0000-00000000aaaa",
            "tenant_id": "00000000-0000-0000-0000-00000000bbbb",
            "scopes": ["read", "write", "admin"],
            "status": "active",
            "expired": False,
        }

    monkeypatch.setattr(api_key_service.ApiKeyService, "verify", _verify)

    # Stub OpenAI embedding call used by /ingest/summary
    from src.routers import ingest_router as ing
    monkeypatch.setattr(ing, "_embed_texts", lambda texts, **kwargs: [[0.0] * 1536 for _ in texts])

    # Stub memory_state so bump_dual returns a stable version
    from src.services import memory_state_service
    def _fake_bump_dual(self, **kw):
        return {"client_version": 42, "tenant_version": 100}
    monkeypatch.setattr(memory_state_service.MemoryStateService, "bump_dual", _fake_bump_dual)

    def _fake_get_state(self, **kw):
        return {"memory_version": 42}
    monkeypatch.setattr(memory_state_service.MemoryStateService, "get_state", _fake_get_state)

    from src.main import app
    c = TestClient(app)
    c.headers.update({"X-API-Key": "dp_test_key"})
    return c, fake_sb


# ── /ingest/summary fast-path ──────────────────────────────────────────────


def test_summary_ingest_creates_one_of_each(client):
    c, fake = client
    body = {
        "tenant_id": "00000000-0000-0000-0000-00000000bbbb",
        "client_id": "00000000-0000-0000-0000-00000000cccc",
        "source_type": "ContextSummary",
        "summary_text": "This tenant is focused on enterprise SaaS onboarding.",
        "topics": ["onboarding", "enterprise"],
        "source_chunk_ids": ["chunk-a", "chunk-b"],
        "source_stats": {"documents_retrieved": 5},
        "memory_version_at_generation": 40,
    }
    resp = c.post("/ingest/summary", json=body)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["document_id"]
    assert data["chunk_id"]
    assert data["node_id"]
    assert data["superseded_document_id"] is None
    assert data["memory_version"] == 42

    # Exactly one document row, no semantic edges implied
    assert len(fake.store["documents"]) == 1
    doc = fake.store["documents"][0]
    assert doc["source_type"] == "ContextSummary"
    assert doc["is_canonical"] is True
    assert doc["status"] == "active"
    assert doc["metadata"]["memory_version_at_generation"] == 40


def test_summary_ingest_invalid_source_type(client):
    c, _ = client
    resp = c.post("/ingest/summary", json={
        "tenant_id": "00000000-0000-0000-0000-00000000bbbb",
        "client_id": "00000000-0000-0000-0000-00000000cccc",
        "source_type": "NotASummary",
        "summary_text": "...",
    })
    assert resp.status_code == 400


def test_summary_ingest_topic_requires_topic_field(client):
    c, _ = client
    resp = c.post("/ingest/summary", json={
        "tenant_id": "00000000-0000-0000-0000-00000000bbbb",
        "client_id": "00000000-0000-0000-0000-00000000cccc",
        "source_type": "TopicSummary",
        "summary_text": "...",
        # topic is omitted → should 400
    })
    assert resp.status_code == 400


def test_summary_ingest_canonical_flip(client):
    c, fake = client
    body = {
        "tenant_id": "00000000-0000-0000-0000-00000000bbbb",
        "client_id": "00000000-0000-0000-0000-00000000cccc",
        "source_type": "ContextSummary",
        "summary_text": "v1",
        "topics": [],
    }
    c.post("/ingest/summary", json=body).raise_for_status()
    body["summary_text"] = "v2"
    r2 = c.post("/ingest/summary", json=body)
    r2.raise_for_status()

    # Two docs total; only one remains canonical+active
    docs = fake.store["documents"]
    assert len(docs) == 2
    canonical = [d for d in docs if d["is_canonical"] and d["status"] == "active"]
    assert len(canonical) == 1
    deprecated = [d for d in docs if not d["is_canonical"] or d["status"] != "active"]
    assert len(deprecated) == 1
    assert r2.json()["superseded_document_id"] is not None


# ── source_types plumbing on /search/graph ─────────────────────────────────


def test_search_graph_passes_source_types_to_rpc(client, monkeypatch):
    c, fake = client
    # Intercept the KG retriever entirely — it's out of scope; we just want to
    # verify the request body deserialises correctly when source_types is set.
    from src.services.kg_retriever_service import KGRetrieverService
    monkeypatch.setattr(
        KGRetrieverService, "_vector_search",
        lambda self, emb, query_text=None: [],   # empty result set is fine for this test
    )
    monkeypatch.setattr(
        KGRetrieverService, "_embed_query",
        lambda self, q: [0.0] * 1536,
    )

    resp = c.post("/search/graph", json={
        "tenant_id": "00000000-0000-0000-0000-00000000bbbb",
        "client_id": "00000000-0000-0000-0000-00000000cccc",
        "query": "summarize what we know",
        "top_k": 5,
        "hop_limit": 0,
        "source_types": ["ContextSummary", "TopicSummary"],
    })
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["results"] == []
    assert payload["query"] == "summarize what we know"


# ── GET /data/context/summary/get reads from documents+chunks ──────────────


def test_get_context_summary_reads_from_documents(client):
    c, fake = client
    # Seed a ContextSummary document + its chunk
    tenant = "00000000-0000-0000-0000-00000000bbbb"
    client_id = "00000000-0000-0000-0000-00000000cccc"
    fake.store["documents"].append({
        "id": "doc-1",
        "tenant_id": tenant,
        "client_id": client_id,
        "source_type": "ContextSummary",
        "is_canonical": True,
        "status": "active",
        "metadata": {
            "topics": ["growth"],
            "source_stats": {"documents_retrieved": 3},
            "source_chunk_ids": ["x"],
            "memory_version_at_generation": 10,
        },
        "created_at": "2026-04-01T00:00:00Z",
        "updated_at": "2026-04-15T00:00:00Z",
    })
    fake.store["chunks"].append({
        "id": "chunk-1",
        "tenant_id": tenant,
        "document_id": "doc-1",
        "chunk_index": 0,
        "content": "Tenant focused on growth.",
        "metadata": {},
    })

    resp = c.get(f"/data/context/summary/get?tenant_id={tenant}&client_id={client_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"] == "Tenant focused on growth."
    assert body["topics"] == ["growth"]
    # current memory_version is 42 (from fake bump), gen is 10 → stale
    assert body["is_stale"] is True
    assert body["current_memory_version"] == 42
    assert body["memory_version_at_generation"] == 10


def test_get_context_summary_404_when_missing(client):
    c, _ = client
    tenant = "00000000-0000-0000-0000-00000000bbbb"
    client_id = "00000000-0000-0000-0000-00000000cccc"
    resp = c.get(f"/data/context/summary/get?tenant_id={tenant}&client_id={client_id}")
    assert resp.status_code == 404


# ── GET /data/summaries lists canonical summaries ──────────────────────────


def test_list_summaries_returns_canonical_only(client):
    c, fake = client
    tenant = "00000000-0000-0000-0000-00000000bbbb"
    client_id = "00000000-0000-0000-0000-00000000cccc"
    fake.store["documents"].extend([
        {
            "id": "d1",
            "tenant_id": tenant,
            "client_id": client_id,
            "source_type": "TopicSummary",
            "is_canonical": True,
            "status": "active",
            "metadata": {"topic": "churn", "memory_version_at_generation": 40},
            "created_at": "2026-04-01T00:00:00Z",
            "updated_at": "2026-04-10T00:00:00Z",
        },
        {
            "id": "d2",
            "tenant_id": tenant,
            "client_id": client_id,
            "source_type": "TopicSummary",
            "is_canonical": False,
            "status": "deprecated",
            "metadata": {"topic": "churn"},
            "created_at": "2026-03-01T00:00:00Z",
            "updated_at": "2026-03-15T00:00:00Z",
        },
    ])

    resp = c.get(f"/data/summaries?tenant_id={tenant}&client_id={client_id}&source_type=TopicSummary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["scope_ref"] == "churn"
    # gen=40 < current=42 → stale
    assert body["items"][0]["is_stale"] is True

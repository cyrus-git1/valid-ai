"""Unit tests for /memory/{state,changes} and rerank_score threading into search results."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-00000000bbbb"


class _Chain:
    def __init__(self, rows): self.rows = rows
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def is_(self, *a, **k): return self
    def gt(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self):
        class R: pass
        r = R(); r.data = self.rows; return r


class _FakeSB:
    def __init__(self, tables): self.tables = tables
    def table(self, name): return _Chain(self.tables.get(name, []))


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import memory_router as mem_mod
    from src.middleware import auth as auth_mod
    fake = _FakeSB({
        "memory_state": [{"memory_version": 42, "last_changed_at": "2026-07-22T00:00:00Z"}],
        "memory_change_log": [
            {"change_type": "ingest", "memory_version": 41, "metadata": {"docs": 2}, "created_at": "2026-07-21T00:00:00Z"},
            {"change_type": "emit", "memory_version": 42, "metadata": {}, "created_at": "2026-07-22T00:00:00Z"},
        ],
    })
    getter = lambda: fake
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(mem_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT, "scopes": ["read"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c


def test_memory_state(client):
    r = client.get(f"/memory/state?tenant_id={TENANT}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["memory_version"] == 42
    assert body["last_changed_at"] == "2026-07-22T00:00:00Z"


def test_memory_changes(client):
    r = client.get(f"/memory/changes?tenant_id={TENANT}&since_version=40")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["current_version"] == 42
    assert [c["memory_version"] for c in body["changes"]] == [41, 42]
    assert body["changes"][0]["change_type"] == "ingest"
    assert body["changes"][0]["metadata"] == {"docs": 2}


def test_rerank_score_threads_into_result_item():
    from src.routers.search_router import _docs_to_result_items

    class _Doc:
        def __init__(self, content, meta): self.page_content = content; self.metadata = meta

    docs = [_Doc("passage", {"node_id": "n1", "node_key": "k", "node_type": "Chunk",
                             "similarity_score": 0.70, "rerank_score": 0.87})]
    items = _docs_to_result_items(docs)
    assert items[0].rerank_score == 0.87
    # absent when not reranked
    docs2 = [_Doc("p", {"node_id": "n2", "node_key": "k2", "node_type": "Chunk", "similarity_score": 0.6})]
    assert _docs_to_result_items(docs2)[0].rerank_score is None

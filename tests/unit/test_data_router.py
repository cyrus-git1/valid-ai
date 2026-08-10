"""Characterization tests for the uncovered /data endpoints.

Written BEFORE extracting DataService so the extraction can be proven
behavior-preserving. Covers the 7 endpoints not already exercised by
test_preview_url / test_summary_router / test_evidence_and_survey:
list_documents, patch_document, delete_documents, kg-entities,
document-titles, summaries/document/{id}, summaries/topic.

Locks the Python-side behavior (response shapes, status codes, PII-redaction
branch, per-item flags). The SQL filters themselves are integration-covered.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")
os.environ.setdefault("SUPABASE_URL", "https://proj.supabase.co")

TENANT = "00000000-0000-0000-0000-00000000bbbb"
CLIENT = "00000000-0000-0000-0000-00000000cccc"
DOC = "11111111-1111-4111-8111-111111111111"


class _Chain:
    """Permissive query/write stub. All filters are no-ops; execute returns the
    table's canned rows; writes are recorded to the shared sink."""

    def __init__(self, name: str, rows: List[Dict[str, Any]], sink: Dict[str, Any]):
        self._name = name
        self._rows = rows
        self._sink = sink

    # read filters — all no-ops returning self
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def is_(self, *a, **k): return self
    def in_(self, *a, **k): return self
    def gte(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self

    @property
    def not_(self): return self

    # writes — record + return self so .execute() chains
    def insert(self, payload, **k):
        self._sink.setdefault("insert", []).append((self._name, payload)); return self

    def update(self, payload, **k):
        self._sink.setdefault("update", []).append((self._name, payload)); return self

    def delete(self, **k):
        self._sink.setdefault("delete", []).append(self._name); return self

    def upsert(self, payload, **k):
        self._sink.setdefault("upsert", []).append((self._name, payload)); return self

    def execute(self):
        class _R: pass
        r = _R(); r.data = self._rows; r.count = len(self._rows)
        return r


class _FakeSB:
    def __init__(self):
        self._rows: Dict[str, List[Dict[str, Any]]] = {}
        self.sink: Dict[str, Any] = {}
        self.rpc_calls: List[tuple] = []

    def set_rows(self, name: str, rows: List[Dict[str, Any]]):
        self._rows[name] = rows

    def table(self, name):
        return _Chain(name, self._rows.get(name, []), self.sink)

    def rpc(self, name, params=None):
        self.rpc_calls.append((name, params))

        class _R: pass
        r = _R(); r.data = {}; r.count = None

        class _Exec:
            def execute(self_inner): return r
        return _Exec()


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import data_router as data_mod
    from src.middleware import auth as auth_mod

    fake = _FakeSB()
    getter = lambda: fake
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(data_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)

    from src.services import api_key_service
    monkeypatch.setattr(
        api_key_service.ApiKeyService, "verify",
        lambda self, k: {"key_id": "k", "tenant_id": TENANT,
                         "scopes": ["read", "write", "admin"], "status": "active", "expired": False},
    )

    from src.main import app
    c = TestClient(app)
    c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


# ── list_documents ───────────────────────────────────────────────────────────


def test_list_documents_shapes_items_and_chunk_flags(client):
    c, fake = client
    fake.set_rows("documents", [{
        "id": DOC, "tenant_id": TENANT, "client_id": CLIENT, "source_type": "pdf",
        "source_uri": "bucket:pdf/x", "title": "Doc", "created_at": "2026-01-01",
        "updated_at": "2026-01-02",
    }])
    fake.set_rows("chunks", [{
        "id": "ch1", "document_id": DOC, "chunk_index": 0, "content": "hello",
        "content_tokens": 5, "metadata": {}, "embedding": [0.1], "pii_annotations": [],
    }])
    r = c.get(f"/data/documents?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["id"] == DOC
    ch = item["chunks"][0]
    assert ch["content"] == "hello"
    assert ch["has_embedding"] is True      # embedding present
    assert ch["has_pii"] is False           # no annotations
    assert "embedding" not in ch            # raw vector never leaked


def test_list_documents_empty(client):
    c, fake = client
    fake.set_rows("documents", [])
    r = c.get(f"/data/documents?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 200
    assert r.json() == {"items": [], "total": 0}


# ── patch_document ───────────────────────────────────────────────────────────


def test_patch_document_invalid_status_400(client):
    c, fake = client
    fake.set_rows("documents", [{"id": DOC}])  # exists
    r = c.patch(f"/data/documents/{DOC}?tenant_id={TENANT}&client_id={CLIENT}",
                json={"status": "bogus"})
    assert r.status_code == 400


def test_patch_document_not_found_404(client):
    c, fake = client
    fake.set_rows("documents", [])  # does not exist
    r = c.patch(f"/data/documents/{DOC}?tenant_id={TENANT}&client_id={CLIENT}",
                json={"is_pinned": True})
    assert r.status_code == 404


def test_patch_document_valid_update(client):
    c, fake = client
    fake.set_rows("documents", [{"id": DOC}])
    r = c.patch(f"/data/documents/{DOC}?tenant_id={TENANT}&client_id={CLIENT}",
                json={"status": "archived", "is_pinned": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["updated"] is True
    assert body["status"] == "archived"
    assert body["is_pinned"] is True
    assert any(t == "documents" for t, _ in fake.sink.get("update", []))


def test_patch_document_noop_when_no_fields(client):
    c, fake = client
    fake.set_rows("documents", [{"id": DOC}])
    r = c.patch(f"/data/documents/{DOC}?tenant_id={TENANT}&client_id={CLIENT}", json={})
    assert r.status_code == 200
    assert r.json() == {"updated": False, "document_id": DOC}


# ── delete_documents ─────────────────────────────────────────────────────────


def test_delete_documents_reports_deleted_and_not_found(client):
    c, fake = client
    # doc exists → gets deleted
    fake.set_rows("documents", [{"id": DOC, "client_id": CLIENT}])
    r = c.post(f"/data/documents/delete?tenant_id={TENANT}&client_id={CLIENT}",
               json={"document_ids": [DOC]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] == 1
    assert body["not_found"] == []
    assert "documents" in fake.sink.get("delete", [])


def test_delete_documents_missing_goes_to_not_found(client):
    c, fake = client
    fake.set_rows("documents", [])  # nothing exists → not_found
    missing = "22222222-2222-4222-8222-222222222222"
    r = c.post(f"/data/documents/delete?tenant_id={TENANT}&client_id={CLIENT}",
               json={"document_ids": [missing]})
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == 0
    assert body["not_found"] == [missing]


# ── kg-entities ──────────────────────────────────────────────────────────────


def test_kg_entities_maps_rows(client):
    c, fake = client
    fake.set_rows("kg_nodes", [
        {"name": "Acme", "type": "ORG", "description": "a co"},
        {"name": "Bob", "type": "PERSON", "description": None},
    ])
    r = c.get(f"/data/kg-entities?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 200, r.text
    ents = r.json()["entities"]
    assert {e["name"] for e in ents} == {"Acme", "Bob"}


# ── document-titles ──────────────────────────────────────────────────────────


def test_document_titles_empty_ids_returns_empty(client):
    c, _ = client
    r = c.post("/data/document-titles",
               json={"tenant_id": TENANT, "client_id": CLIENT, "document_ids": []})
    assert r.status_code == 200
    assert r.json() == {"titles": {}}


def test_document_titles_resolves(client):
    c, fake = client
    fake.set_rows("documents", [
        {"id": DOC, "title": "Real Title", "source_uri": None, "source_type": "pdf"},
    ])
    r = c.post("/data/document-titles",
               json={"tenant_id": TENANT, "client_id": CLIENT, "document_ids": [DOC]})
    assert r.status_code == 200, r.text
    assert r.json()["titles"] == {DOC: "Real Title"}


# ── scoped summaries (document / topic) ──────────────────────────────────────


def test_summary_document_not_found_404(client):
    c, fake = client
    fake.set_rows("documents", [])  # no matching DocumentSummary
    r = c.get(f"/data/summaries/document/{DOC}?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 404


def test_summary_topic_found(client):
    c, fake = client
    fake.set_rows("documents", [{
        "id": "sum1", "tenant_id": TENANT, "client_id": CLIENT,
        "source_type": "TopicSummary", "metadata": {"topic": "pricing"},
        "created_at": "2026-01-01", "updated_at": "2026-01-02",
    }])
    fake.set_rows("chunks", [{"content": "pricing summary text", "pii_annotations": []}])
    r = c.get(f"/data/summaries/topic?tenant_id={TENANT}&client_id={CLIENT}&topic=pricing")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_type"] == "TopicSummary"
    assert body["summary"] == "pricing summary text"

"""Unit tests: context corrections — pure text transform + CRUD endpoints."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-0000000000ab"
CLIENT = "00000000-0000-0000-0000-0000000000cd"


# ── pure transform ───────────────────────────────────────────────────────────


def test_apply_corrections_term_replace_and_disregard():
    from src.services.retrieval_postprocess import apply_corrections
    text = "Valid Technologies makes software. Contact Valid Technologies today. Ignore MEH."
    rows = [
        {"kind": "term_replace", "from_term": "Valid Technologies", "to_term": "NewName"},
        {"kind": "disregard", "from_term": "MEH"},
    ]
    out = apply_corrections(text, rows)
    assert "Valid Technologies" not in out
    assert out.count("NewName") == 2         # case-insensitive, both occurrences
    assert "MEH" not in out                   # disregard drops it


def test_apply_corrections_noop_when_empty():
    from src.services.retrieval_postprocess import apply_corrections
    assert apply_corrections("hello", []) == "hello"
    assert apply_corrections(None, [{"kind": "disregard", "from_term": "x"}]) is None


# ── CRUD endpoints ───────────────────────────────────────────────────────────


class _FakeTable:
    def __init__(self, sb, name):
        self.sb, self.name, self.mode, self.payload = sb, name, None, None
    def insert(self, row): self.mode, self.payload = "insert", row; return self
    def select(self, *a, **k): self.mode = "select"; return self
    def delete(self): self.mode = "delete"; return self
    def eq(self, *a, **k): return self
    def or_(self, *a, **k): return self
    def execute(self):
        class R: pass
        r = R()
        if self.mode == "insert":
            self.sb.inserted.append(self.payload)
            r.data = [{"id": "corr-1"}]
        elif self.mode == "delete":
            r.data = self.sb.delete_result
        else:
            r.data = self.sb.select_rows
        return r


class _FakeSB:
    def __init__(self):
        self.inserted: List[dict] = []
        self.select_rows: List[dict] = []
        self.delete_result: List[dict] = [{"id": "corr-1"}]
    def table(self, name): return _FakeTable(self, name)
    def rpc(self, *a, **k):
        class E:
            def execute(self):
                class R: pass
                r = R(); r.data = None; return r
        return E()


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import data_router as data_mod
    from src.middleware import auth as auth_mod
    from src.services.audit_service import AuditService
    fake = _FakeSB(); getter = lambda: fake
    for m in (sbmod, data_mod, auth_mod):
        monkeypatch.setattr(m, "get_supabase", getter)
    monkeypatch.setattr(AuditService, "record", lambda self, **k: None)
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT,
                                         "scopes": ["read", "write"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_create_term_replace(client):
    c, fake = client
    r = c.post(f"/data/context/corrections?tenant_id={TENANT}&client_id={CLIENT}",
               json={"kind": "term_replace", "from": "Valid Technologies", "to": "NewName", "note": "rename"})
    assert r.status_code == 200, r.text
    assert r.json() == {"correction_id": "corr-1", "applied": True}
    row = fake.inserted[0]
    assert row["kind"] == "term_replace" and row["from_term"] == "Valid Technologies" and row["to_term"] == "NewName"
    assert row["applies_to"] == "all"


def test_create_disregard_without_to_ok(client):
    c, fake = client
    r = c.post(f"/data/context/corrections?tenant_id={TENANT}&client_id={CLIENT}",
               json={"kind": "disregard", "from": "Old Thing"})
    assert r.status_code == 200, r.text
    assert fake.inserted[0]["to_term"] is None


def test_term_replace_without_to_is_422(client):
    c, _ = client
    r = c.post(f"/data/context/corrections?tenant_id={TENANT}&client_id={CLIENT}",
               json={"kind": "term_replace", "from": "X"})
    assert r.status_code == 422


def test_bad_kind_is_422(client):
    c, _ = client
    r = c.post(f"/data/context/corrections?tenant_id={TENANT}&client_id={CLIENT}",
               json={"kind": "nuke", "from": "X"})
    assert r.status_code == 422


def test_list_corrections_returns_from_to(client):
    c, fake = client
    fake.select_rows = [
        {"id": "corr-1", "kind": "term_replace", "from_term": "Valid Technologies",
         "to_term": "NewName", "note": "rename", "applies_to": "all", "created_at": "2026-08-26T00:00:00Z"},
    ]
    r = c.get(f"/data/context/corrections?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 200, r.text
    item = r.json()["corrections"][0]
    assert item["from"] == "Valid Technologies" and item["to"] == "NewName"
    assert item["correction_id"] == "corr-1"


def test_delete_correction(client):
    c, fake = client
    r = c.delete(f"/data/context/corrections/corr-1?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": True, "correction_id": "corr-1"}


def test_delete_missing_is_404(client):
    c, fake = client
    fake.delete_result = []
    r = c.delete(f"/data/context/corrections/nope?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 404

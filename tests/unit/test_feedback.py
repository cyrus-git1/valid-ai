"""Unit test for the /feedback router (durable Vera feedback). Fakes Supabase."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-00000000bbbb"


class _Res:
    def __init__(self, data): self.data = data; self.count = None


class _Query:
    def __init__(self, store, table): self._store = store; self._table = table; self._insert = None
    def insert(self, row): self._insert = row; return self
    def execute(self):
        self._store.setdefault(self._table, []).append(self._insert)
        return _Res([self._insert])


class _FakeSB:
    def __init__(self): self.store: Dict[str, List[dict]] = {}
    def table(self, name): return _Query(self.store, name)


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import feedback_router as fmod
    from src.middleware import auth as auth_mod
    fake = _FakeSB(); getter = lambda: fake
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(fmod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT, "scopes": ["read", "write"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_record_feedback_inserts(client):
    c, fake = client
    r = c.post("/feedback", json={"rating": "up", "id": "f1", "tenant_id": TENANT, "comment": "great"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "f1" and body["persisted"] is True
    assert fake.store["vera_feedback"][0]["rating"] == "up"


def test_feedback_generates_id_when_missing(client):
    c, fake = client
    r = c.post("/feedback", json={"rating": "down"})
    assert r.status_code == 200
    assert r.json()["id"]  # server-generated uuid
    assert fake.store["vera_feedback"][0]["rating"] == "down"


def test_feedback_requires_auth(client):
    c, _ = client
    c.headers.pop("X-API-Key", None)
    assert c.post("/feedback", json={"rating": "up"}).status_code in (401, 403)

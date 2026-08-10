"""Unit tests for the admin-scoped /genomes router (harness genome storage).

Fakes Supabase (table query builder + rpc) so no DB is needed. Verifies the
admin gate, insert, reads, and that set-active forwards to the RPC."""
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
    def __init__(self, store: Dict[str, List[dict]], table: str):
        self._store = store
        self._table = table
        self._filters: Dict[str, Any] = {}
        self._insert = None

    def insert(self, row): self._insert = row; return self
    def select(self, *a, **k): return self
    def eq(self, col, val): self._filters[col] = val; return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self

    def execute(self):
        if self._insert is not None:
            self._store.setdefault(self._table, []).append(self._insert)
            return _Res([self._insert])
        rows = self._store.get(self._table, [])
        for col, val in self._filters.items():
            rows = [r for r in rows if str(r.get(col)) == str(val)]
        return _Res(rows)


class _FakeSB:
    def __init__(self):
        self.store: Dict[str, List[dict]] = {}
        self.rpc_calls: List[tuple] = []

    def table(self, name): return _Query(self.store, name)

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        return _Query(self.store, "__rpc__")  # .execute() returns empty

    def last_rpc(self, name):
        for n, p in reversed(self.rpc_calls):
            if n == name:
                return p
        raise AssertionError(name)


def _client(monkeypatch, scopes):
    from src.db import supabase_client as sbmod
    from src.routers import genomes_router as gmod
    from src.routers import admin_router as amod
    from src.middleware import auth as auth_mod
    fake = _FakeSB()
    getter = lambda: fake
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(gmod, "get_supabase", getter)
    monkeypatch.setattr(amod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)
    from src.services import api_key_service
    monkeypatch.setattr(
        api_key_service.ApiKeyService, "verify",
        lambda self, k: {"key_id": "k", "tenant_id": TENANT, "scopes": scopes,
                         "status": "active", "expired": False},
    )
    from src.main import app
    c = TestClient(app)
    c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


@pytest.fixture
def admin_client(monkeypatch):
    return _client(monkeypatch, ["read", "write", "admin"])


def _genome(step="survey_generation", version=1):
    return {
        "step_name": step, "version": version, "manager_prompt": "m",
        "rubric": [{"name": "relevance", "weight": 1.0, "description": "d"}],
        "agent_system_prompt": "a", "output_format_prompt": "o",
    }


def test_requires_admin_scope(monkeypatch):
    c, _ = _client(monkeypatch, ["read", "write"])          # no admin
    assert c.get("/genomes/survey_generation").status_code == 403


def test_save_then_read_active(admin_client):
    c, fake = admin_client
    assert c.post("/genomes", json=_genome(version=1)).status_code == 200
    assert fake.store["harness_genomes"][0]["is_active"] is False   # never auto-active
    # seed an active row and read it back
    fake.store["harness_genomes"].append({**_genome(version=2), "is_active": True})
    r = c.get("/genomes/survey_generation/active")
    assert r.status_code == 200 and r.json()["version"] == 2


def test_active_null_when_none(admin_client):
    c, _ = admin_client
    r = c.get("/genomes/survey_generation/active")
    assert r.status_code == 200 and r.json() is None


def test_get_missing_version_404(admin_client):
    c, _ = admin_client
    assert c.get("/genomes/survey_generation/version/99").status_code == 404


def test_set_active_forwards_to_rpc(admin_client):
    c, fake = admin_client
    fake.store["harness_genomes"] = [{**_genome(version=3), "is_active": False}]
    r = c.put("/genomes/survey_generation/active", json={"version": 3})
    assert r.status_code == 200
    p = fake.last_rpc("set_active_genome")
    assert p == {"p_step": "survey_generation", "p_version": 3}


def test_deactivate_via_null_version(admin_client):
    c, fake = admin_client
    r = c.put("/genomes/survey_generation/active", json={"version": None})
    assert r.status_code == 200 and r.json() is None
    assert fake.last_rpc("set_active_genome")["p_version"] is None

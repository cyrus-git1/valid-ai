"""Unit tests: /admin/reconcile — read-only embedding-drift report."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-0000000000cc"


class _FakeSB:
    def __init__(self):
        self._rpc: Dict[str, Any] = {}
        self._calls: Dict[str, List[dict]] = {}
    def set_rpc(self, n, d): self._rpc[n] = d
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


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import admin_router as admin_mod
    from src.middleware import auth as auth_mod
    fake = _FakeSB(); getter = lambda: fake
    for m in (sbmod, admin_mod, auth_mod):
        monkeypatch.setattr(m, "get_supabase", getter)
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT,
                                         "scopes": ["read", "admin"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_reconcile_returns_drift_counts(client):
    c, fake = client
    fake.set_rpc("reconcile_report", {
        "null_embedding": 3, "model_mismatch": 1, "content_drift": 4,
        "unknown_hash": 120, "total_vectorized": 200, "total_active": 203,
    })
    r = c.post("/admin/reconcile", json={"tenant_id": TENANT, "types": ["Observation", "Concept"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["null_embedding"] == 3
    assert body["model_mismatch"] == 1
    assert body["content_drift"] == 4
    assert body["unknown_hash"] == 120           # legacy rows not counted as drift
    assert body["total_vectorized"] == 200
    assert body["embedding_model"] == "text-embedding-3-small"
    # types + target model reached the RPC
    call = fake.calls("reconcile_report")[0]
    assert call["p_tenant_id"] == TENANT
    assert call["p_types"] == ["Observation", "Concept"]
    assert call["p_target_model"] == "text-embedding-3-small"


def test_reconcile_clean_tenant_all_zero(client):
    c, fake = client
    fake.set_rpc("reconcile_report", {
        "null_embedding": 0, "model_mismatch": 0, "content_drift": 0,
        "unknown_hash": 0, "total_vectorized": 50, "total_active": 50,
    })
    r = c.post("/admin/reconcile", json={"tenant_id": TENANT})
    assert r.status_code == 200
    body = r.json()
    assert body["null_embedding"] == body["model_mismatch"] == body["content_drift"] == 0
    assert fake.calls("reconcile_report")[0]["p_types"] is None   # None = all types


def test_reconcile_bad_tenant_400(client):
    c, _ = client
    r = c.post("/admin/reconcile", json={"tenant_id": "nope"})
    assert r.status_code == 400

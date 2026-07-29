"""Unit tests: /admin/retire-orphans + /admin/purge-study (dedup/purge sweeps)."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-00000000eeee"
STUDY = "00000000-0000-0000-0000-0000000000d1"


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
    from src.supabase import supabase_client as sbmod
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


# ── retire-orphans ──────────────────────────────────────────────────────────


def test_retire_orphans_returns_count_and_labels(client):
    c, fake = client
    fake.set_rpc("retire_orphan_concepts", {"retired": 2, "retired_labels": ["Purchase intent", "Price sensitivity"]})
    r = c.post("/admin/retire-orphans", json={"tenant_id": TENANT})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["retired"] == 2
    assert body["retired_labels"] == ["Purchase intent", "Price sensitivity"]
    assert fake.calls("retire_orphan_concepts")[0]["p_tenant_id"] == TENANT


def test_retire_orphans_drops_null_labels(client):
    c, fake = client
    fake.set_rpc("retire_orphan_concepts", {"retired": 2, "retired_labels": ["Good", None]})
    r = c.post("/admin/retire-orphans", json={"tenant_id": TENANT})
    assert r.status_code == 200
    assert r.json()["retired_labels"] == ["Good"]   # null label filtered, count untouched


def test_retire_orphans_noop(client):
    c, fake = client
    fake.set_rpc("retire_orphan_concepts", {"retired": 0, "retired_labels": []})
    r = c.post("/admin/retire-orphans", json={"tenant_id": TENANT})
    assert r.status_code == 200
    assert r.json() == {"retired": 0, "retired_labels": []}


def test_retire_orphans_bad_tenant_400(client):
    c, _ = client
    r = c.post("/admin/retire-orphans", json={"tenant_id": "nope"})
    assert r.status_code == 400


# ── purge-study ─────────────────────────────────────────────────────────────


def test_purge_study_returns_deleted_counts(client):
    c, fake = client
    fake.set_rpc("purge_study", {"observations_deleted": 8, "concepts_deleted": 5})
    r = c.post("/admin/purge-study", json={"tenant_id": TENANT, "study_id": STUDY})
    assert r.status_code == 200, r.text
    assert r.json() == {"observations_deleted": 8, "concepts_deleted": 5}
    call = fake.calls("purge_study")[0]
    assert call["p_tenant_id"] == TENANT and call["p_study_id"] == STUDY


def test_purge_study_idempotent_noop(client):
    c, fake = client
    fake.set_rpc("purge_study", {"observations_deleted": 0, "concepts_deleted": 0})
    r = c.post("/admin/purge-study", json={"tenant_id": TENANT, "study_id": STUDY})
    assert r.status_code == 200
    assert r.json() == {"observations_deleted": 0, "concepts_deleted": 0}


def test_purge_study_bad_study_400(client):
    c, _ = client
    r = c.post("/admin/purge-study", json={"tenant_id": TENANT, "study_id": "nope"})
    assert r.status_code == 400

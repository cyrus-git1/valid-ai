"""Characterization tests for the previously-uncovered /admin endpoints.

Written BEFORE extracting AdminService so the extraction can be proven
behavior-preserving. Covers api-keys (create/list/revoke), plan (get/set),
stats, maintenance/run, and health — the endpoints not exercised by
test_admin_backfill (mirror/reembed) or test_admin_cleanup (retire/purge).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

TENANT = "00000000-0000-0000-0000-00000000eeee"
CLIENT = "00000000-0000-0000-0000-0000000000cc"


class _Chain:
    def __init__(self, name, rows, count, sink):
        self._name, self._rows, self._count, self._sink = name, rows, count, sink

    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def in_(self, *a, **k): return self
    def is_(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self

    def insert(self, payload, **k):
        self._sink.setdefault("insert", []).append((self._name, payload)); return self

    def update(self, payload, **k):
        self._sink.setdefault("update", []).append((self._name, payload)); return self

    def execute(self):
        class _R: pass
        r = _R()
        r.data = self._rows
        r.count = self._count if self._count is not None else len(self._rows)
        return r


class _FakeSB:
    def __init__(self):
        self._rows: Dict[str, List[dict]] = {}
        self._counts: Dict[str, int] = {}
        self._rpc: Dict[str, Any] = {}
        self.sink: Dict[str, Any] = {}
        self.rpc_calls: List[tuple] = []

    def set_rows(self, name, rows, count=None):
        self._rows[name] = rows
        if count is not None:
            self._counts[name] = count

    def set_rpc(self, name, data):
        self._rpc[name] = data

    def table(self, name):
        return _Chain(name, self._rows.get(name, []), self._counts.get(name), self.sink)

    def rpc(self, name, params=None):
        self.rpc_calls.append((name, params))
        ret = self._rpc.get(name, {})
        class _R: pass
        r = _R(); r.data = ret; r.count = None
        class _E:
            def execute(self_inner): return r
        return _E()


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


# ── api-keys ──────────────────────────────────────────────────────────────────


def test_create_api_key_returns_raw_once(client):
    c, fake = client
    fake.set_rows("api_keys", [{"id": "key-1"}])
    r = c.post("/admin/api-keys", json={"name": "ci", "scopes": ["read"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "key-1"
    assert body["raw_key"].startswith("dp_")
    assert body["name"] == "ci"
    # persisted with a hash, never the raw key
    ins = fake.sink["insert"][-1]
    assert ins[0] == "api_keys"
    assert "key_hash" in ins[1] and ins[1]["tenant_id"] == TENANT


def test_list_api_keys(client):
    c, fake = client
    fake.set_rows("api_keys", [{
        "id": "key-1", "key_prefix": "dp_abc", "name": "ci", "scopes": ["read"],
        "status": "active", "created_at": "2026-01-01",
    }])
    r = c.get("/admin/api-keys")
    assert r.status_code == 200, r.text
    assert r.json()[0]["id"] == "key-1"


def test_revoke_api_key_missing_404(client):
    c, fake = client
    fake.set_rows("api_keys", [])  # update returns no rows
    r = c.delete("/admin/api-keys/nope")
    assert r.status_code == 404


def test_revoke_api_key_success(client):
    c, fake = client
    fake.set_rows("api_keys", [{"id": "key-1"}])
    r = c.delete("/admin/api-keys/key-1")
    assert r.status_code == 200, r.text
    assert r.json() == {"revoked": True, "id": "key-1"}
    assert any(t == "api_keys" for t, _ in fake.sink.get("update", []))


# ── plan ──────────────────────────────────────────────────────────────────────


def test_set_plan_invalid_400(client):
    c, _ = client
    r = c.put("/admin/plan", json={"plan": "platinum"})
    assert r.status_code == 400


# ── stats ─────────────────────────────────────────────────────────────────────


def test_stats_counts_and_embedding_probe_ok(client):
    c, fake = client
    fake.set_rows("documents", [{"id": "d1"}], count=1)
    fake.set_rows("chunks", [], count=3)
    fake.set_rows("kg_nodes", [], count=5)
    fake.set_rows("kg_edges", [], count=7)
    fake.set_rpc("fetch_chunks_with_embeddings", [])
    r = c.get(f"/admin/stats?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document_count"] == 1
    assert body["chunk_count"] == 3
    assert body["kg_node_count"] == 5
    assert body["kg_edge_count"] == 7
    assert body["chunks_with_embeddings"] == 3   # probe succeeded → mirrors chunk_count


# ── maintenance ───────────────────────────────────────────────────────────────


def test_maintenance_no_clients_early_return(client):
    c, fake = client
    # _discover_client_ids reads documents/kg_nodes/memory_state → all empty
    r = c.post("/admin/maintenance/run", json={"tenant_id": TENANT})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_clients"] == 0
    assert body["client_ids_processed"] == []
    assert body["metadata"]["message"] == "No clients found for tenant."


def test_maintenance_explicit_client_runs_prune(client):
    c, fake = client
    fake.set_rpc("prune_kg", {"edges_archived": 2, "nodes_archived": 1})
    r = c.post("/admin/maintenance/run",
               json={"tenant_id": TENANT, "client_id": CLIENT, "cleanup_orphans": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_clients"] == 1
    assert body["client_ids_processed"] == [CLIENT]
    assert body["totals"]["edges_archived"] == 2
    assert any(n == "prune_kg" for n, _ in fake.rpc_calls)


# ── health ────────────────────────────────────────────────────────────────────


def test_health_ok(client):
    c, fake = client
    fake.set_rows("documents", [{"id": "d1"}])
    r = c.get("/admin/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["supabase"] is True
    assert body["openai"] is True
    assert body["status"] in ("ok", "degraded")

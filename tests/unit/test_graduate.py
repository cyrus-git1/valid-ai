"""Unit tests: POST /concepts/graduate — one-call create-tag + link-in-place,
idempotent on (tenant, label). The whole graduation write, server-side."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-00000000dddd"
CONCEPT = "00000000-0000-0000-0000-0000000000c1"
TAG = "00000000-0000-0000-0000-0000000000e1"


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
    from src.routers import spine_router as spine_mod
    from src.middleware import auth as auth_mod
    fake = _FakeSB(); getter = lambda: fake
    for m in (sbmod, spine_mod, auth_mod):
        monkeypatch.setattr(m, "get_supabase", getter)
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT,
                                         "scopes": ["read", "write"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_graduate_creates_tag_and_links(client):
    c, fake = client
    fake.set_rpc("graduate_concept", {
        "tag_id": TAG, "concept_id": CONCEPT, "external_ref": TAG, "created": True,
    })
    r = c.post("/concepts/graduate", json={
        "tenant_id": TENANT, "concept_id": CONCEPT,
        "label": "Checkout abandonment", "description": "users leaving checkout",
        "cluster_id": "clust-7", "evidence_ids": ["obs-1", "obs-2"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "graduated"    # success marker the agent keys on
    assert body["tag_id"] == TAG
    assert body["external_ref"] == TAG      # governed: node now carries the tag uuid
    assert body["concept_id"] == CONCEPT
    assert body["created"] is True
    # the whole write went through one RPC, carrying provenance
    call = fake.calls("graduate_concept")[0]
    assert call["p_tenant_id"] == TENANT
    assert call["p_concept_id"] == CONCEPT
    assert call["p_label"] == "Checkout abandonment"
    assert call["p_cluster_id"] == "clust-7"
    assert call["p_evidence_ids"] == ["obs-1", "obs-2"]


def test_graduate_idempotent_resweep_returns_created_false(client):
    c, fake = client
    fake.set_rpc("graduate_concept", {
        "tag_id": TAG, "concept_id": CONCEPT, "external_ref": TAG, "created": False,
    })
    r = c.post("/concepts/graduate", json={
        "tenant_id": TENANT, "concept_id": CONCEPT, "label": "Checkout abandonment",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] is False          # re-sweep didn't duplicate the codebook row
    assert body["tag_id"] == TAG
    # evidence omitted → empty list passed, not null
    assert fake.calls("graduate_concept")[0]["p_evidence_ids"] == []


def test_graduate_requires_label(client):
    c, _ = client
    r = c.post("/concepts/graduate", json={"tenant_id": TENANT, "concept_id": CONCEPT, "label": ""})
    assert r.status_code == 422   # min_length=1


def test_graduate_tenant_mismatch_403(client):
    c, _ = client
    other = "00000000-0000-0000-0000-0000000ffff0"
    r = c.post("/concepts/graduate", json={"tenant_id": other, "concept_id": CONCEPT, "label": "X"})
    assert r.status_code == 403   # body tenant != authed key tenant

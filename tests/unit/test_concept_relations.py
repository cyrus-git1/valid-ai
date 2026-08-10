"""Unit tests for signed concept relations: /concepts/relations{/compute,}."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-00000000bbbb"
STUDY = "00000000-0000-0000-0000-0000000000a1"
A = "00000000-0000-0000-0000-0000000000c1"
B = "00000000-0000-0000-0000-0000000000c2"


class _FakeSB:
    def __init__(self):
        self.rpc_calls: List[tuple] = []
        self._next: Dict[str, Any] = {}

    def set_rpc(self, name, data): self._next[name] = data

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        ret = self._next.get(name, [])
        class _R: pass
        r = _R(); r.data = ret; r.count = None
        class _E:
            def __init__(s, r): s._r = r
            def execute(s): return s._r
        return _E(r)

    def last(self, name):
        for n, p in reversed(self.rpc_calls):
            if n == name: return p
        raise AssertionError(name)


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import spine_router as spine_mod
    from src.middleware import auth as auth_mod
    fake = _FakeSB(); getter = lambda: fake
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(spine_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT, "scopes": ["read", "write"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_compute_relations(client):
    c, fake = client
    fake.set_rpc("compute_concept_relations", {"relations_written": 3})
    r = c.post("/concepts/relations/compute", json={"tenant_id": TENANT, "study_ids": [STUDY], "min_cooccur": 2})
    assert r.status_code == 200, r.text
    assert r.json()["relations_written"] == 3
    p = fake.last("compute_concept_relations")
    assert p["p_study_ids"] == [STUDY] and p["p_min_cooccur"] == 2


def test_read_relations(client):
    c, fake = client
    fake.set_rpc("concept_relations", [
        {"src_concept_id": A, "dst_concept_id": B, "rel_type": "contradicts", "weight": 4.0},
        {"src_concept_id": A, "dst_concept_id": B, "rel_type": "supports", "weight": 1.0},
    ])
    r = c.get(f"/concepts/relations?tenant_id={TENANT}&study_ids={STUDY}&rel_types=contradicts&rel_types=supports")
    assert r.status_code == 200, r.text
    rels = r.json()
    assert rels[0]["rel_type"] == "contradicts" and rels[0]["weight"] == 4.0
    assert rels[0]["src_concept_id"] == A and rels[0]["dst_concept_id"] == B
    assert fake.last("concept_relations")["p_rel_types"] == ["contradicts", "supports"]


def test_relations_tenant_mismatch_403(client):
    c, _ = client
    assert c.get(f"/concepts/relations?tenant_id=00000000-0000-0000-0000-00000000dddd").status_code == 403

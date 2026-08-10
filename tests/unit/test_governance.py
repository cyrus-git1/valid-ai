"""Unit tests for theme-taxonomy governance: external_ref on reads + mirror/link."""
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
CAND = "00000000-0000-0000-0000-0000000000c9"


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
    from src.services import spine_service as spine_svc_mod
    from src.middleware import auth as auth_mod
    fake = _FakeSB(); getter = lambda: fake
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(spine_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)
    monkeypatch.setattr(spine_svc_mod, "_embed_in_batches", lambda texts, **kw: [[0.1] * 1536 for _ in texts])
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT, "scopes": ["read", "write"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_nearest_returns_external_ref(client):
    c, fake = client
    fake.set_rpc("nearest_concepts", [
        {"id": "gov", "canonical_id": "cid", "canonical_label": "Onboarding", "external_ref": "tag-123", "similarity": 0.9, "final_score": 0.95},
        {"id": "cand", "canonical_id": "cid2", "canonical_label": "Setup pain", "external_ref": None, "similarity": 0.8, "final_score": 0.8},
    ])
    r = c.post("/concepts/nearest", json={"tenant_id": TENANT, "query_text": "onboarding"})
    assert r.status_code == 200, r.text
    cands = r.json()["candidates"]
    assert cands[0]["external_ref"] == "tag-123"   # governed
    assert cands[1]["external_ref"] is None          # candidate


def test_by_study_returns_external_ref(client):
    c, fake = client
    fake.set_rpc("concepts_by_study", [{"concept_id": "gov", "label": "Onboarding", "external_ref": "tag-123"}])
    r = c.get(f"/concepts/by-study?tenant_id={TENANT}&study_ids={STUDY}")
    assert r.status_code == 200
    assert r.json()["concepts"][0]["external_ref"] == "tag-123"


def test_mirror_tag(client):
    c, fake = client
    fake.set_rpc("mirror_tag_concept", {"concept_id": "11111111-1111-1111-1111-111111111111",
                                        "external_ref": "11111111-1111-1111-1111-111111111111",
                                        "node_key": "concept:11111111-1111-1111-1111-111111111111"})
    r = c.post("/concepts/mirror-tag", json={"tenant_id": TENANT, "tag_id": "11111111-1111-1111-1111-111111111111",
                                             "label": "Onboarding friction", "description": "users give up during setup"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["concept_id"] == body["external_ref"]   # governed concept_id IS the tag id
    p = fake.last("mirror_tag_concept")
    assert p["p_tag_id"] == "11111111-1111-1111-1111-111111111111"
    assert p["p_label"] == "Onboarding friction"
    assert len(p["p_embedding"]) == 1536                 # embedded label+description


def test_link_tag_graduation(client):
    c, fake = client
    fake.set_rpc("link_concept_tag", {"linked": True, "concept_id": CAND, "external_ref": "tag-777"})
    r = c.post("/concepts/link-tag", json={"tenant_id": TENANT, "concept_id": CAND, "tag_id": "00000000-0000-0000-0000-000000000777"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["linked"] is True
    assert body["external_ref"] == "tag-777"
    p = fake.last("link_concept_tag")
    assert p["p_concept_id"] == CAND

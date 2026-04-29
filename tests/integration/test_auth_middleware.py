"""
End-to-end auth middleware checks using FastAPI's TestClient.

Monkeypatches ApiKeyService.verify so the middleware has a deterministic
auth oracle that doesn't require a live Supabase.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"   # auth tests require auth on
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")


@pytest.fixture
def client(monkeypatch):
    # Patch ApiKeyService.verify before importing the app
    from src.services import api_key_service

    def _fake_verify(self, raw_key: str):
        from src.services.api_key_service import (
            ApiKeyNotFound,
            ApiKeyRevoked,
        )
        if raw_key == "dp_good_key":
            return {
                "key_id": "00000000-0000-0000-0000-00000000aaaa",
                "tenant_id": "00000000-0000-0000-0000-00000000bbbb",
                "scopes": ["read", "write"],
                "status": "active",
                "expired": False,
            }
        if raw_key == "dp_revoked_key":
            raise ApiKeyRevoked("revoked")
        raise ApiKeyNotFound("unknown")

    monkeypatch.setattr(api_key_service.ApiKeyService, "verify", _fake_verify)

    # Now import the app — middleware will use the patched verify
    from src.main import app
    return TestClient(app)


def test_missing_api_key_returns_401(client):
    r = client.get("/admin/stats?tenant_id=00000000-0000-0000-0000-00000000bbbb&client_id=00000000-0000-0000-0000-00000000cccc")
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["code"] == "auth.missing_key"
    assert "request_id" in body["error"]
    # Response must also echo the request-id header
    assert "X-Request-ID" in r.headers


def test_invalid_api_key_returns_401(client):
    r = client.get(
        "/admin/stats?tenant_id=00000000-0000-0000-0000-00000000bbbb&client_id=00000000-0000-0000-0000-00000000cccc",
        headers={"X-API-Key": "dp_nope"},
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "auth.invalid_key"


def test_revoked_api_key_returns_401(client):
    r = client.get(
        "/admin/stats?tenant_id=00000000-0000-0000-0000-00000000bbbb&client_id=00000000-0000-0000-0000-00000000cccc",
        headers={"X-API-Key": "dp_revoked_key"},
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "auth.revoked_key"


def test_tenant_mismatch_in_query_returns_403(client):
    # Good key belongs to tenant bbbb; request tenant is ffff → reject
    r = client.get(
        "/admin/stats?tenant_id=00000000-0000-0000-0000-00000000ffff&client_id=00000000-0000-0000-0000-00000000cccc",
        headers={"X-API-Key": "dp_good_key"},
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "auth.tenant_mismatch"


def test_health_bypass_allows_unauthenticated(client):
    r = client.get("/admin/health")
    # Supabase may be unreachable in test, but the endpoint must not 401
    assert r.status_code != 401


def test_root_bypass_allows_unauthenticated(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "Valid Data Plane"


def test_request_id_propagates(client):
    r = client.get("/", headers={"X-Request-ID": "my-custom-id-123"})
    assert r.headers.get("X-Request-ID") == "my-custom-id-123"

"""Unit tests for ApiKeyService."""
from __future__ import annotations

import pytest

from src.services.api_key_service import (
    ApiKeyNotFound,
    ApiKeyRevoked,
    ApiKeyService,
    hash_key,
    key_prefix,
)


def test_hash_key_deterministic():
    assert hash_key("dp_abc") == hash_key("dp_abc")
    assert hash_key("dp_abc") != hash_key("dp_abd")


def test_key_prefix_first_eight():
    assert key_prefix("dp_abc12345extra") == "dp_abc12"


def test_verify_raises_on_missing_key(fake_supabase):
    svc = ApiKeyService(fake_supabase)
    svc._redis = None  # force DB path
    with pytest.raises(ApiKeyNotFound):
        svc.verify("")


def test_verify_returns_entry(monkeypatch, fake_supabase):
    # Seed one active row into the fake table
    raw = "dp_activekey_12345"
    fake_supabase.tables["api_keys"] = []
    svc = ApiKeyService(fake_supabase)
    svc._redis = None

    # Patch the chain to return a row matching our hash
    original_table = fake_supabase.table

    def fake_table(name):
        row = {
            "id": "00000000-0000-0000-0000-000000000abc",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "scopes": ["read", "write"],
            "status": "active",
            "expires_at": None,
        }
        from tests.conftest import _FakeQuery
        return _FakeQuery([row])

    monkeypatch.setattr(fake_supabase, "table", fake_table)
    entry = svc.verify(raw)
    assert entry["tenant_id"] == "00000000-0000-0000-0000-000000000001"
    assert "read" in entry["scopes"]


def test_verify_revoked_raises(monkeypatch, fake_supabase):
    svc = ApiKeyService(fake_supabase)
    svc._redis = None

    def fake_table(name):
        from tests.conftest import _FakeQuery
        return _FakeQuery([{
            "id": "00000000-0000-0000-0000-000000000002",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "scopes": [],
            "status": "revoked",
            "expires_at": None,
        }])

    monkeypatch.setattr(fake_supabase, "table", fake_table)
    with pytest.raises(ApiKeyRevoked):
        svc.verify("dp_revoked")

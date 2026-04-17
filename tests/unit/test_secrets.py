"""Unit tests for secrets accessor."""
from __future__ import annotations

import pytest

from src.config import secrets


def setup_function():
    secrets.clear_cache()


def test_env_backend_returns_value(monkeypatch):
    monkeypatch.setenv("SECRETS_BACKEND", "env")
    monkeypatch.setenv("MY_SECRET", "hello")
    assert secrets.get_secret("MY_SECRET") == "hello"


def test_missing_required_raises(monkeypatch):
    monkeypatch.setenv("SECRETS_BACKEND", "env")
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(secrets.SecretNotFound):
        secrets.get_secret("DEFINITELY_NOT_SET", required=True)


def test_default_used_when_missing(monkeypatch):
    monkeypatch.setenv("SECRETS_BACKEND", "env")
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    assert secrets.get_secret("DEFINITELY_NOT_SET", default="fallback") == "fallback"


def test_cache_hit_avoids_backend(monkeypatch):
    monkeypatch.setenv("SECRETS_BACKEND", "env")
    monkeypatch.setenv("CACHED_SECRET", "first")
    assert secrets.get_secret("CACHED_SECRET") == "first"
    # Mutate env — cached value should persist (TTL=0 by default)
    monkeypatch.setenv("CACHED_SECRET", "second")
    assert secrets.get_secret("CACHED_SECRET") == "first"
    secrets.clear_cache()
    assert secrets.get_secret("CACHED_SECRET") == "second"


def test_unknown_backend_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("SECRETS_BACKEND", "made-up")
    monkeypatch.setenv("STILL_READS", "ok")
    assert secrets.get_secret("STILL_READS") == "ok"

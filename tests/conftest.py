"""
Shared pytest fixtures. Integration tests require Supabase + Redis; mark them
with @pytest.mark.integration and skip by default in CI (override with
--run-integration via a custom pytest option if you add one).
"""
from __future__ import annotations

import os
from typing import Any

import pytest

# Make sure the data plane package is importable
os.environ.setdefault("SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "dummy-service-key")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")


class _FakeQuery:
    """Minimal stub of supabase-py chained query builder. Returns .execute().data."""

    def __init__(self, data: Any = None, count: int | None = None):
        self._data = data if data is not None else []
        self._count = count

    def select(self, *_a, **_kw):
        return self

    def insert(self, payload, **_kw):
        row = payload if isinstance(payload, list) else [payload]
        # Populate id if missing
        for r in row:
            r.setdefault("id", "00000000-0000-0000-0000-000000000001")
        self._data = row
        return self

    def update(self, _payload, **_kw):
        return self

    def delete(self, *_a, **_kw):
        return self

    def upsert(self, _payload, **_kw):
        return self

    def eq(self, *_a, **_kw):
        return self

    def in_(self, *_a, **_kw):
        return self

    def gte(self, *_a, **_kw):
        return self

    def is_(self, *_a, **_kw):
        return self

    def order(self, *_a, **_kw):
        return self

    def limit(self, *_a, **_kw):
        return self

    def execute(self):
        class _R:
            pass

        r = _R()
        r.data = self._data
        r.count = self._count
        return r


class FakeSupabase:
    def __init__(self):
        self.tables: dict[str, list[dict]] = {}
        self.rpc_calls: list[tuple[str, dict]] = []

    def table(self, name: str):
        return _FakeQuery(self.tables.get(name, []))

    def rpc(self, name: str, params: dict):
        self.rpc_calls.append((name, params))
        return _FakeQuery({})


@pytest.fixture
def fake_supabase() -> FakeSupabase:
    return FakeSupabase()

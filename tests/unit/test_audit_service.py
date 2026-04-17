"""Unit tests for AuditService."""
from __future__ import annotations

from types import SimpleNamespace

from src.services.audit_service import AuditService


class _CapturingSupabase:
    def __init__(self):
        self.inserted = []
        self._table_name = None

    def table(self, name):
        self._table_name = name
        return self

    def insert(self, row):
        self.inserted.append((self._table_name, row))
        return self

    def execute(self):
        class _R:
            data = []
        return _R()


def _fake_request(*, tenant_id="00000000-0000-0000-0000-0000000000aa",
                  key_id="00000000-0000-0000-0000-0000000000bb",
                  request_id="req-123"):
    state = SimpleNamespace(tenant_id=tenant_id, key_id=key_id, request_id=request_id)
    return SimpleNamespace(state=state, client=SimpleNamespace(host="127.0.0.1"))


def test_record_inserts_row():
    sb = _CapturingSupabase()
    AuditService(sb).record(
        request=_fake_request(),
        action="document.patch",
        resource_type="document",
        resource_id="doc-1",
        metadata={"updates": {"status": "archived"}},
    )
    assert len(sb.inserted) == 1
    table, row = sb.inserted[0]
    assert table == "audit_log"
    assert row["action"] == "document.patch"
    assert row["resource_id"] == "doc-1"
    assert row["request_id"] == "req-123"
    assert row["tenant_id"].startswith("00000000")


def test_record_skipped_when_no_tenant():
    sb = _CapturingSupabase()
    AuditService(sb).record(
        request=_fake_request(tenant_id=None),
        action="document.patch",
    )
    assert sb.inserted == []


def test_record_never_raises_on_db_failure():
    class _BoomSupabase:
        def table(self, name):
            raise RuntimeError("db down")

    # Must not raise
    AuditService(_BoomSupabase()).record(
        request=_fake_request(),
        action="document.patch",
    )

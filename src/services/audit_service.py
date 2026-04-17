"""
Audit log service — append-only record of who did what.

All writes are best-effort: a failure to log NEVER blocks the underlying
operation. Actor identity (tenant_id, key_id) and request_id are read from
request.state (set by auth + request_id middleware).
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import Request
from supabase import Client

from src.logging_config import get_logger

logger = get_logger(__name__)


class AuditService:
    def __init__(self, supabase: Client):
        self.sb = supabase

    def record(
        self,
        *,
        request: Optional[Request],
        action: str,
        status: str = "success",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        key_id: Optional[str] = None,
    ) -> None:
        """Insert an audit row. Never raises."""
        try:
            state = getattr(request, "state", None) if request else None
            t_id = tenant_id or (getattr(state, "tenant_id", None) if state else None)
            k_id = key_id or (getattr(state, "key_id", None) if state else None)
            req_id = getattr(state, "request_id", "-") if state else "-"
            source_ip = None
            if request is not None:
                client = getattr(request, "client", None)
                source_ip = client.host if client else None

            if not t_id:
                logger.debug("audit_skipped_no_tenant", action=action)
                return

            self.sb.table("audit_log").insert({
                "tenant_id": str(t_id),
                "key_id": str(k_id) if k_id else None,
                "request_id": str(req_id),
                "action": action,
                "resource_type": resource_type,
                "resource_id": str(resource_id) if resource_id is not None else None,
                "status": status,
                "metadata": metadata or {},
                "source_ip": source_ip,
            }).execute()
        except Exception as e:
            logger.warning("audit_write_failed", action=action, error=str(e))

"""
/memory router
--------------
Read-only exposure of the memory-state version + append-only change log that
memory_state / memory_change_log (migrations 18/20) already track. Powers the
agent's What-changed board and scope_fresh_at.

  GET /memory/state    ?tenant_id&client_id?          -> { memory_version, last_changed_at }
  GET /memory/changes  ?tenant_id&since_version&client_id? -> { changes:[…], current_version }

Reads over existing tables; no schema change. Tenant-scoped by the query param
(same open posture as the rest of the data plane while auth is off).
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from src.models.api.memory import MemoryChange, MemoryChangesResponse, MemoryStateResponse
from src.logging_config import get_logger
from src.services.memory_state_service import MemoryStateService
from src.db.supabase_client import get_supabase

logger = get_logger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/state", response_model=MemoryStateResponse)
def memory_state(
    tenant_id: UUID = Query(...),
    client_id: Optional[UUID] = Query(None),
) -> MemoryStateResponse:
    """Current memory version + last-changed for a scope. The app caches this as scope_fresh_at."""
    sb = get_supabase()
    st = MemoryStateService(sb).get_state(tenant_id=tenant_id, client_id=client_id)
    return MemoryStateResponse(
        memory_version=int(st.get("memory_version") or 0),
        last_changed_at=str(st["last_changed_at"]) if st.get("last_changed_at") else None,
    )


@router.get("/changes", response_model=MemoryChangesResponse)
def memory_changes(
    tenant_id: UUID = Query(...),
    since_version: int = Query(0, ge=0),
    client_id: Optional[UUID] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
) -> MemoryChangesResponse:
    """Change-log rows after `since_version` (real change events for the What-changed board)."""
    sb = get_supabase()
    try:
        q = (
            sb.table("memory_change_log")
            .select("change_type, memory_version, metadata, created_at")
            .eq("tenant_id", str(tenant_id))
            .gt("memory_version", since_version)
            .order("memory_version", desc=False)
            .limit(limit)
        )
        if client_id is not None:
            q = q.eq("client_id", str(client_id))
        rows = q.execute().data or []
    except Exception as e:
        logger.exception("memory_changes read failed")
        raise HTTPException(status_code=500, detail=str(e))

    current = MemoryStateService(sb).get_state(tenant_id=tenant_id, client_id=client_id)
    changes = [
        MemoryChange(
            change_type=r.get("change_type", ""),
            memory_version=int(r.get("memory_version") or 0),
            metadata=r.get("metadata") or {},
            created_at=str(r["created_at"]) if r.get("created_at") else None,
        )
        for r in rows
    ]
    return MemoryChangesResponse(changes=changes, current_version=int(current.get("memory_version") or 0))

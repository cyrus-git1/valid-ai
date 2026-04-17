"""
/admin router
-------------
Operational endpoints — health and stats.

GET  /admin/health — Liveness check (Supabase + OpenAI reachable)
GET  /admin/stats  — Document/chunk/node/edge counts for a client

Reindex and rebuild-kg have moved to the agent service.
"""
from __future__ import annotations

import logging
import os
from uuid import UUID

import secrets
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.supabase.supabase_client import get_supabase
from src.services.memory_state_service import MemoryStateService
from src.services.api_key_service import hash_key, key_prefix
from src.models.api.admin import (
    HealthResponse,
    MaintenanceClientResult,
    MaintenanceRunRequest,
    MaintenanceRunResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def _discover_client_ids(sb, tenant_id: UUID) -> list[str]:
    client_ids: set[str] = set()
    for table in ("documents", "kg_nodes", "memory_state"):
        try:
            res = (
                sb.table(table)
                .select("client_id")
                .eq("tenant_id", str(tenant_id))
                .execute()
            )
            for row in (res.data or []):
                client_id = row.get("client_id")
                if client_id:
                    client_ids.add(client_id)
        except Exception as e:
            logger.warning("Failed to discover client_ids from %s for tenant=%s: %s", table, tenant_id, e)
    return sorted(client_ids)


# ── API key management ────────────────────────────────────────────────────────


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    scopes: List[str] = Field(default_factory=list)
    expires_at: Optional[str] = None


class ApiKeyCreateResponse(BaseModel):
    id: str
    raw_key: str = Field(description="Shown once — store it now.")
    key_prefix: str
    name: str
    scopes: List[str]


class ApiKeyListItem(BaseModel):
    id: str
    key_prefix: str
    name: str
    scopes: List[str]
    status: str
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: str
    revoked_at: Optional[str] = None


def _require_admin(request: Request) -> str:
    scopes = getattr(request.state, "scopes", []) or []
    if "admin" not in scopes:
        raise HTTPException(status_code=403, detail="admin scope required")
    return getattr(request.state, "tenant_id")


@router.post("/api-keys", response_model=ApiKeyCreateResponse)
def create_api_key(body: ApiKeyCreateRequest, request: Request) -> ApiKeyCreateResponse:
    """Issue a new API key for the caller's tenant. Raw key returned ONCE."""
    tenant_id = _require_admin(request)
    sb = get_supabase()

    # Generate 32 bytes of entropy → 43-char urlsafe token
    raw = "dp_" + secrets.token_urlsafe(32)
    res = sb.table("api_keys").insert({
        "tenant_id": str(tenant_id),
        "key_hash": hash_key(raw),
        "key_prefix": key_prefix(raw),
        "name": body.name,
        "scopes": body.scopes,
        "status": "active",
        "expires_at": body.expires_at,
    }).execute()
    row = (res.data or [{}])[0]

    logger.info(
        "admin.api_key.create tenant=%s key_id=%s prefix=%s scopes=%s",
        tenant_id, row.get("id"), key_prefix(raw), body.scopes,
    )
    return ApiKeyCreateResponse(
        id=row.get("id", ""),
        raw_key=raw,
        key_prefix=key_prefix(raw),
        name=body.name,
        scopes=body.scopes,
    )


@router.get("/api-keys", response_model=List[ApiKeyListItem])
def list_api_keys(request: Request) -> List[ApiKeyListItem]:
    """List API keys for the caller's tenant (prefixes only, never raw)."""
    tenant_id = _require_admin(request)
    sb = get_supabase()
    res = (
        sb.table("api_keys")
        .select("id, key_prefix, name, scopes, status, last_used_at, expires_at, created_at, revoked_at")
        .eq("tenant_id", str(tenant_id))
        .order("created_at", desc=True)
        .execute()
    )
    return [ApiKeyListItem(**row) for row in (res.data or [])]


@router.delete("/api-keys/{key_id}")
def revoke_api_key(key_id: str, request: Request) -> dict:
    """Revoke an API key."""
    tenant_id = _require_admin(request)
    sb = get_supabase()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    res = (
        sb.table("api_keys")
        .update({"status": "revoked", "revoked_at": now})
        .eq("id", key_id)
        .eq("tenant_id", str(tenant_id))
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="API key not found")
    logger.info("admin.api_key.revoke tenant=%s key_id=%s", tenant_id, key_id)
    return {"revoked": True, "id": key_id}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + dependency check."""
    sb_ok = False
    detail = None

    try:
        sb = get_supabase()
        sb.table("documents").select("id").limit(1).execute()
        sb_ok = True
    except Exception as e:
        detail = f"Supabase unreachable: {e}"
        logger.error(detail)

    openai_ok = bool(os.environ.get("OPENAI_API_KEY"))
    if not openai_ok:
        detail = (detail or "") + " OPENAI_API_KEY missing."

    overall = "ok" if (sb_ok and openai_ok) else "degraded"

    return HealthResponse(
        status=overall,
        supabase=sb_ok,
        openai=openai_ok,
        detail=detail,
    )


@router.get("/stats", response_model=StatsResponse)
def stats(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> StatsResponse:
    """Document, chunk, KG node, and KG edge counts for a tenant+client."""
    sb = get_supabase()
    memory_state = MemoryStateService(sb)

    def _count(table: str, filters: dict) -> int:
        q = sb.table(table).select("id", count="exact")
        for col, val in filters.items():
            q = q.eq(col, val)
        return q.execute().count or 0

    doc_count = _count("documents", {"tenant_id": str(tenant_id), "client_id": str(client_id)})
    doc_ids_res = (
        sb.table("documents")
        .select("id")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .execute()
    )
    doc_ids = [row["id"] for row in (doc_ids_res.data or [])]
    if doc_ids:
        chunk_count = (
            sb.table("chunks")
            .select("id", count="exact")
            .eq("tenant_id", str(tenant_id))
            .in_("document_id", doc_ids)
            .execute()
            .count
            or 0
        )
    else:
        chunk_count = 0

    try:
        emb_res = sb.rpc(
            "fetch_chunks_with_embeddings",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_document_id": None,
                "p_limit": 1,
                "p_offset": 0,
            },
        ).execute()
        chunks_with_embeddings = chunk_count
    except Exception:
        chunks_with_embeddings = -1

    node_count = _count("kg_nodes", {"tenant_id": str(tenant_id), "client_id": str(client_id)})
    edge_count = _count("kg_edges", {"tenant_id": str(tenant_id), "client_id": str(client_id)})
    client_state = memory_state.get_state(tenant_id=tenant_id, client_id=client_id)
    tenant_state = memory_state.get_state(tenant_id=tenant_id, client_id=None)

    return StatsResponse(
        tenant_id=str(tenant_id),
        client_id=str(client_id),
        document_count=doc_count,
        chunk_count=chunk_count,
        chunks_with_embeddings=chunks_with_embeddings,
        kg_node_count=node_count,
        kg_edge_count=edge_count,
        client_memory_version=int(client_state.get("memory_version") or 0),
        tenant_memory_version=int(tenant_state.get("memory_version") or 0),
        client_last_ingested_at=client_state.get("last_ingested_at"),
        client_last_changed_at=client_state.get("last_changed_at"),
        client_last_summary_at=client_state.get("last_summary_at"),
    )


@router.post("/maintenance/run", response_model=MaintenanceRunResponse)
def run_maintenance(req: MaintenanceRunRequest) -> MaintenanceRunResponse:
    """Run KG pruning and orphan cleanup for one client or every client in a tenant."""
    sb = get_supabase()
    memory_state = MemoryStateService(sb)

    try:
        tenant_id = UUID(req.tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tenant_id: {e}")

    client_ids = [req.client_id] if req.client_id else _discover_client_ids(sb, tenant_id)
    if not client_ids:
        tenant_memory_version = memory_state.bump(
            tenant_id=tenant_id,
            client_id=None,
            change_type="maintenance",
            metadata={"clients_processed": 0, "cleanup_orphans": req.cleanup_orphans},
        ) or 0
        return MaintenanceRunResponse(
            tenant_id=req.tenant_id,
            client_ids_processed=[],
            total_clients=0,
            results=[],
            totals={
                "edges_archived": 0,
                "nodes_archived": 0,
                "edge_evidence_deleted": 0,
                "node_evidence_deleted": 0,
                "orphaned_nodes_deleted": 0,
            },
            tenant_memory_version=tenant_memory_version,
            metadata={"message": "No clients found for tenant."},
        )

    results: list[MaintenanceClientResult] = []
    totals = {
        "edges_archived": 0,
        "nodes_archived": 0,
        "edge_evidence_deleted": 0,
        "node_evidence_deleted": 0,
        "orphaned_nodes_deleted": 0,
    }

    for client_id in client_ids:
        try:
            prune_res = sb.rpc(
                "prune_kg",
                {
                    "p_tenant_id": str(tenant_id),
                    "p_client_id": client_id,
                    "p_edge_stale_days": req.edge_stale_days,
                    "p_node_stale_days": req.node_stale_days,
                    "p_min_degree": req.min_degree,
                    "p_keep_edge_evidence": req.keep_edge_evidence,
                    "p_keep_node_evidence": req.keep_node_evidence,
                },
            ).execute()
            prune_data = prune_res.data or {}

            orphaned_nodes_deleted = 0
            if req.cleanup_orphans:
                cleanup_res = sb.rpc(
                    "cleanup_orphaned_kg_nodes",
                    {"p_tenant_id": str(tenant_id), "p_client_id": client_id},
                ).execute()
                cleanup_data = cleanup_res.data or {}
                orphaned_nodes_deleted = int(cleanup_data.get("nodes_deleted") or 0)

            client_memory_version = memory_state.bump(
                tenant_id=tenant_id,
                client_id=UUID(client_id),
                change_type="maintenance",
                metadata={
                    "cleanup_orphans": req.cleanup_orphans,
                    "edges_archived": int(prune_data.get("edges_archived") or 0),
                    "nodes_archived": int(prune_data.get("nodes_archived") or 0),
                    "orphaned_nodes_deleted": orphaned_nodes_deleted,
                },
            ) or 0

            result = MaintenanceClientResult(
                client_id=client_id,
                edges_archived=int(prune_data.get("edges_archived") or 0),
                nodes_archived=int(prune_data.get("nodes_archived") or 0),
                edge_evidence_deleted=int(prune_data.get("edge_evidence_deleted") or 0),
                node_evidence_deleted=int(prune_data.get("node_evidence_deleted") or 0),
                orphaned_nodes_deleted=orphaned_nodes_deleted,
                client_memory_version=client_memory_version,
            )
            totals["edges_archived"] += result.edges_archived
            totals["nodes_archived"] += result.nodes_archived
            totals["edge_evidence_deleted"] += result.edge_evidence_deleted
            totals["node_evidence_deleted"] += result.node_evidence_deleted
            totals["orphaned_nodes_deleted"] += result.orphaned_nodes_deleted
            logger.info(
                "admin.maintenance tenant=%s client=%s edges_archived=%d nodes_archived=%d edge_evidence_deleted=%d node_evidence_deleted=%d orphaned_nodes_deleted=%d client_memory_version=%d",
                tenant_id,
                client_id,
                result.edges_archived,
                result.nodes_archived,
                result.edge_evidence_deleted,
                result.node_evidence_deleted,
                result.orphaned_nodes_deleted,
                result.client_memory_version,
            )
        except Exception as e:
            logger.exception("Maintenance failed tenant=%s client=%s", tenant_id, client_id)
            result = MaintenanceClientResult(client_id=client_id, error=str(e))
        results.append(result)

    tenant_memory_version = memory_state.bump(
        tenant_id=tenant_id,
        client_id=None,
        change_type="maintenance",
        metadata={
            "clients_processed": len(client_ids),
            "cleanup_orphans": req.cleanup_orphans,
            **totals,
        },
    ) or 0

    return MaintenanceRunResponse(
        tenant_id=req.tenant_id,
        client_ids_processed=client_ids,
        total_clients=len(client_ids),
        results=results,
        totals=totals,
        tenant_memory_version=tenant_memory_version,
        metadata={
            "cleanup_orphans": req.cleanup_orphans,
            "edge_stale_days": req.edge_stale_days,
            "node_stale_days": req.node_stale_days,
            "min_degree": req.min_degree,
            "keep_edge_evidence": req.keep_edge_evidence,
            "keep_node_evidence": req.keep_node_evidence,
        },
    )

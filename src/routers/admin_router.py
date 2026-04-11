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

from fastapi import APIRouter, Query

from src.supabase.supabase_client import get_supabase
from src.models.api.admin import (
    HealthResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


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

    def _count(table: str, filters: dict) -> int:
        q = sb.table(table).select("id", count="exact")
        for col, val in filters.items():
            q = q.eq(col, val)
        return q.execute().count or 0

    doc_count = _count("documents", {"tenant_id": str(tenant_id), "client_id": str(client_id)})
    chunk_count = _count("chunks", {"tenant_id": str(tenant_id)})

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

    return StatsResponse(
        tenant_id=str(tenant_id),
        client_id=str(client_id),
        document_count=doc_count,
        chunk_count=chunk_count,
        chunks_with_embeddings=chunks_with_embeddings,
        kg_node_count=node_count,
        kg_edge_count=edge_count,
    )

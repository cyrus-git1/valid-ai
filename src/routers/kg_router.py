"""
/kg router
----------
Build, inspect and maintain the Knowledge Graph.

POST /kg/build                     — Build KG nodes + similarity edges from chunk embeddings
POST /kg/prune                     — Archive stale nodes/edges, trim evidence tables
GET  /kg/nodes                     — List KG nodes (filterable by type, status)
GET  /kg/nodes/{node_id}           — Get a single node
GET  /kg/nodes/{node_id}/edges     — Get edges connected to a node
GET  /kg/edges/{edge_id}           — Get a single edge
"""
from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from src.supabase.supabase_client import get_supabase
from src.models.api.kg import (
    KGBuildRequest,
    KGBuildResponse,
    KGEdgeListResponse,
    KGEdgeResponse,
    KGNodeListResponse,
    KGNodeResponse,
    PruneRequest,
    PruneResponse,
)
from src.services.kg_service import KGBuildConfig, KGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/kg", tags=["knowledge-graph"])


@router.post("/build", response_model=KGBuildResponse)
def build_kg(req: KGBuildRequest) -> KGBuildResponse:
    """
    Build KG nodes and similarity edges from existing chunk embeddings.

    This is the step you call after /ingest/file or /ingest/web finishes.
    It reads all embedded chunks for the given client, upserts a KG node
    for each chunk, then draws cosine-similarity edges between nodes whose
    embedding similarity exceeds the threshold.

    Scoping:
      - document_id=None  → process all documents for this client (default)
      - document_id=<id>  → process only that one document
    """
    sb = get_supabase()
    svc = KGService(sb)

    cfg = KGBuildConfig(
        similarity_threshold=req.similarity_threshold,
        max_edges_per_chunk=req.max_edges_per_chunk,
        max_chunks=req.max_chunks,
        batch_size=req.batch_size,
    )

    result = svc.build_kg_from_chunk_embeddings(
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        document_id=req.document_id,
        config=cfg,
    )

    return KGBuildResponse(
        chunks_fetched=result.get("chunks_fetched", 0),
        chunks_valid=result.get("chunks_valid", 0),
        chunks_skipped=result.get("chunks_skipped", 0),
        nodes_upserted=result.get("nodes_upserted", 0),
        edges_upserted=result.get("edges_upserted", 0),
        similarity_threshold=result.get("similarity_threshold", req.similarity_threshold),
        max_edges_per_chunk=result.get("max_edges_per_chunk", req.max_edges_per_chunk),
    )


@router.post("/prune", response_model=PruneResponse)
def prune_kg(req: PruneRequest) -> PruneResponse:
    """
    Archive stale KG nodes and edges, trim evidence tables.

    Stale = not seen (last_seen_at) within the configured window.
    Low-degree nodes (fewer active edges than min_degree) are also archived.
    Evidence rows beyond keep_* limits are deleted oldest/lowest-score first.
    """
    sb = get_supabase()
    svc = KGService(sb)

    result = svc.prune(
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        edge_stale_days=req.edge_stale_days,
        node_stale_days=req.node_stale_days,
        min_degree=req.min_degree,
        keep_edge_evidence=req.keep_edge_evidence,
        keep_node_evidence=req.keep_node_evidence,
    )

    return PruneResponse(
        edges_archived=result.get("edges_archived", 0),
        nodes_archived=result.get("nodes_archived", 0),
        edge_evidence_deleted=result.get("edge_evidence_deleted", 0),
        node_evidence_deleted=result.get("node_evidence_deleted", 0),
    )


@router.get("/nodes", response_model=KGNodeListResponse)
def list_nodes(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    node_type: Optional[str] = Query(default=None, description="Filter by type e.g. 'Chunk'"),
    status: Optional[str] = Query(default="active", description="active | archived | pending_linking"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> KGNodeListResponse:
    """
    List KG nodes for a tenant + client.
    Filterable by node type and status. Paginated.
    """
    sb = get_supabase()

    q = (
        sb.table("kg_nodes")
        .select(
            "id, node_key, type, name, description, properties, status, seen_count",
            count="exact",
        )
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
    )
    if status:
        q = q.eq("status", status)
    if node_type:
        q = q.eq("type", node_type)

    res = q.order("created_at", desc=True).range(offset, offset + limit - 1).execute()

    items = [KGNodeResponse(**row) for row in (res.data or [])]
    return KGNodeListResponse(items=items, total=res.count or 0, limit=limit, offset=offset)


@router.get("/nodes/{node_id}", response_model=KGNodeResponse)
def get_node(
    node_id: str,
    tenant_id: UUID = Query(...),
) -> KGNodeResponse:
    """Get a single KG node by ID."""
    sb = get_supabase()

    res = (
        sb.table("kg_nodes")
        .select("id, node_key, type, name, description, properties, status, seen_count")
        .eq("id", node_id)
        .eq("tenant_id", str(tenant_id))
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")

    return KGNodeResponse(**res.data[0])


@router.get("/nodes/{node_id}/edges", response_model=KGEdgeListResponse)
def get_node_edges(
    node_id: str,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    direction: str = Query(default="out", description="'out' | 'in' | 'both'"),
) -> KGEdgeListResponse:
    """
    Get all edges connected to a node.

    direction:
      out  — edges where this node is the source (default)
      in   — edges where this node is the destination
      both — all edges touching this node
    """
    sb = get_supabase()

    def _fetch(src_col: str, dst_col: str) -> list:
        return (
            sb.table("kg_edges")
            .select("id, src_id, dst_id, rel_type, weight, properties, is_active")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .eq(src_col, node_id)
            .eq("is_active", True)
            .execute()
            .data or []
        )

    rows = []
    if direction in ("out", "both"):
        rows.extend(_fetch("src_id", "dst_id"))
    if direction in ("in", "both"):
        rows.extend(_fetch("dst_id", "src_id"))

    seen = set()
    items = []
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            items.append(KGEdgeResponse(**row))

    return KGEdgeListResponse(items=items)


@router.get("/edges/{edge_id}", response_model=KGEdgeResponse)
def get_edge(
    edge_id: str,
    tenant_id: UUID = Query(...),
) -> KGEdgeResponse:
    """Get a single KG edge by ID."""
    sb = get_supabase()

    res = (
        sb.table("kg_edges")
        .select("id, src_id, dst_id, rel_type, weight, properties, is_active")
        .eq("id", edge_id)
        .eq("tenant_id", str(tenant_id))
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail=f"Edge '{edge_id}' not found.")

    return KGEdgeResponse(**res.data[0])

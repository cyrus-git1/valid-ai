"""Pydantic models for the /kg router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class KGBuildRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    document_id: Optional[UUID] = None       # scope to one doc, or None for all
    similarity_threshold: float = 0.82
    max_edges_per_chunk: int = 10
    max_chunks: int = 2000
    batch_size: int = 500


class KGBuildResponse(BaseModel):
    chunks_fetched: int
    chunks_valid: int
    chunks_skipped: int
    nodes_upserted: int
    edges_upserted: int
    similarity_threshold: float
    max_edges_per_chunk: int


class PruneRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    edge_stale_days: int = 90
    node_stale_days: int = 180
    min_degree: int = 3
    keep_edge_evidence: int = 5
    keep_node_evidence: int = 10


class PruneResponse(BaseModel):
    edges_archived: int
    nodes_archived: int
    edge_evidence_deleted: int
    node_evidence_deleted: int


class KGNodeResponse(BaseModel):
    id: str
    node_key: str
    type: str
    name: str
    description: Optional[str]
    properties: Dict[str, Any]
    status: str
    seen_count: int


class KGNodeListResponse(BaseModel):
    items: List[KGNodeResponse]
    total: int
    limit: int
    offset: int


class KGEdgeResponse(BaseModel):
    id: str
    src_id: str
    dst_id: str
    rel_type: str
    weight: Optional[float]
    properties: Dict[str, Any]
    is_active: bool


class KGEdgeListResponse(BaseModel):
    items: List[KGEdgeResponse]

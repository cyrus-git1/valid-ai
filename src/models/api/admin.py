"""Pydantic models for the /admin router."""
from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str             # "ok" | "degraded"
    supabase: bool
    openai: bool
    detail: Optional[str] = None


class StatsResponse(BaseModel):
    tenant_id: str
    client_id: str
    document_count: int
    chunk_count: int
    chunks_with_embeddings: int
    kg_node_count: int
    kg_edge_count: int


class ReindexRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64


class ReindexResponse(BaseModel):
    document_id: str
    chunks_upserted: int
    warnings: list


class RebuildKGRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    similarity_threshold: float = 0.82
    max_edges_per_chunk: int = 10
    batch_size: int = 500


class RebuildKGResponse(BaseModel):
    nodes_upserted: int
    edges_upserted: int
    chunks_processed: int

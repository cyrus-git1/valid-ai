"""Pydantic models for the /admin router."""
from __future__ import annotations

from typing import Optional

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

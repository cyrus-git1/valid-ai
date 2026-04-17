"""Pydantic models for the /admin router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    client_memory_version: int = 0
    tenant_memory_version: int = 0
    client_last_ingested_at: Optional[str] = None
    client_last_changed_at: Optional[str] = None
    client_last_summary_at: Optional[str] = None


class MaintenanceRunRequest(BaseModel):
    tenant_id: str
    client_id: Optional[str] = None
    edge_stale_days: int = Field(default=90, ge=1, le=3650)
    node_stale_days: int = Field(default=180, ge=1, le=3650)
    min_degree: int = Field(default=3, ge=0, le=1000)
    keep_edge_evidence: int = Field(default=5, ge=0, le=1000)
    keep_node_evidence: int = Field(default=10, ge=0, le=1000)
    cleanup_orphans: bool = True


class MaintenanceClientResult(BaseModel):
    client_id: str
    edges_archived: int = 0
    nodes_archived: int = 0
    edge_evidence_deleted: int = 0
    node_evidence_deleted: int = 0
    orphaned_nodes_deleted: int = 0
    client_memory_version: int = 0
    error: Optional[str] = None


class MaintenanceRunResponse(BaseModel):
    tenant_id: str
    client_ids_processed: List[str] = Field(default_factory=list)
    total_clients: int = 0
    results: List[MaintenanceClientResult] = Field(default_factory=list)
    totals: Dict[str, int] = Field(default_factory=dict)
    tenant_memory_version: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

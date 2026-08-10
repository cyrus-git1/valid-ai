"""Pydantic models for the /admin router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str             # "ok" | "degraded"
    supabase: bool
    openai: bool
    reranker: bool = False  # cross-encoder rerank available? (False = degraded retrieval, not an outage)
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


# ── API key management ───────────────────────────────────────────────────────


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


# ── Tenant plan management ───────────────────────────────────────────────────


class TenantPlanResponse(BaseModel):
    tenant_id: str
    plan: str
    notes: Optional[str] = None
    daily_embedding_tokens_limit: int
    daily_embedding_tokens_used: int = 0
    max_body_bytes: int
    max_chunks_per_ingest: int


class TenantPlanSetRequest(BaseModel):
    plan: str = Field(description="free | pro | enterprise")
    notes: Optional[str] = None


# ── Taxonomy mirroring ───────────────────────────────────────────────────────


class TaxonomyTag(BaseModel):
    tag_id: str = Field(description="transcript_tags uuid — becomes the concept node id + external_ref")
    label: str = Field(min_length=1)
    description: Optional[str] = None


class MirrorTaxonomyRequest(BaseModel):
    tenant_id: str
    client_id: Optional[str] = None
    tags: List[TaxonomyTag] = Field(default_factory=list, description="Seeded codebook tags to mirror")


class MirrorTaxonomyResultItem(BaseModel):
    tag_id: str
    concept_id: str = ""
    external_ref: str = ""
    embedded: bool = False
    error: Optional[str] = None


class MirrorTaxonomyResponse(BaseModel):
    tenant_id: str
    mirrored: int
    embedded: int
    failed: int
    results: List[MirrorTaxonomyResultItem]


# ── Re-embed backfill ────────────────────────────────────────────────────────


class ReembedRequest(BaseModel):
    tenant_id: str
    types: Optional[List[str]] = Field(
        default=None, description="Node types to re-embed, e.g. ['Observation','Concept']. None = all embeddable."
    )
    limit: int = Field(default=500, ge=1, le=2000, description="Max nodes to re-embed this sweep (chunk large backfills).")


class ReembedResponse(BaseModel):
    tenant_id: str
    scanned: int          # null-embedding nodes picked this sweep
    reembedded: int       # rows actually filled
    remaining: int        # null-embedding nodes still left after this sweep
    embedding_model: str


# ── Orphan retirement / study purge ──────────────────────────────────────────


class RetireOrphansRequest(BaseModel):
    tenant_id: str


class RetireOrphansResponse(BaseModel):
    retired: int
    retired_labels: List[str] = Field(default_factory=list)


class PurgeStudyRequest(BaseModel):
    tenant_id: str
    study_id: str


class PurgeStudyResponse(BaseModel):
    observations_deleted: int
    concepts_deleted: int

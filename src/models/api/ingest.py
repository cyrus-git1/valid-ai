"""Pydantic models for the /ingest router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkItem(BaseModel):
    text: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    token_count: Optional[int] = None


class EntityItem(BaseModel):
    name: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class _ProvenanceMixin(BaseModel):
    """Optional provenance fields. Agents may pass these to associate an
    ingest with a specific actor / source app / request for audit + history.
    """
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None  # 'user' | 'service' | 'agent' | 'anonymous'
    source_app: Optional[str] = None
    request_id: Optional[str] = None
    previous_version_id: Optional[str] = None


class ProcessedDocumentRequest(_ProvenanceMixin):
    tenant_id: UUID
    client_id: UUID
    file_name: str
    file_bytes_b64: str
    source_type: str
    title: str
    source_timestamp: Optional[datetime] = None
    is_pinned: bool = False
    is_canonical: bool = False
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[ChunkItem] = Field(default_factory=list)
    entities: List[EntityItem] = Field(default_factory=list)


class ProcessedWebRequest(_ProvenanceMixin):
    tenant_id: UUID
    client_id: UUID
    url: str
    title: str
    source_timestamp: Optional[datetime] = None
    is_pinned: bool = False
    is_canonical: bool = False
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[ChunkItem] = Field(default_factory=list)
    entities: List[EntityItem] = Field(default_factory=list)


class IngestProcessedResponse(BaseModel):
    document_id: str
    source_uri: str
    chunks_upserted: int
    entities_linked: int = 0
    warnings: List[str] = Field(default_factory=list)


class SummaryIngestRequest(_ProvenanceMixin):
    """Thin, synchronous ingest path for LLM-generated summary chunks.

    Produces exactly one document + one chunk + one KG node. Skips the
    O(n^2) semantic-linking phase because there's only one chunk. Entity
    mentions (if provided) use weight=0.5 so source evidence always
    outranks summary evidence during graph expansion.
    """
    tenant_id: UUID
    client_id: UUID
    source_type: str = Field(
        description="ContextSummary | DocumentSummary | TopicSummary",
    )
    summary_text: str
    topics: List[str] = Field(default_factory=list)
    # Scope-identity: exactly one of document_id or topic may be set depending
    # on source_type. For ContextSummary, both are None.
    document_id: Optional[str] = None
    topic: Optional[str] = None
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_stats: Dict[str, Any] = Field(default_factory=dict)
    memory_version_at_generation: Optional[int] = None
    entities: List[EntityItem] = Field(default_factory=list)
    # Optional arbitrary metadata from the agent (model, prompt version, etc.)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class SummaryIngestResponse(BaseModel):
    document_id: str
    chunk_id: str
    node_id: str
    superseded_document_id: Optional[str] = None
    memory_version: int


class IngestJobAck(BaseModel):
    job_id: str
    status: str = "queued"


class IngestJobStatus(BaseModel):
    job_id: str
    status: str
    job_type: str
    tenant_id: str
    client_id: Optional[str] = None
    document_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    # Provenance
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None
    source_app: Optional[str] = None
    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    request_id: Optional[str] = None


class IngestJobsListResponse(BaseModel):
    items: List[IngestJobStatus]
    total: int

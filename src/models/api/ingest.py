"""Pydantic models for the /ingest router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

from src.models.base import TenantScoped


# ── Service-layer DTOs ────────────────────────────────────────────────────────


class IngestInput(BaseModel):
    """Parameters for the ingest pipeline (service layer)."""
    tenant_id: UUID
    client_id: UUID

    # File ingest — provide both
    file_bytes: Optional[bytes] = None
    file_name: Optional[str] = None

    # Web ingest — provide this
    web_url: Optional[str] = None

    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64
    prune_after_ingest: bool = False

    model_config = {"arbitrary_types_allowed": True}


class IngestOutput(BaseModel):
    """Result returned by the ingest pipeline (service layer)."""
    document_id: UUID
    source_type: str
    source_uri: str
    chunks_upserted: int
    chunk_ids: List[UUID] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    prune_result: Optional[Dict[str, Any]] = None


# ── Router response models ───────────────────────────────────────────────────


class IngestFileResponse(BaseModel):
    job_id: str
    document_id: str
    source_type: str
    source_uri: str
    chunks_upserted: int
    warnings: List[str] = []
    prune_result: Optional[Dict[str, Any]] = None


class IngestWebRequest(TenantScoped):
    url: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prune_after_ingest: bool = False


class IngestWebResponse(BaseModel):
    job_id: str
    document_id: str
    source_type: str
    source_uri: str
    chunks_upserted: int
    warnings: List[str] = []
    prune_result: Optional[Dict[str, Any]] = None


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str        # "complete" | "running" | "failed"
    detail: Optional[str] = None


# ── Batch ingest models ──────────────────────────────────────────────────────

class BatchWebItem(BaseModel):
    url: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchWebRequest(TenantScoped):
    items: List[BatchWebItem] = Field(..., min_length=1, max_length=50)
    prune_after_ingest: bool = False


class BatchItemStatus(BaseModel):
    index: int
    source: str                # file name or URL
    status: str                # "running" | "complete" | "failed"
    document_id: Optional[str] = None
    chunks_upserted: int = 0
    warnings: List[str] = []
    detail: Optional[str] = None


class BatchIngestResponse(BaseModel):
    batch_id: str
    total: int
    status: str                # "running" | "complete" | "partial_failure"
    items: List[BatchItemStatus] = []


class BatchIngestStatusResponse(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    running: int
    status: str                # "running" | "complete" | "partial_failure"
    items: List[BatchItemStatus] = []

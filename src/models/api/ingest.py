"""Pydantic models for the /ingest router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class IngestFileResponse(BaseModel):
    job_id: str
    document_id: str
    source_type: str
    source_uri: str
    chunks_upserted: int
    warnings: List[str] = []
    prune_result: Optional[Dict[str, Any]] = None


class IngestWebRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
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


class BatchWebRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
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

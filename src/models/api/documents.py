"""Pydantic models for the /documents router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: str
    tenant_id: str
    client_id: Optional[str]
    source_type: str
    source_uri: Optional[str]
    title: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse]
    total: int
    limit: int
    offset: int


class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    page_start: Optional[int]
    page_end: Optional[int]
    content: str
    content_tokens: Optional[int]
    metadata: Dict[str, Any]
    has_embedding: bool


class ChunkListResponse(BaseModel):
    items: List[ChunkResponse]
    total: int
    limit: int
    offset: int


class BulkDeleteRequest(BaseModel):
    document_ids: List[str] = Field(
        min_length=1,
        description="List of document IDs to delete (with cascading chunk cleanup).",
    )


class BulkDeleteResponse(BaseModel):
    deleted: int = Field(description="Number of documents successfully deleted.")
    not_found: List[str] = Field(
        default_factory=list,
        description="Document IDs that were not found.",
    )

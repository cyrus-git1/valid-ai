"""Pydantic models for the /documents router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class DocumentWithChunksResponse(BaseModel):
    id: str
    tenant_id: str
    client_id: Optional[str]
    source_type: str
    source_uri: Optional[str]
    title: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    chunks: List[ChunkResponse] = Field(default_factory=list)


class DocumentWithChunksListResponse(BaseModel):
    items: List[DocumentWithChunksResponse]
    total: int


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

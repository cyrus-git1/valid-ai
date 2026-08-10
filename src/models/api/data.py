"""Pydantic models for the /data router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class _DocumentPatchRequest(BaseModel):
    status: Optional[str] = Field(default=None, description="active | draft | deprecated | archived | flagged")
    is_pinned: Optional[bool] = None
    is_canonical: Optional[bool] = None


class _BulkDeleteRequest(BaseModel):
    document_ids: List[str] = Field(min_length=1)


class PreviewUrlResponse(BaseModel):
    url: str
    expires_at: str
    source_type: Optional[str] = None
    filename: Optional[str] = None


class SurveyQuestions(BaseModel):
    survey_id: str
    output_type: str
    study_id: Optional[str] = None            # from metadata.study_id (no native column)
    questions: List[Dict[str, Any]] = Field(default_factory=list)   # agent-authored shape
    created_at: Optional[str] = None


class SurveyOutputsResponse(BaseModel):
    surveys: List[SurveyQuestions] = Field(default_factory=list)


class KgEntity(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None


class KgEntitiesResponse(BaseModel):
    entities: List[KgEntity]


class DocumentTitlesRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    document_ids: List[str]


class DocumentTitlesResponse(BaseModel):
    titles: Dict[str, str]


class ContextSummaryReadResponse(BaseModel):
    document_id: str
    tenant_id: str
    client_id: str
    source_type: str
    summary: str
    topics: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_stats: Dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: List[str] = Field(default_factory=list)
    memory_version_at_generation: int = 0
    current_memory_version: int = 0
    is_stale: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SummaryListItem(BaseModel):
    document_id: str
    source_type: str
    tenant_id: str
    client_id: Optional[str] = None
    scope_ref: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    memory_version_at_generation: int = 0
    is_stale: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SummaryListResponse(BaseModel):
    items: List[SummaryListItem]
    total: int

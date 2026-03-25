"""Pydantic models for the /panel router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.config.llm import LLMConfig
from src.models.base import TenantScoped


# ── Ingest models ────────────────────────────────────────────────────────────


class PanelIngestRequest(TenantScoped):
    """Request body for POST /panel/ingest."""
    vendor_name: str
    participants: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of participant objects from the vendor",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64
    build_kg: bool = True


class PanelParticipantResult(BaseModel):
    """Per-participant result within a batch ingest."""
    participant_index: int
    document_id: UUID
    chunks_upserted: int
    warnings: List[str] = Field(default_factory=list)


class PanelIngestResponse(BaseModel):
    """Response for POST /panel/ingest."""
    job_id: str
    vendor_name: str
    total_participants: int
    status: str                     # "running" | "complete" | "partial_failure"
    completed: int = 0
    failed: int = 0
    results: List[PanelParticipantResult] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ── Filter models ────────────────────────────────────────────────────────────


class PanelFilterRequest(TenantScoped):
    """Request body for POST /panel/filter."""
    filter_mode: str = Field(
        default="full",
        description="'label' | 'llm' | 'embedding' | 'full'",
    )
    top_k: int = Field(default=20, ge=1, le=200)
    similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    llm_model: str = LLMConfig.DEFAULT


class PanelFilterResult(BaseModel):
    """Single participant match from filtering."""
    document_id: str
    participant_name: str
    relevance_score: float
    match_reasons: List[str] = Field(default_factory=list)
    matched_labels: List[str] = Field(default_factory=list)


class PanelFilterResponse(BaseModel):
    """Response for POST /panel/filter."""
    filter_mode: str
    total_evaluated: int
    total_matched: int
    results: List[PanelFilterResult] = Field(default_factory=list)

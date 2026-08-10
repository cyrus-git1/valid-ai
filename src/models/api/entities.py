"""Pydantic models for the /entities and /kg/entities routers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Tier 1 — extraction cache ────────────────────────────────────────────────


class CachedEntitiesResponse(BaseModel):
    transcript_sha256: str
    entities: List[Dict[str, Any]]
    model_name: str
    prompt_version: str
    created_at: str
    expires_at: str


class CacheUpsertRequest(BaseModel):
    transcript_sha256: str = Field(min_length=64, max_length=64,
                                   description="sha256 hex digest of the transcript text")
    tenant_id: UUID
    client_id: Optional[UUID] = None
    entities: List[Dict[str, Any]]
    model_name: str = Field(min_length=1)
    prompt_version: str = Field(default="v1")


class CacheUpsertResponse(BaseModel):
    transcript_sha256: str


# ── Tier 2 — entity KG nodes + MENTIONS edges ────────────────────────────────


class EntitySentiment(BaseModel):
    compound_mean: Optional[float] = None
    compound_min: Optional[float] = None
    compound_max: Optional[float] = None
    valence: Optional[str] = None   # positive|negative|neutral|mixed
    mention_count: int = 0


class EntityInput(BaseModel):
    canonical_id: str = Field(min_length=12, max_length=12)
    canonical_text: str
    label: str
    source: str = "spacy"                  # spacy|llm_custom|both
    count: int = 1
    confidence: float = 1.0
    cue_indices: List[int] = Field(default_factory=list)
    sample_contexts: List[str] = Field(default_factory=list)
    sentiment: Optional[EntitySentiment] = None


class EntityUpsertRequest(BaseModel):
    tenant_id: UUID
    client_id: Optional[UUID] = None
    document_id: UUID = Field(description="Session in the brief; maps to documents.id")
    survey_id: Optional[UUID] = None       # carried through to edge properties only
    entities: List[EntityInput]
    extractor_version: str = "1.0"


class EntityRedirect(BaseModel):
    canonical_id: str
    surviving_id: str


class EntityUpsertResponse(BaseModel):
    upserted_entities: int
    upserted_mentions: int
    merged_into_existing: List[EntityRedirect] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class EntityMergeRequest(BaseModel):
    tenant_id: UUID
    surviving_canonical_id: str = Field(min_length=12, max_length=12)
    surviving_label: str


class EntityMergeResponse(BaseModel):
    merged: bool
    redirect_count: int


# ── Tier 3 — rollups ─────────────────────────────────────────────────────────


class MentionItem(BaseModel):
    document_id: str
    survey_id: Optional[str] = None
    count: int = 1
    sentiment: Optional[EntitySentiment] = None
    sample_contexts: List[str] = Field(default_factory=list)
    extracted_at: Optional[str] = None


class EntityRollup(BaseModel):
    canonical_id: str
    canonical_text: str
    label: str
    total_mentions: int = 0
    distinct_session_count: int = 0
    sentiment_aggregate: Optional[Dict[str, Any]] = None
    first_seen_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    status: str = "active"
    merged_into: Optional[str] = None


class EntityRollupResponse(BaseModel):
    entity: EntityRollup
    mentions: List[MentionItem] = Field(default_factory=list)


class EntitySearchResultItem(BaseModel):
    canonical_id: str
    canonical_text: str
    label: str
    total_mentions: int = 0
    sentiment_aggregate: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class EntitySearchResponse(BaseModel):
    entities: List[EntitySearchResultItem]

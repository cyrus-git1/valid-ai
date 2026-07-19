"""
src/models/api/app_entities.py
------------------------------
Request/response envelopes for app-entity recall (a near-clone of the
observation/concept spine). App objects — insights, quotes, sessions — are
stored as kg_nodes keyed by their app id and recalled semantically.
"""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.models.base import TenantOwned

AppEntityKind = Literal["insight", "quote", "session"]


# ── upsert ──────────────────────────────────────────────────────────────────


class AppEntityUpsertRequest(TenantOwned):
    study_id: UUID
    kind: AppEntityKind
    external_id: str = Field(min_length=1, description="The app's row id (insights.id / transcript_highlights.id / calls.id)")
    text: str = Field(min_length=1, description="Embeddable text: insight body / quote text / session summary")
    status: Literal["active", "inactive"] = Field(
        default="active", description="'inactive' soft-deletes (excluded from recall)"
    )
    concept_hints: List[str] = Field(default_factory=list, description="Optional; reserved for about_concept linking")


class AppEntityUpsertResponse(BaseModel):
    node_id: str
    external_id: str
    kind: str
    created: bool


# ── nearest ─────────────────────────────────────────────────────────────────


class AppEntityNearestRequest(TenantOwned):
    query_text: Optional[str] = Field(default=None, description="Embedded server-side if `embedding` omitted")
    embedding: Optional[List[float]] = Field(default=None, description="Pre-computed 1536-d embedding")
    study_ids: List[UUID] = Field(default_factory=list, description="SCOPE — only nodes in these studies")
    kinds: Optional[List[AppEntityKind]] = Field(default=None, description="Optional kind filter; default all")
    top_k: int = Field(default=20, ge=1, le=200)


class AppEntityMatch(BaseModel):
    external_id: str
    kind: str
    study_id: Optional[str] = None
    similarity: float


class AppEntityNearestResponse(BaseModel):
    matches: List[AppEntityMatch] = Field(default_factory=list)

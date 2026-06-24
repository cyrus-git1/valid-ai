"""
src/models/api/observations.py
------------------------------
Request/response envelopes for the observation spine (first slice).

An observation is a structured signal the agent layer absorbs from any modality.
The data plane stores it verbatim: the `value {number, unit}` payload is OPAQUE —
never parsed for meaning, never recomputed, returned byte-for-byte on read. The
descriptive fields are queryable; the source block carries provenance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import TenantOwned


# ── Sub-objects (carried through verbatim where noted) ──────────────────────


class Prevalence(BaseModel):
    pct: Optional[float] = None
    n: Optional[int] = None


class Reliability(BaseModel):
    sample_n: Optional[int] = None
    method: Optional[str] = None
    quality_flags: List[str] = Field(default_factory=list)
    diarization_confidence: Optional[float] = None


class Segment(BaseModel):
    persona: Optional[str] = None
    variant_key: Optional[str] = None


class ObservationSource(BaseModel):
    aggregate_id: Optional[str] = None
    call_id: Optional[str] = None
    input_hash: Optional[str] = None
    agent_version: Optional[str] = None
    evidence_ref: Optional[str] = None


# ── Upsert ──────────────────────────────────────────────────────────────────


class ObservationUpsertRequest(TenantOwned):
    """The agent layer's observation envelope. Idempotent by observation_id."""

    observation_id: str = Field(min_length=1, description="Stable hash; the idempotency key")
    nl_text: str = Field(min_length=1, description="Natural-language form; embedded into the shared space")

    # OPAQUE, citable payload — stored & returned verbatim, never parsed.
    value: Dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. {number, unit}. Stored verbatim, never recomputed.",
    )

    # Queryable descriptive fields.
    modality: Optional[str] = None
    signal_type: Optional[str] = None
    direction: Optional[str] = None
    prevalence: Optional[Prevalence] = None
    confidence: Optional[float] = None
    reliability: Optional[Reliability] = None
    segment: Optional[Segment] = None
    occurred_at: Optional[str] = None

    # Provenance.
    source: Optional[ObservationSource] = None

    # Resolution links (agent-resolved; optional in the first slice).
    study_id: Optional[UUID] = None
    concept_id: Optional[UUID] = Field(
        default=None, description="Resolved Concept node id → observation:about_concept edge"
    )
    evidence_chunk_id: Optional[UUID] = Field(
        default=None, description="Underlying chunk; creates the observation→evidence link"
    )


class ObservationUpsertResponse(BaseModel):
    observation_id: str
    node_id: str
    evidence_linked: bool = False
    concept_linked: bool = False


# ── Scoped fetch: observations by concept ───────────────────────────────────


class ObservationRecord(BaseModel):
    """One observation with structured value + full provenance, as stored."""

    node_id: str
    observation_id: Optional[str] = None
    nl_text: Optional[str] = None
    value: Optional[Dict[str, Any]] = None
    modality: Optional[str] = None
    signal_type: Optional[str] = None
    direction: Optional[str] = None
    prevalence: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    reliability: Optional[Dict[str, Any]] = None
    segment: Optional[Dict[str, Any]] = None
    occurred_at: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    study_id: Optional[str] = None
    evidence_chunk_ids: List[str] = Field(default_factory=list)


class ObservationsByConceptResponse(BaseModel):
    concept_id: str
    observations: List[ObservationRecord] = Field(default_factory=list)

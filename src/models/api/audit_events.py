"""
src/models/api/audit_events.py
------------------------------
Write and read envelopes for durable audit history — one row per `/audit/survey`
result, append-only, retained a year.

Replaces an in-process dict in valid-agents that claimed to be a SOC 2 record and was
emptied by every deploy. The write is the half with a deadline: history is the one thing
that cannot be backfilled.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# These models deliberately do NOT extend TenantOwned, which every other spine model
# does. TenantOwned carries `client_id: UUID`, and on this platform the client id IS the
# authenticated user's id — a raw one in an append-only row retained a year past the
# study's deletion is personal data that can never be scrubbed, because the table refuses
# UPDATE. So the client arrives already hashed, as `client_ref`, exactly like `actor_ref`.
# tenant_id stays a real id: an organisation is not a person, and scoping needs it.


class _AuditScope(BaseModel):
    tenant_id: UUID
    client_ref: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "PSEUDONYMOUS client reference — a keyed hash of the client id, computed by "
            "the caller. Never a raw client id: see the note above."
        ),
    )


class AuditEventRecordRequest(_AuditScope):
    """One audit result. Every field but the scope is optional, because a DEGRADED audit
    still happened and is still worth recording — an audit that could not compute a score
    is itself the fact an auditor would want."""

    study_id: Optional[UUID] = Field(default=None, description="Study this audit was of.")

    actor_ref: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "PSEUDONYMOUS actor reference — a keyed hash of the actor id, computed by the "
            "caller. Never a raw user id: this row outlives the study it describes, so a "
            "raw id would keep identifying a person after the study was deleted. The hash "
            "keeps same-person linkage and lets a known person be found by hashing their "
            "id; it drops only the ability to read a name out of the table."
        ),
    )
    request_id: Optional[str] = Field(default=None, max_length=128)

    goal_version: Optional[int] = None
    alignment_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status: str = Field(default="complete", pattern="^(complete|partial|failed|empty)$")
    degraded: List[str] = Field(
        default_factory=list,
        description="Analyzers that did not run. Non-empty means status is 'partial'.",
    )
    finding_counts: Dict[str, Any] = Field(
        default_factory=dict, description="Counts by severity and by check.",
    )
    payload_digest: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "sha256 of the audited questions. Ties the entry to exactly what was audited "
            "without storing a survey copy per audit."
        ),
    )
    summary: Optional[str] = Field(
        default=None,
        description=(
            "The audit's own one-line conclusion. Stored because a trend of scores with "
            "no words is barely readable; the FULL result still is not, because that is a "
            "survey copy per audit rather than a sentence."
        ),
    )


class AuditEventRecordResponse(BaseModel):
    status: str = "ok"
    id: Optional[str] = None


class AuditEventsRequest(_AuditScope):
    study_id: Optional[UUID] = Field(
        default=None, description="Study scope; omit for the tenant-wide feed.",
    )
    limit: int = Field(default=100, ge=1, le=500)


class AuditEvent(BaseModel):
    id: Optional[str] = None
    study_id: Optional[str] = None
    occurred_at: Optional[str] = None
    actor_ref: Optional[str] = None
    request_id: Optional[str] = None
    goal_version: Optional[int] = None
    alignment_score: Optional[float] = None
    quality_score: Optional[float] = None
    status: Optional[str] = None
    degraded: List[str] = Field(default_factory=list)
    finding_counts: Dict[str, Any] = Field(default_factory=dict)
    payload_digest: Optional[str] = None
    summary: Optional[str] = None


class AuditEventsResponse(BaseModel):
    events: List[AuditEvent] = Field(default_factory=list)


class AuditRetentionStatus(BaseModel):
    """Is the one-year window actually holding?

    `expired_count` is the number that matters and it should always be 0. Deliberately
    not a last-ran timestamp: that tells you the purge fired, not that it is keeping up,
    and the two come apart exactly when it matters. A non-zero count is true whether the
    schedule was never installed, installed and errored, or ran and fell behind.
    """

    total_count: int = 0
    expired_count: int = 0
    oldest_occurred_at: Optional[str] = None
    window_starts_at: Optional[str] = None


class AuditEventsPurgeResponse(BaseModel):
    """What the purge removed, plus where retention stands afterwards.

    Returning the post-purge status means one call answers both "did it work" and "is it
    now clean", so a scheduler's log line is enough to audit the control.
    """

    deleted: int = 0
    status: AuditRetentionStatus = Field(default_factory=AuditRetentionStatus)

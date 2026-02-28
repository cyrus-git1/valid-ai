from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ._time import utcnow

JsonDict = Dict[str, Any]


# ── Video transcript (Daily.js WebVTT) ──────────────────────────────────────

class VideoTranscriptCreate(BaseModel):
    """Intent to store a Daily.js video session transcript (WebVTT source)."""
    tenant_id: UUID
    client_id: Optional[UUID] = None

    source_uri: Optional[str] = None
    title: Optional[str] = None
    metadata: JsonDict = Field(default_factory=dict)


class VideoTranscriptRow(VideoTranscriptCreate):
    id: UUID
    created_at: datetime
    updated_at: datetime


# ── Chat transcript (Daily.js text chat) ─────────────────────────────────────

class ChatTranscriptCreate(BaseModel):
    """Intent to store a Daily.js text-chat transcript."""
    tenant_id: UUID
    client_id: Optional[UUID] = None

    source_uri: Optional[str] = None
    metadata: JsonDict = Field(default_factory=dict)


class ChatTranscriptRow(ChatTranscriptCreate):
    id: UUID
    created_at: datetime
    ended_at: Optional[datetime] = None


# ── WebVTT cue (parsed from .vtt file) ──────────────────────────────────────

class VTTCue(BaseModel):
    """A single parsed cue from a WebVTT file."""
    index: int
    start_seconds: float
    end_seconds: float
    speaker: Optional[str] = None
    text: str

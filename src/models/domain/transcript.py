from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import TenantOwned
from src.models.domain._time import TimestampMixin

JsonDict = Dict[str, Any]


# ── Video transcript (Daily.js WebVTT) ──────────────────────────────────────

class VideoTranscriptCreate(TenantOwned):
    """Intent to store a Daily.js video session transcript (WebVTT source)."""
    source_uri: Optional[str] = None
    title: Optional[str] = None
    metadata: JsonDict = Field(default_factory=dict)


class VideoTranscriptRow(VideoTranscriptCreate, TimestampMixin):
    id: UUID


# ── Chat transcript (Daily.js text chat) ─────────────────────────────────────

class ChatTranscriptCreate(TenantOwned):
    """Intent to store a Daily.js text-chat transcript."""
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

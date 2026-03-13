"""Pydantic models for the /transcription router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single timed cue from the WebVTT transcript."""
    index: int
    start: str = Field(description="Start timestamp (HH:MM:SS.mmm)")
    end: str = Field(description="End timestamp (HH:MM:SS.mmm)")
    text: str = Field(description="Spoken text for this segment")


class TranscriptionResponse(BaseModel):
    tenant_id: UUID
    survey_id: UUID
    vtt: str = Field(description="Raw WebVTT formatted transcript")
    segments: List[TranscriptSegment] = Field(
        description="Parsed transcript segments with timestamps"
    )
    full_text: str = Field(description="Plain-text transcript (no timestamps)")
    metadata: Dict[str, Any] = Field(default_factory=dict)

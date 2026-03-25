"""Pydantic models for the /transcription router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SpeakerLogEntry(BaseModel):
    """A single turn from the platform's speaker log."""
    name: str
    role: str = Field(description="'local' or 'remote'")
    peerId: str
    startedAt: str = Field(description="ISO 8601 timestamp")
    endedAt: str = Field(description="ISO 8601 timestamp")
    durationMs: int


class TranscriptSegment(BaseModel):
    """A single timed cue from the WebVTT transcript, with speaker annotation."""
    index: int
    start: str = Field(description="Start timestamp (HH:MM:SS.mmm)")
    end: str = Field(description="End timestamp (HH:MM:SS.mmm)")
    speaker: Optional[str] = Field(default=None, description="Speaker label from diarization")
    text: str = Field(description="Spoken text for this segment")
    crosstalk: bool = Field(default=False, description="True if crosstalk/overlap detected by Asteroid")


class SpeakerEmbedding(BaseModel):
    """A speaker-attributed chunk with its pyannote embedding vector."""
    chunk_index: int
    speaker: str
    role: str = Field(description="'local' or 'remote'")
    start: str = Field(description="Start timestamp (HH:MM:SS.mmm)")
    end: str = Field(description="End timestamp (HH:MM:SS.mmm)")
    text: str
    embedding: List[float] = Field(description="Speaker embedding vector from pyannote")


class SpeakerStats(BaseModel):
    """Aggregated statistics for a single speaker."""
    peerId: Optional[str] = Field(default=None, description="Peer ID if available")
    role: str = Field(description="'local' or 'remote'")
    total_duration_ms: int
    turn_count: int
    talk_ratio: float
    avg_turn_duration_ms: float


class CrosstalkFlag(BaseModel):
    """A region flagged as containing overlapping speech by Asteroid."""
    start: str = Field(description="Start timestamp (HH:MM:SS.mmm)")
    end: str = Field(description="End timestamp (HH:MM:SS.mmm)")
    speakers: List[str] = Field(description="Speakers involved in the overlap")
    confidence: float = Field(description="Asteroid overlap detection confidence")


class TranscriptionResult(BaseModel):
    """Combined output of the pyannote->asteroid->whisper pipeline (service layer)."""
    vtt: str = Field(description="Raw WebVTT string with speaker annotations")
    segments: List[TranscriptSegment] = Field(description="Parsed timed segments with speakers")
    full_text: str = Field(description="Plain-text concatenation")
    embeddings: List[SpeakerEmbedding] = Field(default_factory=list, description="Per-chunk speaker embeddings")
    speaker_stats: Dict[str, SpeakerStats] = Field(default_factory=dict, description="Per-speaker aggregated stats")
    crosstalk_flags: List[CrosstalkFlag] = Field(default_factory=list, description="Regions with overlapping speech")


class TranscriptionResponse(BaseModel):
    tenant_id: UUID
    survey_id: UUID
    vtt: str = Field(description="Raw WebVTT formatted transcript with speaker annotations")
    segments: List[TranscriptSegment] = Field(
        description="Parsed transcript segments with timestamps and speakers"
    )
    full_text: str = Field(description="Plain-text transcript (no timestamps)")
    embeddings: List[SpeakerEmbedding] = Field(
        default_factory=list, description="Per-chunk speaker embeddings from pyannote"
    )
    speaker_stats: Dict[str, SpeakerStats] = Field(
        default_factory=dict, description="Per-speaker aggregated statistics"
    )
    crosstalk_flags: List[CrosstalkFlag] = Field(
        default_factory=list, description="Overlapping speech regions detected by Asteroid"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

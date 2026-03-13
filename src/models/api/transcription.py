"""Pydantic models for the /transcription router."""
from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    tenant_id: UUID
    survey_id: UUID
    transcript: str = Field(description="WebVTT formatted transcript")
    metadata: Dict[str, Any] = Field(default_factory=dict)

"""
/transcription router
---------------------
Audio-to-WebVTT transcription via OpenAI Whisper.

POST /transcription/generate — Upload M4A audio, get back WebVTT + parsed JSON
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.models.api.transcription import TranscriptionResponse, TranscriptSegment
from src.services.transcription_service import TranscriptionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcription", tags=["transcription"])


@router.post("/generate", response_model=TranscriptionResponse)
async def generate_transcription(
    audio_file: UploadFile = File(..., description="M4A audio file to transcribe"),
    tenant_id: UUID = Form(...),
    survey_id: UUID = Form(...),
    metadata: str = Form(default="{}", description="JSON-encoded metadata"),
) -> TranscriptionResponse:
    """
    Transcribe an M4A audio file to WebVTT using OpenAI Whisper (medium model).

    Returns:
      - tenant_id, survey_id, metadata  — echoed back
      - vtt       — raw WebVTT string (can be saved as .vtt file)
      - segments  — parsed JSON array of {index, start, end, text}
      - full_text — plain-text transcript without timestamps
    """
    # Validate file type
    file_name = audio_file.filename or "audio.m4a"
    if not file_name.lower().endswith((".m4a", ".mp3", ".wav", ".webm", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_name}. Supported: m4a, mp3, wav, webm, mp4, mpeg, mpga, oga, ogg",
        )

    # Parse metadata JSON
    try:
        meta_dict: Dict[str, Any] = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON")

    file_bytes = await audio_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    svc = TranscriptionService()

    try:
        result = svc.transcribe(
            file_bytes=file_bytes,
            file_name=file_name,
        )
    except Exception as e:
        logger.exception("Transcription failed for %s", file_name)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return TranscriptionResponse(
        tenant_id=tenant_id,
        survey_id=survey_id,
        vtt=result.vtt,
        segments=[
            TranscriptSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            for seg in result.segments
        ],
        full_text=result.full_text,
        metadata=meta_dict,
    )

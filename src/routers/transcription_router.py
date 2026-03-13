"""
/transcription router
---------------------
Audio/Video-to-WebVTT transcription via OpenAI Whisper.

POST /transcription/generate/audio — Upload M4A audio, get back WebVTT + parsed JSON
POST /transcription/generate/video — Upload MP4 video, get back WebVTT + parsed JSON
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


def _build_response(
    result, tenant_id: UUID, survey_id: UUID, meta_dict: Dict[str, Any]
) -> TranscriptionResponse:
    """Build a TranscriptionResponse from a TranscriptionResult."""
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


async def _parse_metadata(metadata: str) -> Dict[str, Any]:
    """Parse and validate JSON metadata string."""
    try:
        return json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON")


async def _transcribe_file(file_bytes: bytes, file_name: str):
    """Run transcription via Whisper and return the result."""
    svc = TranscriptionService()
    try:
        return svc.transcribe(file_bytes=file_bytes, file_name=file_name)
    except Exception as e:
        logger.exception("Transcription failed for %s", file_name)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@router.post("/generate/audio", response_model=TranscriptionResponse)
async def generate_audio_transcription(
    audio_file: UploadFile = File(..., description="M4A audio file to transcribe"),
    tenant_id: UUID = Form(...),
    survey_id: UUID = Form(...),
    metadata: str = Form(default="{}", description="JSON-encoded metadata"),
) -> TranscriptionResponse:
    """
    Transcribe an M4A audio file to WebVTT using OpenAI Whisper.

    Returns:
      - tenant_id, survey_id, metadata  — echoed back
      - vtt       — raw WebVTT string (can be saved as .vtt file)
      - segments  — parsed JSON array of {index, start, end, text}
      - full_text — plain-text transcript without timestamps
    """
    file_name = audio_file.filename or "audio.m4a"
    if not file_name.lower().endswith((".m4a", ".mp3", ".wav", ".webm", ".mpeg", ".mpga", ".oga", ".ogg")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_name}. Supported: m4a, mp3, wav, webm, mpeg, mpga, oga, ogg",
        )

    meta_dict = await _parse_metadata(metadata)

    file_bytes = await audio_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    result = await _transcribe_file(file_bytes, file_name)
    return _build_response(result, tenant_id, survey_id, meta_dict)


@router.post("/generate/video", response_model=TranscriptionResponse)
async def generate_video_transcription(
    video_file: UploadFile = File(..., description="MP4 video file to transcribe"),
    tenant_id: UUID = Form(...),
    survey_id: UUID = Form(...),
    metadata: str = Form(default="{}", description="JSON-encoded metadata"),
) -> TranscriptionResponse:
    """
    Transcribe an MP4 video file to WebVTT using OpenAI Whisper.

    Returns:
      - tenant_id, survey_id, metadata  — echoed back
      - vtt       — raw WebVTT string (can be saved as .vtt file)
      - segments  — parsed JSON array of {index, start, end, text}
      - full_text — plain-text transcript without timestamps
    """
    file_name = video_file.filename or "video.mp4"
    if not file_name.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {file_name}. Supported: mp4",
        )

    meta_dict = await _parse_metadata(metadata)

    file_bytes = await video_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Video file is empty")

    result = await _transcribe_file(file_bytes, file_name)
    return _build_response(result, tenant_id, survey_id, meta_dict)

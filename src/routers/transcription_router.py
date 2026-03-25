"""
/transcription router
---------------------
Audio/Video-to-WebVTT transcription via pyannote API (diarization) + whisper pipeline.

POST /transcription/generate/audio — Upload audio, get back speaker-annotated WebVTT + embeddings
POST /transcription/generate/video — Upload MP4 video, get back speaker-annotated WebVTT + embeddings
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.models.api.transcription import (
    SpeakerLogEntry,
    TranscriptionResponse,
)
from src.services.transcription_service import get_transcription_service

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
        segments=result.segments,
        full_text=result.full_text,
        embeddings=result.embeddings,
        speaker_stats=result.speaker_stats,
        crosstalk_flags=result.crosstalk_flags,
        metadata=meta_dict,
    )


async def _parse_json_field(value: str, field_name: str):
    """Parse and validate a JSON form field."""
    try:
        return json.loads(value) if value else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"{field_name} must be valid JSON")


def _parse_speaker_log(raw: Any) -> Optional[List[SpeakerLogEntry]]:
    """Validate and convert raw speaker_log JSON into model instances."""
    if not raw:
        return None
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="speaker_log must be a JSON array")
    try:
        return [SpeakerLogEntry(**entry) for entry in raw]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid speaker_log entry: {e}")


async def _transcribe_file(
    file_bytes: bytes,
    file_name: str,
    speaker_log: Optional[List[SpeakerLogEntry]] = None,
):
    """Run the full transcription pipeline and return the result."""
    svc = get_transcription_service()
    try:
        return svc.transcribe(
            file_bytes=file_bytes,
            file_name=file_name,
            speaker_log=speaker_log,
        )
    except Exception as e:
        logger.exception("Transcription failed for %s", file_name)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@router.post("/generate/audio", response_model=TranscriptionResponse)
async def generate_audio_transcription(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    tenant_id: UUID = Form(...),
    survey_id: UUID = Form(...),
    metadata: str = Form(default="{}", description="JSON-encoded metadata"),
    speaker_log: str = Form(
        default="[]",
        description="JSON array of speaker log entries with name, role, peerId, startedAt, endedAt, durationMs",
    ),
) -> TranscriptionResponse:
    """
    Transcribe an audio file using pyannote API (diarization) + whisper pipeline.

    The speaker_log from the video call platform is used to map pyannote's
    generic labels (SPEAKER_00, etc.) to real participant names via timing overlap.

    Returns speaker-annotated WebVTT, per-chunk embeddings, speaker stats,
    and crosstalk flags.
    """
    file_name = audio_file.filename or "audio.m4a"
    if not file_name.lower().endswith((".m4a", ".mp3", ".wav", ".webm", ".mpeg", ".mpga", ".oga", ".ogg")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_name}. Supported: m4a, mp3, wav, webm, mpeg, mpga, oga, ogg",
        )

    meta_dict = await _parse_json_field(metadata, "metadata") or {}
    raw_log = await _parse_json_field(speaker_log, "speaker_log")
    parsed_log = _parse_speaker_log(raw_log)

    file_bytes = await audio_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    result = await _transcribe_file(file_bytes, file_name, speaker_log=parsed_log)
    return _build_response(result, tenant_id, survey_id, meta_dict)


@router.post("/generate/video", response_model=TranscriptionResponse)
async def generate_video_transcription(
    video_file: UploadFile = File(..., description="MP4 video file to transcribe"),
    tenant_id: UUID = Form(...),
    survey_id: UUID = Form(...),
    metadata: str = Form(default="{}", description="JSON-encoded metadata"),
    speaker_log: str = Form(
        default="[]",
        description="JSON array of speaker log entries with name, role, peerId, startedAt, endedAt, durationMs",
    ),
) -> TranscriptionResponse:
    """
    Transcribe an MP4 video file using pyannote API (diarization) + whisper pipeline.

    The speaker_log from the video call platform is used to map pyannote's
    generic labels (SPEAKER_00, etc.) to real participant names via timing overlap.

    Returns speaker-annotated WebVTT, per-chunk embeddings, speaker stats,
    and crosstalk flags.
    """
    file_name = video_file.filename or "video.mp4"
    if not file_name.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {file_name}. Supported: mp4",
        )

    meta_dict = await _parse_json_field(metadata, "metadata") or {}
    raw_log = await _parse_json_field(speaker_log, "speaker_log")
    parsed_log = _parse_speaker_log(raw_log)

    file_bytes = await video_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Video file is empty")

    result = await _transcribe_file(file_bytes, file_name, speaker_log=parsed_log)
    return _build_response(result, tenant_id, survey_id, meta_dict)

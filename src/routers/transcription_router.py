"""
/transcription router
---------------------
Audio-to-WebVTT transcription via OpenAI Whisper.

POST /transcription/generate — Upload M4A audio, get back a .vtt file download
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from src.services.transcription_service import TranscriptionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcription", tags=["transcription"])


@router.post("/generate")
async def generate_transcription(
    audio_file: UploadFile = File(..., description="M4A audio file to transcribe"),
    tenant_id: UUID = Form(...),
    survey_id: UUID = Form(...),
    metadata: str = Form(default="{}", description="JSON-encoded metadata"),
) -> Response:
    """
    Transcribe an M4A audio file to WebVTT using OpenAI Whisper (medium model).

    Accepts an audio file upload and returns the transcript as a downloadable
    .vtt file. tenant_id, survey_id, and metadata are included in response headers.
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
        transcript_vtt = svc.transcribe(
            file_bytes=file_bytes,
            file_name=file_name,
        )
    except Exception as e:
        logger.exception("Transcription failed for %s", file_name)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # Build .vtt filename from the original audio filename
    vtt_filename = Path(file_name).stem + ".vtt"

    return Response(
        content=transcript_vtt,
        media_type="text/vtt",
        headers={
            "Content-Disposition": f'attachment; filename="{vtt_filename}"',
            "X-Tenant-Id": str(tenant_id),
            "X-Survey-Id": str(survey_id),
            "X-Metadata": json.dumps(meta_dict),
        },
    )

"""
HTTP client for the Transcription Service.

Used by the main app to call the standalone transcription service
instead of importing TranscriptionService directly.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

TRANSCRIPTION_SERVICE_URL = os.environ.get("TRANSCRIPTION_SERVICE_URL", "http://localhost:8001")


def transcribe_audio(
    *,
    file_bytes: bytes,
    file_name: str,
    tenant_id: str,
    survey_id: str,
    speaker_log: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the transcription service to transcribe an audio file."""
    return _call_transcription(
        endpoint="/transcription/generate/audio",
        file_field_name="audio_file",
        file_bytes=file_bytes,
        file_name=file_name,
        tenant_id=tenant_id,
        survey_id=survey_id,
        speaker_log=speaker_log,
        metadata=metadata,
    )


def transcribe_video(
    *,
    file_bytes: bytes,
    file_name: str,
    tenant_id: str,
    survey_id: str,
    speaker_log: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the transcription service to transcribe a video file."""
    return _call_transcription(
        endpoint="/transcription/generate/video",
        file_field_name="video_file",
        file_bytes=file_bytes,
        file_name=file_name,
        tenant_id=tenant_id,
        survey_id=survey_id,
        speaker_log=speaker_log,
        metadata=metadata,
    )


def _call_transcription(
    *,
    endpoint: str,
    file_field_name: str,
    file_bytes: bytes,
    file_name: str,
    tenant_id: str,
    survey_id: str,
    speaker_log: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send a multipart request to the transcription service and return raw JSON."""
    url = f"{TRANSCRIPTION_SERVICE_URL}{endpoint}"

    files = {file_field_name: (file_name, file_bytes)}
    data = {
        "tenant_id": tenant_id,
        "survey_id": survey_id,
        "metadata": json.dumps(metadata or {}),
        "speaker_log": json.dumps(speaker_log or []),
    }

    with httpx.Client(timeout=900) as client:
        resp = client.post(url, files=files, data=data)
        resp.raise_for_status()
        return resp.json()

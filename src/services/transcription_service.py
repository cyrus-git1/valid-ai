"""
src/services/transcription_service.py
--------------------------------------
Audio-to-WebVTT transcription via OpenAI Whisper API.

Uses the whisper-1 model (medium) to transcribe M4A audio files
and returns structured WebVTT output.

Import
------
    from src.services.transcription_service import TranscriptionService
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

import dotenv
from openai import OpenAI

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class TranscriptionService:
    """Transcribe audio files to WebVTT using OpenAI Whisper."""

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)

    def transcribe(
        self,
        *,
        file_bytes: bytes,
        file_name: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio bytes to WebVTT format using Whisper.

        Parameters
        ----------
        file_bytes : bytes
            Raw audio file content (M4A, MP3, WAV, etc.)
        file_name : str
            Original filename (used for the temp file extension).
        language : str, optional
            ISO-639-1 language code hint (e.g. "en"). Auto-detected if omitted.

        Returns
        -------
        str
            The transcript in WebVTT format.
        """
        suffix = Path(file_name).suffix or ".m4a"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            logger.info(
                "Transcribing %s (%d bytes) with Whisper medium model",
                file_name, len(file_bytes),
            )

            kwargs: Dict[str, Any] = {
                "model": "whisper-1",
                "file": open(tmp_path, "rb"),
                "response_format": "vtt",
            }
            if language:
                kwargs["language"] = language

            transcript_vtt: str = self._client.audio.transcriptions.create(**kwargs)

            logger.info(
                "Transcription complete for %s — %d chars of VTT",
                file_name, len(transcript_vtt),
            )
            return transcript_vtt

        finally:
            Path(tmp_path).unlink(missing_ok=True)

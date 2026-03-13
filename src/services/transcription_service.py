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
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import dotenv
from openai import OpenAI

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single timed segment from the transcript."""
    index: int
    start: str          # "00:00:01.234"
    end: str            # "00:00:05.678"
    text: str


@dataclass
class TranscriptionResult:
    """Combined output of a Whisper transcription."""
    vtt: str                                # raw WebVTT string
    segments: List[TranscriptSegment]       # parsed timed segments
    full_text: str                          # plain-text concatenation


# Regex to parse WebVTT cues:  "00:00:01.000 --> 00:00:04.500\nSome text"
_VTT_CUE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n(.*?)(?=\n\n|\Z)",
    re.DOTALL,
)


def _parse_vtt(vtt: str) -> List[TranscriptSegment]:
    """Parse a WebVTT string into a list of TranscriptSegments."""
    segments: List[TranscriptSegment] = []
    for idx, match in enumerate(_VTT_CUE_RE.finditer(vtt), start=1):
        segments.append(TranscriptSegment(
            index=idx,
            start=match.group(1),
            end=match.group(2),
            text=match.group(3).strip(),
        ))
    return segments


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
    ) -> TranscriptionResult:
        """
        Transcribe audio bytes using Whisper and return VTT + parsed JSON.

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
        TranscriptionResult
            Contains the raw VTT string, parsed segments, and full plain text.
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

            segments = _parse_vtt(transcript_vtt)
            full_text = " ".join(seg.text for seg in segments)

            return TranscriptionResult(
                vtt=transcript_vtt,
                segments=segments,
                full_text=full_text,
            )

        finally:
            Path(tmp_path).unlink(missing_ok=True)

"""
src/services/transcription_service.py
--------------------------------------
Audio/Video-to-WebVTT transcription pipeline:

    whisper (speech-to-text via OpenAI API)
    -> GPT-4o-mini (match speaker_log to VTT segments)
    -> speaker-annotated WebVTT

Import
------
    from src.services.transcription_service import TranscriptionService
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dotenv
from openai import OpenAI

from src.models.api.transcription import (
    CrosstalkFlag,
    SpeakerEmbedding,
    SpeakerLogEntry,
    SpeakerStats,
    TranscriptSegment,
    TranscriptionResult,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Regex to parse WebVTT cues
_VTT_CUE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n(.*?)(?=\n\n|\Z)",
    re.DOTALL,
)

# ---------- timestamp helpers ----------

def _ts_to_ms(ts: str) -> int:
    """Convert 'HH:MM:SS.mmm' to milliseconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


def _ms_to_ts(ms: int) -> str:
    """Convert milliseconds to 'HH:MM:SS.mmm'."""
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    millis = ms % 1_000
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"


def _parse_vtt(vtt: str) -> List[dict]:
    """Parse a WebVTT string into a list of raw dicts (start, end, text)."""
    segments = []
    for match in _VTT_CUE_RE.finditer(vtt):
        segments.append({
            "start": match.group(1),
            "end": match.group(2),
            "text": match.group(3).strip(),
        })
    return segments


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Return overlap in ms between two intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


# ---------- crosstalk detection from speaker_log ----------

def _speaker_log_to_turns(
    speaker_log: List[SpeakerLogEntry],
) -> Tuple[List[dict], Dict[str, str], Dict[str, str]]:
    """Convert speaker_log (absolute ISO timestamps) to relative ms turns."""
    if not speaker_log:
        return [], {}, {}

    rec_start = datetime.fromisoformat(
        min(e.startedAt for e in speaker_log).replace("Z", "+00:00")
    )

    turns: List[dict] = []
    roles: Dict[str, str] = {}
    peer_ids: Dict[str, str] = {}

    for entry in speaker_log:
        entry_start = datetime.fromisoformat(entry.startedAt.replace("Z", "+00:00"))
        entry_end = datetime.fromisoformat(entry.endedAt.replace("Z", "+00:00"))
        start_ms = max(0, int((entry_start - rec_start).total_seconds() * 1000))
        end_ms = max(0, int((entry_end - rec_start).total_seconds() * 1000))
        turns.append({
            "speaker": entry.name,
            "start_ms": start_ms,
            "end_ms": end_ms,
        })
        roles[entry.name] = entry.role
        peer_ids[entry.name] = entry.peerId

    return turns, roles, peer_ids


def _detect_crosstalk(speaker_turns: List[dict]) -> List[CrosstalkFlag]:
    """Detect crosstalk from overlapping speaker_log turns (>=200ms)."""
    candidates: List[dict] = []
    for i, t1 in enumerate(speaker_turns):
        for t2 in speaker_turns[i + 1:]:
            if t1["speaker"] == t2["speaker"]:
                continue
            ov_start = max(t1["start_ms"], t2["start_ms"])
            ov_end = min(t1["end_ms"], t2["end_ms"])
            if ov_end > ov_start and (ov_end - ov_start) >= 200:
                candidates.append({
                    "start_ms": ov_start,
                    "end_ms": ov_end,
                    "speakers": sorted({t1["speaker"], t2["speaker"]}),
                })

    if not candidates:
        return []

    candidates.sort(key=lambda c: c["start_ms"])
    merged: List[dict] = [candidates[0]]
    for c in candidates[1:]:
        prev = merged[-1]
        if c["start_ms"] <= prev["end_ms"]:
            prev["end_ms"] = max(prev["end_ms"], c["end_ms"])
            prev["speakers"] = sorted(set(prev["speakers"]) | set(c["speakers"]))
        else:
            merged.append(c)

    flags: List[CrosstalkFlag] = []
    for region in merged:
        overlap_ms = region["end_ms"] - region["start_ms"]
        confidence = min(1.0, overlap_ms / 2000.0)
        flags.append(CrosstalkFlag(
            start=_ms_to_ts(region["start_ms"]),
            end=_ms_to_ts(region["end_ms"]),
            speakers=region["speakers"],
            confidence=round(confidence, 4),
        ))

    logger.info("Crosstalk detection: %d flags from speaker_log overlaps", len(flags))
    return flags


# ---------- singleton ----------

_singleton_instance: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Return a singleton TranscriptionService."""
    global _singleton_instance
    if _singleton_instance is None:
        _singleton_instance = TranscriptionService()
    return _singleton_instance


_SPEAKER_MATCH_PROMPT = """\
You are a speaker attribution engine. You receive two inputs:

1. A WebVTT transcript with numbered segments (index, start time, end time, text).
2. A speaker_log from a video call platform showing who was speaking at what times.

Your job: assign the correct speaker name from the speaker_log to each VTT segment \
based on timing overlap and conversational context.

Rules:
- Each segment gets exactly one speaker.
- Use the speaker_log timing as the primary signal. The speaker whose talk turn \
overlaps the most with a VTT segment's time range is usually the correct speaker.
- Use conversational context (questions vs answers, topic continuity) as a secondary \
signal when timing is ambiguous.
- If a segment has no clear speaker match, assign the most likely speaker based on context.

Return ONLY a JSON array where each element is:
{"index": <segment index>, "speaker": "<speaker name from speaker_log>"}

No explanation, no markdown, just the JSON array."""


class TranscriptionService:
    """
    Transcribe audio/video files using:
    whisper (OpenAI API) -> GPT-4o-mini (speaker matching) -> annotated WebVTT

    No local ML models required.
    """

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)

    def _run_whisper(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio with Whisper and return raw VTT string."""
        logger.info("Running Whisper transcription...")
        kwargs: Dict[str, Any] = {
            "model": "whisper-1",
            "file": open(audio_path, "rb"),
            "response_format": "vtt",
        }
        if language:
            kwargs["language"] = language

        vtt: str = self._client.audio.transcriptions.create(**kwargs)
        logger.info("Whisper transcription complete — %d chars of VTT", len(vtt))
        return vtt

    def _match_speakers_with_llm(
        self,
        raw_segments: List[dict],
        speaker_log: List[SpeakerLogEntry],
    ) -> Dict[int, str]:
        """
        Use GPT-4o-mini to match VTT segments to speakers from the speaker_log.

        Returns a dict mapping segment index (1-based) to speaker name.
        """
        # Build the VTT summary for the prompt
        vtt_lines = []
        for idx, seg in enumerate(raw_segments, start=1):
            vtt_lines.append(f"[{idx}] {seg['start']} --> {seg['end']}: {seg['text']}")
        vtt_text = "\n".join(vtt_lines)

        # Build the speaker_log summary
        rec_start = datetime.fromisoformat(
            min(e.startedAt for e in speaker_log).replace("Z", "+00:00")
        )
        log_lines = []
        for entry in speaker_log:
            entry_start = datetime.fromisoformat(entry.startedAt.replace("Z", "+00:00"))
            entry_end = datetime.fromisoformat(entry.endedAt.replace("Z", "+00:00"))
            start_sec = (entry_start - rec_start).total_seconds()
            end_sec = (entry_end - rec_start).total_seconds()
            start_ts = _ms_to_ts(int(start_sec * 1000))
            end_ts = _ms_to_ts(int(end_sec * 1000))
            log_lines.append(f"{entry.name} ({entry.role}): {start_ts} --> {end_ts}")
        log_text = "\n".join(log_lines)

        user_msg = f"## WebVTT Segments\n{vtt_text}\n\n## Speaker Log\n{log_text}"

        logger.info("Matching speakers with GPT-4o-mini...")
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SPEAKER_MATCH_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )

        result_text = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if result_text.startswith("```"):
            result_text = re.sub(r"^```\w*\n?", "", result_text)
            result_text = re.sub(r"\n?```$", "", result_text)

        try:
            assignments = json.loads(result_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse speaker assignments: %s", result_text)
            return {}

        mapping: Dict[int, str] = {}
        for item in assignments:
            mapping[item["index"]] = item["speaker"]

        logger.info("Speaker matching complete: %d segments assigned", len(mapping))
        return mapping

    def _build_annotated_output(
        self,
        raw_segments: List[dict],
        speaker_map: Dict[int, str],
        crosstalk_flags: List[CrosstalkFlag],
    ) -> Tuple[List[TranscriptSegment], str]:
        """Build speaker-annotated segments and WebVTT."""
        ct_regions = [
            (_ts_to_ms(f.start), _ts_to_ms(f.end))
            for f in crosstalk_flags
        ]

        segments: List[TranscriptSegment] = []
        vtt_lines = ["WEBVTT", ""]

        for idx, raw in enumerate(raw_segments, start=1):
            seg_start_ms = _ts_to_ms(raw["start"])
            seg_end_ms = _ts_to_ms(raw["end"])

            speaker = speaker_map.get(idx)

            is_crosstalk = any(
                _overlap(seg_start_ms, seg_end_ms, ct_start, ct_end) > 0
                for ct_start, ct_end in ct_regions
            )

            segments.append(TranscriptSegment(
                index=idx,
                start=raw["start"],
                end=raw["end"],
                speaker=speaker,
                text=raw["text"],
                crosstalk=is_crosstalk,
            ))

            vtt_lines.append(str(idx))
            vtt_lines.append(f"{raw['start']} --> {raw['end']}")
            prefix = f"<v {speaker}>" if speaker else ""
            suffix = "</v>" if speaker else ""
            ct_tag = " [crosstalk]" if is_crosstalk else ""
            vtt_lines.append(f"{prefix}{raw['text']}{suffix}{ct_tag}")
            vtt_lines.append("")

        vtt_str = "\n".join(vtt_lines)
        return segments, vtt_str

    def _compute_speaker_stats(
        self,
        segments: List[TranscriptSegment],
        speaker_roles: Dict[str, str],
        speaker_peer_ids: Dict[str, str],
    ) -> Dict[str, SpeakerStats]:
        """Compute aggregated stats per speaker."""
        stats: Dict[str, dict] = {}
        for seg in segments:
            spk = seg.speaker or "UNKNOWN"
            if spk not in stats:
                stats[spk] = {"total_ms": 0, "turn_count": 0}
            dur = _ts_to_ms(seg.end) - _ts_to_ms(seg.start)
            stats[spk]["total_ms"] += dur
            stats[spk]["turn_count"] += 1

        total_all = sum(s["total_ms"] for s in stats.values()) or 1

        result: Dict[str, SpeakerStats] = {}
        for spk, s in stats.items():
            result[spk] = SpeakerStats(
                peerId=speaker_peer_ids.get(spk),
                role=speaker_roles.get(spk, "remote"),
                total_duration_ms=s["total_ms"],
                turn_count=s["turn_count"],
                talk_ratio=round(s["total_ms"] / total_all, 4),
                avg_turn_duration_ms=round(s["total_ms"] / s["turn_count"], 1) if s["turn_count"] else 0,
            )
        return result

    def transcribe(
        self,
        *,
        file_bytes: bytes,
        file_name: str,
        language: Optional[str] = None,
        speaker_log: Optional[List[SpeakerLogEntry]] = None,
    ) -> TranscriptionResult:
        """
        Transcription pipeline: whisper -> GPT-4o-mini speaker matching -> WebVTT.
        """
        suffix = Path(file_name).suffix or ".m4a"
        speaker_log = speaker_log or []

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            logger.info(
                "Starting transcription pipeline for %s (%d bytes)",
                file_name, len(file_bytes),
            )

            # Step 1: Whisper transcription
            raw_vtt = self._run_whisper(tmp_path, language=language)
            raw_segments = _parse_vtt(raw_vtt)

            # Step 2: Speaker matching via GPT-4o-mini
            speaker_map: Dict[int, str] = {}
            roles: Dict[str, str] = {}
            peer_ids: Dict[str, str] = {}

            if speaker_log:
                speaker_map = self._match_speakers_with_llm(raw_segments, speaker_log)

                # Build role/peerId lookups from speaker_log
                for entry in speaker_log:
                    roles[entry.name] = entry.role
                    peer_ids[entry.name] = entry.peerId

            # Step 3: Detect crosstalk from speaker_log timing overlaps
            speaker_turns, _, _ = _speaker_log_to_turns(speaker_log)
            crosstalk_flags = _detect_crosstalk(speaker_turns)

            # Step 4: Build annotated output
            segments, annotated_vtt = self._build_annotated_output(
                raw_segments, speaker_map, crosstalk_flags,
            )

            full_text = " ".join(seg.text for seg in segments)

            # Build embeddings structure (empty vectors — no pyannote)
            embeddings = [
                SpeakerEmbedding(
                    chunk_index=idx,
                    speaker=seg.speaker or "UNKNOWN",
                    role=roles.get(seg.speaker or "UNKNOWN", "remote"),
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    embedding=[],
                )
                for idx, seg in enumerate(segments)
            ]

            # Compute speaker stats
            stats = self._compute_speaker_stats(segments, roles, peer_ids)

            logger.info(
                "Pipeline complete for %s — %d segments, %d speakers, %d crosstalk flags",
                file_name, len(segments), len(stats), len(crosstalk_flags),
            )

            return TranscriptionResult(
                vtt=annotated_vtt,
                segments=segments,
                full_text=full_text,
                embeddings=embeddings,
                speaker_stats=stats,
                crosstalk_flags=crosstalk_flags,
            )

        finally:
            Path(tmp_path).unlink(missing_ok=True)

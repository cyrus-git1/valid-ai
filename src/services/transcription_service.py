"""
src/services/transcription_service.py
--------------------------------------
Audio/Video-to-WebVTT transcription pipeline:

    pyannote API (speaker diarization)
    + whisper (speech-to-text via OpenAI API)
    -> merge diarization with VTT segments
    -> speaker_log name mapping
    -> speaker-annotated WebVTT

Import
------
    from src.services.transcription_service import TranscriptionService
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import httpx
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

_PYANNOTE_API_BASE = "https://api.pyannote.ai/v1"
_PYANNOTE_POLL_INTERVAL = 5  # seconds between status checks
_PYANNOTE_MAX_WAIT = 600  # max seconds to wait for a job

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


def _sec_to_ts(sec: float) -> str:
    """Convert seconds (float) to 'HH:MM:SS.mmm'."""
    return _ms_to_ts(int(sec * 1000))


def _parse_vtt(vtt: str) -> List[dict]:
    """Parse a WebVTT string into a list of raw dicts (start, end, text)."""
    segments = []
    for match in _VTT_CUE_RE.finditer(vtt):
        segments.append({
            "start": match.group(1),
            "end": match.group(2),
            "text": match.group(3).strip(),
        })
    return _clean_whisper_segments(segments)


def _clean_whisper_segments(segments: List[dict]) -> List[dict]:
    """Remove Whisper hallucination artifacts: zero-duration and duplicate trailing segments."""
    if not segments:
        return segments

    cleaned = []
    seen_texts: set = set()

    for seg in segments:
        # Drop zero-duration segments
        if seg["start"] == seg["end"]:
            continue

        # Drop segments shorter than 100ms (likely artifacts)
        if _ts_to_ms(seg["end"]) - _ts_to_ms(seg["start"]) < 100:
            continue

        # Drop trailing duplicates (same or very similar text already seen)
        text_norm = seg["text"].strip().lower()
        if text_norm in seen_texts:
            continue

        seen_texts.add(text_norm)
        cleaned.append(seg)

    if len(cleaned) < len(segments):
        logger.info(
            "Whisper cleanup: removed %d hallucinated segments",
            len(segments) - len(cleaned),
        )

    return cleaned


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Return overlap in ms between two intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


# ---------- pyannote label -> real name mapping ----------

def _map_pyannote_labels_to_names(
    diarization: List[dict],
    speaker_log: List[SpeakerLogEntry],
) -> Dict[str, str]:
    """
    Map pyannote labels (SPEAKER_00, SPEAKER_01, ...) to real names from speaker_log
    by computing timing overlap between pyannote segments and speaker_log turns.

    Returns a dict: pyannote_label -> real_name.
    """
    if not speaker_log or not diarization:
        return {}

    rec_start = datetime.fromisoformat(
        min(e.startedAt for e in speaker_log).replace("Z", "+00:00")
    )

    # Build speaker_log turns in seconds (relative to recording start)
    log_turns: List[dict] = []
    for entry in speaker_log:
        entry_start = datetime.fromisoformat(entry.startedAt.replace("Z", "+00:00"))
        entry_end = datetime.fromisoformat(entry.endedAt.replace("Z", "+00:00"))
        start_sec = max(0.0, (entry_start - rec_start).total_seconds())
        end_sec = max(0.0, (entry_end - rec_start).total_seconds())
        log_turns.append({
            "name": entry.name,
            "start": start_sec,
            "end": end_sec,
        })

    # Get unique pyannote labels
    pyannote_labels = sorted(set(seg["speaker"] for seg in diarization))
    real_names = sorted(set(entry.name for entry in speaker_log))

    # Build overlap matrix: pyannote_label x real_name -> total overlap seconds
    overlap_matrix: Dict[str, Dict[str, float]] = {
        label: {name: 0.0 for name in real_names} for label in pyannote_labels
    }

    for seg in diarization:
        label = seg["speaker"]
        seg_start = seg["start"]
        seg_end = seg["end"]
        for turn in log_turns:
            ov = max(0.0, min(seg_end, turn["end"]) - max(seg_start, turn["start"]))
            if ov > 0:
                overlap_matrix[label][turn["name"]] += ov

    # Greedy assignment: for each pyannote label, pick the real name with most overlap
    # (avoiding duplicate assignments where possible)
    label_to_name: Dict[str, str] = {}
    assigned_names: set = set()

    # Sort labels by their total speech time (most speech first) for better matching
    label_total = {
        label: sum(overlap_matrix[label].values()) for label in pyannote_labels
    }
    sorted_labels = sorted(pyannote_labels, key=lambda l: label_total[l], reverse=True)

    for label in sorted_labels:
        best_name = None
        best_overlap = 0.0
        for name in real_names:
            ov = overlap_matrix[label][name]
            if ov > best_overlap:
                # Prefer unassigned names, but allow duplicates as fallback
                if name not in assigned_names or ov > best_overlap * 1.5:
                    best_name = name
                    best_overlap = ov
        if best_name:
            label_to_name[label] = best_name
            assigned_names.add(best_name)
        else:
            label_to_name[label] = label  # keep pyannote label as fallback

    logger.info("Label mapping: %s", label_to_name)
    return label_to_name


# ---------- crosstalk detection from diarization ----------

def _detect_crosstalk_from_diarization(
    diarization: List[dict],
) -> List[CrosstalkFlag]:
    """Detect crosstalk from overlapping diarization segments (>=200ms)."""
    candidates: List[dict] = []
    for i, s1 in enumerate(diarization):
        for s2 in diarization[i + 1:]:
            if s1["speaker"] == s2["speaker"]:
                continue
            ov_start = max(s1["start"], s2["start"])
            ov_end = min(s1["end"], s2["end"])
            ov_ms = int((ov_end - ov_start) * 1000)
            if ov_ms >= 200:
                candidates.append({
                    "start_ms": int(ov_start * 1000),
                    "end_ms": int(ov_end * 1000),
                    "speakers": sorted({s1["speaker"], s2["speaker"]}),
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

    logger.info("Crosstalk detection: %d flags from diarization overlaps", len(flags))
    return flags


# ---------- speaker_log fallback ----------

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


def _detect_crosstalk_from_speaker_log(speaker_turns: List[dict]) -> List[CrosstalkFlag]:
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


class TranscriptionService:
    """
    Transcribe audio/video files using:
    pyannote API (diarization) + whisper (OpenAI API) -> annotated WebVTT

    Speaker_log from the platform is used to map pyannote labels to real names.
    """

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)

        self._pyannote_key = os.environ.get("PYANNOTE_API_KEY")
        if not self._pyannote_key:
            logger.warning("PYANNOTE_API_KEY is not set — diarization will fall back to speaker_log only.")

    # ---------- pyannote API ----------

    def _pyannote_upload(self, file_bytes: bytes, file_name: str) -> str:
        """Upload audio to pyannote and return the media:// URL."""
        media_url = f"media://valid/{file_name}"

        with httpx.Client(timeout=60) as client:
            # Step 1: Get presigned upload URL
            resp = client.post(
                f"{_PYANNOTE_API_BASE}/media/input",
                headers={
                    "Authorization": f"Bearer {self._pyannote_key}",
                    "Content-Type": "application/json",
                },
                json={"url": media_url},
            )
            resp.raise_for_status()
            presigned_url = resp.json()["url"]

            # Step 2: PUT the file
            resp = client.put(
                presigned_url,
                content=file_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
            resp.raise_for_status()

        logger.info("Uploaded %s to pyannote (%d bytes)", media_url, len(file_bytes))
        return media_url

    def _pyannote_diarize(self, media_url: str, num_speakers: Optional[int] = None) -> str:
        """Submit a diarization job and return the jobId."""
        body: Dict[str, Any] = {"url": media_url}
        if num_speakers:
            body["numSpeakers"] = num_speakers

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{_PYANNOTE_API_BASE}/diarize",
                headers={
                    "Authorization": f"Bearer {self._pyannote_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        job_id = data["jobId"]
        logger.info("Diarization job submitted: %s", job_id)
        return job_id

    def _pyannote_poll(self, job_id: str) -> List[dict]:
        """Poll a pyannote job until completion and return diarization segments."""
        elapsed = 0
        with httpx.Client(timeout=30) as client:
            while elapsed < _PYANNOTE_MAX_WAIT:
                resp = client.get(
                    f"{_PYANNOTE_API_BASE}/jobs/{job_id}",
                    headers={"Authorization": f"Bearer {self._pyannote_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                status = data["status"]

                if status == "succeeded":
                    diarization = data.get("output", {}).get("diarization", [])
                    logger.info(
                        "Diarization complete: %d segments from pyannote",
                        len(diarization),
                    )
                    return diarization

                if status in ("failed", "canceled"):
                    raise RuntimeError(f"Pyannote diarization job {status}: {data}")

                logger.debug("Diarization job %s status: %s (waited %ds)", job_id, status, elapsed)
                time.sleep(_PYANNOTE_POLL_INTERVAL)
                elapsed += _PYANNOTE_POLL_INTERVAL

        raise RuntimeError(f"Pyannote diarization timed out after {_PYANNOTE_MAX_WAIT}s")

    def _run_diarization(
        self, file_bytes: bytes, file_name: str, num_speakers: Optional[int] = None
    ) -> List[dict]:
        """Upload audio and run pyannote diarization. Returns list of {speaker, start, end}."""
        media_url = self._pyannote_upload(file_bytes, file_name)
        job_id = self._pyannote_diarize(media_url, num_speakers=num_speakers)
        return self._pyannote_poll(job_id)

    # ---------- whisper ----------

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

    # ---------- merge diarization + VTT ----------

    def _assign_speakers_from_diarization(
        self,
        raw_segments: List[dict],
        diarization: List[dict],
        label_to_name: Dict[str, str],
    ) -> Dict[int, str]:
        """
        Assign speakers to VTT segments based on pyannote diarization timing overlap.

        For each VTT segment, finds the diarization turn with the most overlap
        and assigns that speaker (mapped to real name via label_to_name).

        Returns dict: segment index (1-based) -> speaker name.
        """
        speaker_map: Dict[int, str] = {}

        for idx, seg in enumerate(raw_segments, start=1):
            seg_start = _ts_to_ms(seg["start"]) / 1000.0
            seg_end = _ts_to_ms(seg["end"]) / 1000.0

            best_label = None
            best_overlap = 0.0

            for d_seg in diarization:
                ov = max(0.0, min(seg_end, d_seg["end"]) - max(seg_start, d_seg["start"]))
                if ov > best_overlap:
                    best_overlap = ov
                    best_label = d_seg["speaker"]

            if best_label:
                speaker_map[idx] = label_to_name.get(best_label, best_label)

        logger.info("Speaker assignment: %d/%d segments matched", len(speaker_map), len(raw_segments))
        return speaker_map

    # ---------- build output ----------

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

    # ---------- main pipeline ----------

    def transcribe(
        self,
        *,
        file_bytes: bytes,
        file_name: str,
        language: Optional[str] = None,
        speaker_log: Optional[List[SpeakerLogEntry]] = None,
    ) -> TranscriptionResult:
        """
        Transcription pipeline:
        1. pyannote API (diarization — who spoke when)
        2. Whisper API (speech-to-text -> VTT)
        3. Merge diarization with VTT segments
        4. Map pyannote labels to real names via speaker_log
        5. Detect crosstalk from diarization overlaps
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

            # Step 2: Diarization + speaker assignment
            speaker_map: Dict[int, str] = {}
            roles: Dict[str, str] = {}
            peer_ids: Dict[str, str] = {}
            crosstalk_flags: List[CrosstalkFlag] = []

            # Build role/peerId lookups from speaker_log
            for entry in speaker_log:
                roles[entry.name] = entry.role
                peer_ids[entry.name] = entry.peerId

            if self._pyannote_key:
                # Use pyannote API for diarization
                num_speakers = len(set(e.name for e in speaker_log)) if speaker_log else None
                diarization = self._run_diarization(
                    file_bytes, file_name, num_speakers=num_speakers
                )

                # Map pyannote labels (SPEAKER_00) to real names from speaker_log
                label_to_name = _map_pyannote_labels_to_names(diarization, speaker_log)

                # Assign speakers to VTT segments via timing overlap
                speaker_map = self._assign_speakers_from_diarization(
                    raw_segments, diarization, label_to_name
                )

                # Detect crosstalk from diarization overlaps (use real names)
                renamed_diarization = [
                    {**seg, "speaker": label_to_name.get(seg["speaker"], seg["speaker"])}
                    for seg in diarization
                ]
                crosstalk_flags = _detect_crosstalk_from_diarization(renamed_diarization)
            elif speaker_log:
                # Fallback: use speaker_log timing for crosstalk detection only
                speaker_turns, _, _ = _speaker_log_to_turns(speaker_log)
                crosstalk_flags = _detect_crosstalk_from_speaker_log(speaker_turns)
                logger.warning("No PYANNOTE_API_KEY — skipping diarization, no speaker assignment")

            # Step 3: Build annotated output
            segments, annotated_vtt = self._build_annotated_output(
                raw_segments, speaker_map, crosstalk_flags,
            )

            full_text = " ".join(seg.text for seg in segments)

            # Build embeddings structure (empty vectors for now)
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

"""
src/services/transcription_service.py
--------------------------------------
Audio/Video-to-WebVTT transcription pipeline:

    pyannote (speaker diarization + embeddings + crosstalk detection)
    -> whisper (speech-to-text)
    -> WebVTT with speaker annotations

Import
------
    from src.services.transcription_service import TranscriptionService
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import subprocess

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

# Regex to parse WebVTT cues:  "00:00:01.000 --> 00:00:04.500\nSome text"
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


# ---------- diarization helpers ----------

def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Return overlap in ms between two intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _assign_speaker(
    seg_start_ms: int,
    seg_end_ms: int,
    diarization_turns: List[dict],
) -> Optional[str]:
    """Assign the speaker with the most overlap to a whisper segment."""
    best_speaker = None
    best_overlap = 0
    for turn in diarization_turns:
        ov = _overlap(seg_start_ms, seg_end_ms, turn["start_ms"], turn["end_ms"])
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = turn["speaker"]
    return best_speaker


def _detect_crosstalk_from_diarization(
    diarization_turns: List[dict],
) -> List[CrosstalkFlag]:
    """
    Detect crosstalk by finding overlapping diarization turns from different speakers.

    No ML model needed — if pyannote says two different speakers are talking
    at the same time, that's crosstalk.
    """
    candidates: List[dict] = []
    for i, t1 in enumerate(diarization_turns):
        for t2 in diarization_turns[i + 1:]:
            if t1["speaker"] == t2["speaker"]:
                continue
            ov_start = max(t1["start_ms"], t2["start_ms"])
            ov_end = min(t1["end_ms"], t2["end_ms"])
            if ov_end > ov_start and (ov_end - ov_start) >= 200:  # min 200ms overlap
                candidates.append({
                    "start_ms": ov_start,
                    "end_ms": ov_end,
                    "speakers": sorted({t1["speaker"], t2["speaker"]}),
                })

    if not candidates:
        return []

    # Merge overlapping candidates
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
        # Confidence based on overlap duration: longer overlap = higher confidence
        confidence = min(1.0, overlap_ms / 2000.0)  # 2s+ overlap = 1.0
        flags.append(CrosstalkFlag(
            start=_ms_to_ts(region["start_ms"]),
            end=_ms_to_ts(region["end_ms"]),
            speakers=region["speakers"],
            confidence=round(confidence, 4),
        ))

    logger.info("Crosstalk detection complete: %d flags from diarization overlaps", len(flags))
    return flags


def _match_pyannote_to_speaker_log(
    diarization_turns: List[dict],
    speaker_log: List[SpeakerLogEntry],
    recording_start_iso: str,
) -> Dict[str, dict]:
    """
    Match pyannote labels (SPEAKER_00, SPEAKER_01, ...) to platform speaker_log
    entries by correlating timing overlaps.

    Returns
    -------
    dict mapping pyannote label -> {"name": str, "role": str, "peerId": str}
    """
    from datetime import datetime

    rec_start = datetime.fromisoformat(recording_start_iso.replace("Z", "+00:00"))

    log_turns: List[dict] = []
    for entry in speaker_log:
        entry_start = datetime.fromisoformat(entry.startedAt.replace("Z", "+00:00"))
        entry_end = datetime.fromisoformat(entry.endedAt.replace("Z", "+00:00"))
        start_ms = int((entry_start - rec_start).total_seconds() * 1000)
        end_ms = int((entry_end - rec_start).total_seconds() * 1000)
        log_turns.append({
            "name": entry.name,
            "role": entry.role,
            "peerId": entry.peerId,
            "start_ms": max(0, start_ms),
            "end_ms": max(0, end_ms),
        })

    pyannote_labels = sorted(set(t["speaker"] for t in diarization_turns))

    pyannote_by_label: Dict[str, List[dict]] = {}
    for t in diarization_turns:
        pyannote_by_label.setdefault(t["speaker"], []).append(t)

    unique_speakers = sorted(set(lt["name"] for lt in log_turns))
    log_by_name: Dict[str, List[dict]] = {}
    for lt in log_turns:
        log_by_name.setdefault(lt["name"], []).append(lt)

    # Build overlap matrix
    overlap_matrix: Dict[str, Dict[str, int]] = {}
    for label in pyannote_labels:
        overlap_matrix[label] = {}
        for spk_name in unique_speakers:
            total_ov = 0
            for pt in pyannote_by_label[label]:
                for lt in log_by_name[spk_name]:
                    total_ov += _overlap(pt["start_ms"], pt["end_ms"], lt["start_ms"], lt["end_ms"])
            overlap_matrix[label][spk_name] = total_ov

    # Greedy assignment: longest-duration pyannote label first
    label_durations = {
        label: sum(t["end_ms"] - t["start_ms"] for t in turns)
        for label, turns in pyannote_by_label.items()
    }
    sorted_labels = sorted(pyannote_labels, key=lambda l: label_durations.get(l, 0), reverse=True)

    result: Dict[str, dict] = {}
    assigned_speakers: set = set()

    for label in sorted_labels:
        best_name = None
        best_ov = 0
        for spk_name in unique_speakers:
            if spk_name in assigned_speakers:
                continue
            ov = overlap_matrix[label].get(spk_name, 0)
            if ov > best_ov:
                best_ov = ov
                best_name = spk_name

        if best_name:
            assigned_speakers.add(best_name)
            first_entry = log_by_name[best_name][0]
            result[label] = {
                "name": best_name,
                "role": first_entry["role"],
                "peerId": first_entry["peerId"],
            }
        else:
            result[label] = {
                "name": label,
                "role": "remote",
                "peerId": None,
            }

    logger.info("Speaker mapping: %s", {k: v["name"] for k, v in result.items()})
    return result


# ---------- singleton for heavy ML models ----------

_singleton_instance: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Return a singleton TranscriptionService (models loaded once)."""
    global _singleton_instance
    if _singleton_instance is None:
        _singleton_instance = TranscriptionService()
    return _singleton_instance


class TranscriptionService:
    """
    Transcribe audio/video files using the pipeline:
    pyannote (diarization + embeddings + crosstalk) -> whisper -> WebVTT

    Models are loaded lazily on first transcription call, not at import time.
    """

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)

        # ML models are loaded lazily on first use
        self._device = None
        self._diarization_pipeline = None
        self._embedding_model = None
        self._models_loaded = False

    def _load_models(self):
        """Lazily load pyannote models on first call."""
        if self._models_loaded:
            return

        import torch
        from pyannote.audio import Pipeline as PyannotePipeline
        from pyannote.audio import Inference as PyannoteInference
        from pyannote.audio import Model as PyannoteModel

        hf_token = os.environ.get("HF_AUTH_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_AUTH_TOKEN is not set (required for pyannote).")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading pyannote models on device: %s", self._device)

        # pyannote speaker diarization pipeline
        self._diarization_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        ).to(self._device)

        # pyannote speaker embedding model — load model first, then wrap in Inference
        embedding_model = PyannoteModel.from_pretrained(
            "pyannote/embedding",
            token=hf_token,
        ).to(self._device)
        self._embedding_model = PyannoteInference(
            embedding_model,
            device=self._device,
        )

        self._models_loaded = True
        logger.info("Pyannote models loaded successfully.")

    # ------------------------------------------------------------------ #
    #  Step 1: pyannote diarization + embeddings
    # ------------------------------------------------------------------ #

    def _run_diarization(self, audio_path: str) -> Tuple[List[dict], dict]:
        """Run pyannote diarization on the audio file."""
        logger.info("Running pyannote speaker diarization...")
        diarization = self._diarization_pipeline(audio_path)

        turns: List[dict] = []
        speaker_set: set = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            turns.append({
                "speaker": speaker,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_ts": _ms_to_ts(start_ms),
                "end_ts": _ms_to_ts(end_ms),
            })
            speaker_set.add(speaker)

        speaker_map = {s: s for s in sorted(speaker_set)}
        logger.info(
            "Diarization complete: %d turns, %d speakers", len(turns), len(speaker_set)
        )
        return turns, speaker_map

    def _extract_embeddings(
        self,
        audio_path: str,
        segments: List[TranscriptSegment],
        speaker_roles: Dict[str, str],
    ) -> List[SpeakerEmbedding]:
        """Extract pyannote speaker embeddings for each transcript segment."""
        logger.info("Extracting speaker embeddings for %d segments...", len(segments))
        from pyannote.core import Segment

        embeddings: List[SpeakerEmbedding] = []
        for idx, seg in enumerate(segments):
            start_sec = _ts_to_ms(seg.start) / 1000.0
            end_sec = _ts_to_ms(seg.end) / 1000.0
            duration = end_sec - start_sec

            if duration < 0.1:
                emb_vector = []
            else:
                try:
                    excerpt = Segment(start_sec, end_sec)
                    emb = self._embedding_model.crop(audio_path, excerpt)
                    emb_vector = emb.flatten().tolist()
                except Exception as e:
                    logger.warning("Embedding extraction failed for segment %d: %s", idx, e)
                    emb_vector = []

            speaker = seg.speaker or "UNKNOWN"
            role = speaker_roles.get(speaker, "remote")

            embeddings.append(SpeakerEmbedding(
                chunk_index=idx,
                speaker=speaker,
                role=role,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                embedding=emb_vector,
            ))

        return embeddings

    # ------------------------------------------------------------------ #
    #  Step 2: whisper transcription
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Step 3: merge diarization with whisper segments -> annotated WebVTT
    # ------------------------------------------------------------------ #

    def _merge_and_build(
        self,
        raw_vtt_segments: List[dict],
        diarization_turns: List[dict],
        crosstalk_flags: List[CrosstalkFlag],
    ) -> Tuple[List[TranscriptSegment], str]:
        """Merge whisper segments with diarization labels and crosstalk flags."""
        ct_regions = [
            (_ts_to_ms(f.start), _ts_to_ms(f.end))
            for f in crosstalk_flags
        ]

        segments: List[TranscriptSegment] = []
        vtt_lines = ["WEBVTT", ""]

        for idx, raw in enumerate(raw_vtt_segments, start=1):
            seg_start_ms = _ts_to_ms(raw["start"])
            seg_end_ms = _ts_to_ms(raw["end"])

            speaker = _assign_speaker(seg_start_ms, seg_end_ms, diarization_turns)

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

    # ------------------------------------------------------------------ #
    #  Speaker stats computation
    # ------------------------------------------------------------------ #

    def _compute_speaker_stats(
        self,
        segments: List[TranscriptSegment],
        speaker_roles: Dict[str, str],
        speaker_peer_ids: Dict[str, str],
    ) -> Dict[str, SpeakerStats]:
        """Compute aggregated stats per speaker from merged segments."""
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

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def transcribe(
        self,
        *,
        file_bytes: bytes,
        file_name: str,
        language: Optional[str] = None,
        speaker_log: Optional[List[SpeakerLogEntry]] = None,
    ) -> TranscriptionResult:
        """
        Full transcription pipeline: pyannote -> whisper -> WebVTT.

        Crosstalk is detected from pyannote diarization overlaps (no asteroid).
        """
        self._load_models()

        suffix = Path(file_name).suffix or ".m4a"
        speaker_log = speaker_log or []

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Extract audio to WAV for pyannote (it can't read MP4/M4A directly)
        wav_path = tmp_path + ".wav"

        try:
            logger.info(
                "Starting transcription pipeline for %s (%d bytes)",
                file_name, len(file_bytes),
            )

            # Convert to WAV using ffmpeg
            logger.info("Extracting audio to WAV via ffmpeg...")
            subprocess.run(
                ["ffmpeg", "-i", tmp_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", "-y", wav_path],
                capture_output=True, check=True,
            )

            # Step 1: pyannote diarization (on WAV file)
            diarization_turns, raw_speaker_map = self._run_diarization(wav_path)

            # Match pyannote labels to platform speakers via timing correlation
            effective_roles: Dict[str, str] = {}
            effective_peer_ids: Dict[str, str] = {}

            if speaker_log:
                recording_start = min(e.startedAt for e in speaker_log)
                speaker_mapping = _match_pyannote_to_speaker_log(
                    diarization_turns, speaker_log, recording_start,
                )

                for turn in diarization_turns:
                    raw_label = turn["speaker"]
                    if raw_label in speaker_mapping:
                        mapped = speaker_mapping[raw_label]
                        turn["speaker"] = mapped["name"]
                        effective_roles[mapped["name"]] = mapped["role"]
                        if mapped["peerId"]:
                            effective_peer_ids[mapped["name"]] = mapped["peerId"]

            # Detect crosstalk from diarization overlaps
            crosstalk_flags = _detect_crosstalk_from_diarization(diarization_turns)

            # Step 2: whisper transcription
            raw_vtt = self._run_whisper(tmp_path, language=language)
            raw_segments = _parse_vtt(raw_vtt)

            # Step 3: merge diarization + crosstalk flags with whisper segments
            segments, annotated_vtt = self._merge_and_build(
                raw_segments, diarization_turns, crosstalk_flags,
            )

            full_text = " ".join(seg.text for seg in segments)

            # Extract pyannote embeddings per segment (using WAV)
            embeddings = self._extract_embeddings(
                wav_path, segments, effective_roles,
            )

            # Compute speaker stats
            stats = self._compute_speaker_stats(
                segments, effective_roles, effective_peer_ids,
            )

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
            Path(wav_path).unlink(missing_ok=True)

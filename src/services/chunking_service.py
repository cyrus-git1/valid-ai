"""
src/services/chunking_service.py
---------------------------------
Document chunking: PDF, DOCX, XLSX, WebVTT, and web-scraped JSON.

Converts raw bytes into token-aware chunks ready for embedding.
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional

import fitz
import pandas as pd
import spacy
import tiktoken
from docx import Document

from src.config.llm import LLMConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = LLMConfig.DEFAULT
MAX_TOKENS = 800
OVERLAP_TOKENS = 120
DOCX_PARAS_PER_PAGE = 8

nlp = spacy.load("en_core_web_sm")
enc = tiktoken.encoding_for_model(MODEL_NAME)


class ChunkingService:
    """Stateless service — all methods are static / class-level."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def llm_token_len(text: str) -> int:
        return len(enc.encode(text))

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        text = re.sub(r"[ \t]+", " ", text)
        return text

    # ── Page extraction ───────────────────────────────────────────────────────

    @staticmethod
    def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text") or ""
            text = ChunkingService._normalize_text(text)
            if text:
                pages.append({"page": i + 1, "text": text})
        return pages

    @staticmethod
    def extract_pages_from_docx_bytes(
        docx_bytes: bytes,
        paras_per_page: int = DOCX_PARAS_PER_PAGE,
    ) -> List[Dict[str, Any]]:
        doc = Document(io.BytesIO(docx_bytes))
        paras: List[str] = []
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if t:
                paras.append(t)

        pages: List[Dict[str, Any]] = []
        if not paras:
            return pages

        page_no = 1
        buf: List[str] = []
        for i, para in enumerate(paras, start=1):
            buf.append(para)
            if i % paras_per_page == 0:
                text = ChunkingService._normalize_text("\n\n".join(buf))
                if text:
                    pages.append({"page": page_no, "text": text})
                    page_no += 1
                buf = []

        if buf:
            text = ChunkingService._normalize_text("\n\n".join(buf))
            if text:
                pages.append({"page": page_no, "text": text})

        return pages

    @staticmethod
    def extract_pages_from_xlsx_bytes(
        xlsx_bytes: bytes,
        rows_per_page: int = 50,
    ) -> List[Dict[str, Any]]:
        xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
        pages: List[Dict[str, Any]] = []
        page_no = 1

        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name, dtype=str).fillna("")
            if df.empty:
                continue

            columns = [str(c) for c in df.columns]
            buf: List[str] = [f"[Sheet: {sheet_name}]"]

            for _row_idx, row in df.iterrows():
                parts = [
                    f"{col}: {val}" for col, val in zip(columns, row)
                    if str(val).strip()
                ]
                if parts:
                    buf.append(" | ".join(parts))

                if len(buf) >= rows_per_page:
                    text = ChunkingService._normalize_text("\n".join(buf))
                    if text:
                        pages.append({"page": page_no, "text": text})
                        page_no += 1
                    buf = [f"[Sheet: {sheet_name} (cont.)]"]

            if buf:
                text = ChunkingService._normalize_text("\n".join(buf))
                if text:
                    pages.append({"page": page_no, "text": text})
                    page_no += 1

        return pages

    # ── Chunking ──────────────────────────────────────────────────────────────

    @staticmethod
    def chunk_pages_spacy_token_aware(
        pages: List[Dict[str, Any]],
        max_tokens: int = MAX_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        buffer_sents: List[str] = []
        buffer_tokens = 0
        chunk_start_page: Optional[int] = None
        last_page: Optional[int] = None

        for page in pages:
            page_no = page["page"]
            last_page = page_no
            doc = nlp(page["text"])

            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue

                sent_tokens = ChunkingService.llm_token_len(sent_text)

                if sent_tokens > max_tokens:
                    words = sent_text.split()
                    tmp: List[str] = []
                    for w in words:
                        tmp.append(w)
                        if ChunkingService.llm_token_len(" ".join(tmp)) >= max_tokens:
                            chunk_text = " ".join(tmp)
                            chunks.append({
                                "text": chunk_text,
                                "start_page": page_no,
                                "end_page": page_no,
                                "token_count": ChunkingService.llm_token_len(chunk_text),
                            })
                            tmp = []
                    if tmp:
                        chunk_text = " ".join(tmp)
                        chunks.append({
                            "text": chunk_text,
                            "start_page": page_no,
                            "end_page": page_no,
                            "token_count": ChunkingService.llm_token_len(chunk_text),
                        })
                    continue

                if buffer_sents and buffer_tokens + sent_tokens > max_tokens:
                    chunk_text = " ".join(buffer_sents)
                    chunks.append({
                        "text": chunk_text,
                        "start_page": chunk_start_page or page_no,
                        "end_page": page_no,
                        "token_count": buffer_tokens,
                    })

                    if overlap_tokens > 0:
                        overlap_sents: List[str] = []
                        overlap_count = 0
                        for s in reversed(buffer_sents):
                            t = ChunkingService.llm_token_len(s)
                            if overlap_sents and overlap_count + t > overlap_tokens:
                                break
                            overlap_sents.insert(0, s)
                            overlap_count += t
                        buffer_sents = overlap_sents
                        buffer_tokens = overlap_count
                        chunk_start_page = page_no
                    else:
                        buffer_sents = []
                        buffer_tokens = 0
                        chunk_start_page = None

                if not buffer_sents:
                    chunk_start_page = page_no

                buffer_sents.append(sent_text)
                buffer_tokens += sent_tokens

        if buffer_sents:
            chunk_text = " ".join(buffer_sents)
            chunks.append({
                "text": chunk_text,
                "start_page": chunk_start_page or (last_page or 1),
                "end_page": last_page or (chunk_start_page or 1),
                "token_count": buffer_tokens,
            })

        return chunks

    # ── Format-specific convenience methods ───────────────────────────────────

    @staticmethod
    def pdf_bytes_to_chunks(pdf_bytes: bytes) -> List[Dict[str, Any]]:
        pages = ChunkingService.extract_pages_from_pdf_bytes(pdf_bytes)
        return ChunkingService.chunk_pages_spacy_token_aware(pages)

    @staticmethod
    def docx_bytes_to_chunks(docx_bytes: bytes) -> List[Dict[str, Any]]:
        pages = ChunkingService.extract_pages_from_docx_bytes(docx_bytes)
        return ChunkingService.chunk_pages_spacy_token_aware(pages)

    @staticmethod
    def xlsx_bytes_to_chunks(xlsx_bytes: bytes) -> List[Dict[str, Any]]:
        pages = ChunkingService.extract_pages_from_xlsx_bytes(xlsx_bytes)
        return ChunkingService.chunk_pages_spacy_token_aware(pages)

    @staticmethod
    def vtt_bytes_to_chunks(vtt_bytes: bytes) -> List[Dict[str, Any]]:
        vtt_text = vtt_bytes.decode("utf-8", errors="replace")
        cues = ChunkingService.parse_vtt(vtt_text)
        pages = ChunkingService.vtt_cues_to_pages(cues)
        return ChunkingService.chunk_pages_spacy_token_aware(pages)

    @staticmethod
    def document_bytes_to_chunks(file_bytes: bytes, file_type: str) -> List[Dict[str, Any]]:
        ft = file_type.lower().strip(".")
        if ft == "pdf":
            return ChunkingService.pdf_bytes_to_chunks(file_bytes)
        if ft == "docx":
            return ChunkingService.docx_bytes_to_chunks(file_bytes)
        if ft == "vtt":
            return ChunkingService.vtt_bytes_to_chunks(file_bytes)
        if ft in ("xlsx", "xls"):
            return ChunkingService.xlsx_bytes_to_chunks(file_bytes)
        raise ValueError(f"Unsupported file_type: {file_type}")

    # ── WebVTT parsing ────────────────────────────────────────────────────────

    _VTT_TS_RE = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})")

    @staticmethod
    def _parse_vtt_timestamp(ts: str) -> float:
        m = ChunkingService._VTT_TS_RE.match(ts.strip())
        if not m:
            raise ValueError(f"Invalid VTT timestamp: {ts!r}")
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        millis = int(m.group(4))
        return hours * 3600 + minutes * 60 + seconds + millis / 1000

    @staticmethod
    def parse_vtt(vtt_text: str) -> List[Dict[str, Any]]:
        lines = vtt_text.strip().splitlines()
        cues: List[Dict[str, Any]] = []
        i = 0

        while i < len(lines) and not re.match(r"\d{2}:\d{2}", lines[i]):
            i += 1

        cue_index = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            if line.isdigit():
                i += 1
                continue

            if "-->" in line:
                parts = line.split("-->")
                start = ChunkingService._parse_vtt_timestamp(parts[0])
                end_raw = parts[1].strip().split()[0]
                end = ChunkingService._parse_vtt_timestamp(end_raw)

                i += 1
                text_lines: List[str] = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1

                raw_text = " ".join(text_lines)

                speaker: Optional[str] = None
                voice_match = re.match(r"<v\s+([^>]+)>(.+)", raw_text)
                if voice_match:
                    speaker = voice_match.group(1).strip()
                    raw_text = voice_match.group(2).strip()

                clean_text = re.sub(r"</?[^>]+>", "", raw_text).strip()

                if clean_text:
                    cues.append({
                        "index": cue_index,
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "text": clean_text,
                    })
                    cue_index += 1
            else:
                i += 1

        return cues

    @staticmethod
    def vtt_cues_to_pages(
        cues: List[Dict[str, Any]],
        window_seconds: float = 120.0,
    ) -> List[Dict[str, Any]]:
        if not cues:
            return []

        pages: List[Dict[str, Any]] = []
        page_no = 1
        buf: List[str] = []
        window_start = cues[0]["start"]

        for cue in cues:
            if cue["start"] - window_start >= window_seconds and buf:
                pages.append({"page": page_no, "text": ChunkingService._normalize_text("\n".join(buf))})
                page_no += 1
                buf = []
                window_start = cue["start"]

            prefix = f"[{cue['speaker']}]: " if cue.get("speaker") else ""
            buf.append(f"{prefix}{cue['text']}")

        if buf:
            pages.append({"page": page_no, "text": ChunkingService._normalize_text("\n".join(buf))})

        return pages

    # ── Web scraped content ───────────────────────────────────────────────────

    @staticmethod
    def web_scraped_json_to_pages(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        pages = []
        for page_data in json_data.get("pages", []):
            page_num = page_data.get("page", 0)
            text = page_data.get("text", "")
            if text:
                normalized_text = ChunkingService._normalize_text(text)
                if normalized_text:
                    pages.append({"page": page_num, "text": normalized_text})
        return pages

    @staticmethod
    def web_scraped_json_to_chunks(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        pages = ChunkingService.web_scraped_json_to_pages(json_data)
        return ChunkingService.chunk_pages_spacy_token_aware(pages)


# Module-level convenience functions for backward compat
def document_bytes_to_chunks(file_bytes: bytes, file_type: str) -> List[Dict[str, Any]]:
    return ChunkingService.document_bytes_to_chunks(file_bytes, file_type)


def web_scraped_json_to_chunks(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return ChunkingService.web_scraped_json_to_chunks(json_data)


def parse_vtt(vtt_text: str) -> List[Dict[str, Any]]:
    return ChunkingService.parse_vtt(vtt_text)


def vtt_cues_to_pages(cues: List[Dict[str, Any]], window_seconds: float = 120.0) -> List[Dict[str, Any]]:
    return ChunkingService.vtt_cues_to_pages(cues, window_seconds)

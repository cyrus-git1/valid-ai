import re
from typing import List, Dict, Any, Optional
import io

import fitz
import pandas as pd
import spacy
import tiktoken
from docx import Document

# ---- config ----
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 800
OVERLAP_TOKENS = 120
DOCX_PARAS_PER_PAGE = 8  # pseudo-page size for DOCX (since DOCX has no real pages)
# ---------------

nlp = spacy.load("en_core_web_sm")
enc = tiktoken.encoding_for_model(MODEL_NAME)


def llm_token_len(text: str) -> int:
    return len(enc.encode(text))


def _normalize_text(text: str) -> str:
    # normalize (shared)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # de-hyphenate across line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)     # single newline -> space
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip() # normalize blank lines
    text = re.sub(r"[ \t]+", " ", text)              # collapse runs of spaces/tabs
    return text


def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        text = _normalize_text(text)
        if text:
            pages.append({"page": i + 1, "text": text})

    return pages


def extract_pages_from_docx_bytes(
    docx_bytes: bytes,
    paras_per_page: int = DOCX_PARAS_PER_PAGE,
) -> List[Dict[str, Any]]:
    """
    DOCX has no stable 'page numbers' unless you render it.
    This groups paragraphs into 'pseudo-pages' so the rest of the pipeline stays identical.
    """
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
            text = _normalize_text("\n\n".join(buf))
            if text:
                pages.append({"page": page_no, "text": text})
                page_no += 1
            buf = []

    if buf:
        text = _normalize_text("\n\n".join(buf))
        if text:
            pages.append({"page": page_no, "text": text})

    return pages


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

            sent_tokens = llm_token_len(sent_text)

            # If a single sentence is too large, fall back to splitting by words
            if sent_tokens > max_tokens:
                words = sent_text.split()
                tmp: List[str] = []
                for w in words:
                    tmp.append(w)
                    if llm_token_len(" ".join(tmp)) >= max_tokens:
                        chunk_text = " ".join(tmp)
                        chunks.append({
                            "text": chunk_text,
                            "start_page": page_no,
                            "end_page": page_no,
                            "token_count": llm_token_len(chunk_text),
                        })
                        tmp = []
                if tmp:
                    chunk_text = " ".join(tmp)
                    chunks.append({
                        "text": chunk_text,
                        "start_page": page_no,
                        "end_page": page_no,
                        "token_count": llm_token_len(chunk_text),
                    })
                continue

            # Flush if this sentence would exceed chunk limit
            if buffer_sents and buffer_tokens + sent_tokens > max_tokens:
                chunk_text = " ".join(buffer_sents)
                chunks.append({
                    "text": chunk_text,
                    "start_page": chunk_start_page or page_no,
                    "end_page": page_no,
                    "token_count": buffer_tokens,
                })

                # Build overlap
                if overlap_tokens > 0:
                    overlap_sents: List[str] = []
                    overlap_count = 0
                    for s in reversed(buffer_sents):
                        t = llm_token_len(s)
                        if overlap_sents and overlap_count + t > overlap_tokens:
                            break
                        overlap_sents.insert(0, s)
                        overlap_count += t

                    buffer_sents = overlap_sents
                    buffer_tokens = overlap_count
                    chunk_start_page = page_no  # conservative
                else:
                    buffer_sents = []
                    buffer_tokens = 0
                    chunk_start_page = None

            if not buffer_sents:
                chunk_start_page = page_no

            buffer_sents.append(sent_text)
            buffer_tokens += sent_tokens

    # Flush remainder
    if buffer_sents:
        chunk_text = " ".join(buffer_sents)
        chunks.append({
            "text": chunk_text,
            "start_page": chunk_start_page or (last_page or 1),
            "end_page": last_page or (chunk_start_page or 1),
            "token_count": buffer_tokens,
        })

    return chunks


def pdf_bytes_to_chunks(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    pages = extract_pages_from_pdf_bytes(pdf_bytes)
    return chunk_pages_spacy_token_aware(pages)


def docx_bytes_to_chunks(docx_bytes: bytes) -> List[Dict[str, Any]]:
    pages = extract_pages_from_docx_bytes(docx_bytes)
    return chunk_pages_spacy_token_aware(pages)


def extract_pages_from_xlsx_bytes(
    xlsx_bytes: bytes,
    rows_per_page: int = 50,
) -> List[Dict[str, Any]]:
    """
    Read an Excel workbook with pandas and convert to pseudo-pages.

    Strategy:
      - Each sheet is read independently.
      - Rows are converted to a readable text representation (one line per row
        with column headers as keys).
      - Rows are grouped into pseudo-pages of ``rows_per_page`` each so the
        downstream chunker can apply the same sentence / token-aware logic.
      - Sheets are processed in order; page numbering is continuous across
        all sheets.
    """
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    pages: List[Dict[str, Any]] = []
    page_no = 1

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name, dtype=str).fillna("")

        if df.empty:
            continue

        columns = [str(c) for c in df.columns]
        buf: List[str] = []
        buf.append(f"[Sheet: {sheet_name}]")

        for row_idx, row in df.iterrows():
            parts = [
                f"{col}: {val}" for col, val in zip(columns, row)
                if str(val).strip()
            ]
            if parts:
                buf.append(" | ".join(parts))

            if len(buf) >= rows_per_page:
                text = _normalize_text("\n".join(buf))
                if text:
                    pages.append({"page": page_no, "text": text})
                    page_no += 1
                buf = [f"[Sheet: {sheet_name} (cont.)]"]

        # Flush remainder for this sheet
        if buf:
            text = _normalize_text("\n".join(buf))
            if text:
                pages.append({"page": page_no, "text": text})
                page_no += 1

    return pages


def xlsx_bytes_to_chunks(xlsx_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse Excel bytes and return token-aware chunks."""
    pages = extract_pages_from_xlsx_bytes(xlsx_bytes)
    return chunk_pages_spacy_token_aware(pages)


def document_bytes_to_chunks(file_bytes: bytes, file_type: str) -> List[Dict[str, Any]]:
    """
    file_type: "pdf", "docx", "vtt", or "xlsx"
    """
    ft = file_type.lower().strip(".")
    if ft == "pdf":
        return pdf_bytes_to_chunks(file_bytes)
    if ft == "docx":
        return docx_bytes_to_chunks(file_bytes)
    if ft == "vtt":
        return vtt_bytes_to_chunks(file_bytes)
    if ft in ("xlsx", "xls"):
        return xlsx_bytes_to_chunks(file_bytes)
    raise ValueError(f"Unsupported file_type: {file_type}")


# ─── WebVTT (Daily.js transcript) ────────────────────────────────────────────

_VTT_TS_RE = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)


def _parse_vtt_timestamp(ts: str) -> float:
    """Convert a WebVTT timestamp like '00:01:23.456' to seconds."""
    m = _VTT_TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid VTT timestamp: {ts!r}")
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    millis = int(m.group(4))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def parse_vtt(vtt_text: str) -> List[Dict[str, Any]]:
    """
    Parse a WebVTT string into a list of cue dicts.

    Each cue dict has keys: index, start, end, speaker (optional), text.
    Handles the ``<v Speaker>`` voice tag convention used by Daily.js.
    """
    lines = vtt_text.strip().splitlines()
    cues: List[Dict[str, Any]] = []
    i = 0

    # Skip the WEBVTT header and any metadata lines
    while i < len(lines) and not re.match(r"\d{2}:\d{2}", lines[i]):
        i += 1

    cue_index = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip blank lines
        if not line:
            i += 1
            continue

        # Skip numeric cue identifiers
        if line.isdigit():
            i += 1
            continue

        # Expect a timestamp line: 00:00:01.000 --> 00:00:04.500
        if "-->" in line:
            parts = line.split("-->")
            start = _parse_vtt_timestamp(parts[0])
            # Strip position/alignment metadata after the end timestamp
            end_raw = parts[1].strip().split()[0]
            end = _parse_vtt_timestamp(end_raw)

            # Collect cue text lines until blank line or end of file
            i += 1
            text_lines: List[str] = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            raw_text = " ".join(text_lines)

            # Extract speaker from <v SpeakerName> tag
            speaker: Optional[str] = None
            voice_match = re.match(r"<v\s+([^>]+)>(.+)", raw_text)
            if voice_match:
                speaker = voice_match.group(1).strip()
                raw_text = voice_match.group(2).strip()

            # Strip remaining VTT tags like </v>
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


def vtt_cues_to_pages(
    cues: List[Dict[str, Any]],
    window_seconds: float = 120.0,
) -> List[Dict[str, Any]]:
    """
    Group VTT cues into pseudo-pages by time window.

    Each "page" covers ``window_seconds`` of transcript. Speaker labels are
    prepended so the chunker preserves attribution context.
    """
    if not cues:
        return []

    pages: List[Dict[str, Any]] = []
    page_no = 1
    buf: List[str] = []
    window_start = cues[0]["start"]

    for cue in cues:
        if cue["start"] - window_start >= window_seconds and buf:
            pages.append({"page": page_no, "text": _normalize_text("\n".join(buf))})
            page_no += 1
            buf = []
            window_start = cue["start"]

        prefix = f"[{cue['speaker']}]: " if cue.get("speaker") else ""
        buf.append(f"{prefix}{cue['text']}")

    if buf:
        pages.append({"page": page_no, "text": _normalize_text("\n".join(buf))})

    return pages


def vtt_bytes_to_chunks(vtt_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse WebVTT bytes and return token-aware chunks."""
    vtt_text = vtt_bytes.decode("utf-8", errors="replace")
    cues = parse_vtt(vtt_text)
    pages = vtt_cues_to_pages(cues)
    return chunk_pages_spacy_token_aware(pages)


# ─── Web scraped content ─────────────────────────────────────────────────────

def web_scraped_json_to_pages(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert web scraped JSON data to page format for tokenization.
    
    Expected JSON format:
    {
        "source_url": "...",
        "scraped_at": "...",
        "total_pages": N,
        "pages": [
            {"page": 1, "url": "...", "title": "...", "text": "..."},
            ...
        ]
    }
    
    Returns list of pages in format: [{"page": N, "text": "..."}, ...]
    """
    pages = []
    for page_data in json_data.get("pages", []):
        page_num = page_data.get("page", 0)
        text = page_data.get("text", "")
        if text:
            # Normalize text using the same function as PDF/DOCX
            normalized_text = _normalize_text(text)
            if normalized_text:
                pages.append({"page": page_num, "text": normalized_text})
    return pages


def web_scraped_json_to_chunks(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert web scraped JSON data directly to chunks.
    
    Args:
        json_data: Dictionary containing scraped web data in the format
                  returned by run_scraper.py
    
    Returns:
        List of chunks in format: [
            {
                "text": "...",
                "start_page": N,
                "end_page": M,
                "token_count": K
            },
            ...
        ]
    """
    pages = web_scraped_json_to_pages(json_data)
    return chunk_pages_spacy_token_aware(pages)
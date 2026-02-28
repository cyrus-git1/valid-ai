"""
src/services/ingest_service.py
-------------------------------
Orchestrates the full ingest pipeline for PDF, DOCX, and web sources.
No FastAPI / HTTP coupling — call from a router, background task, or worker.

Supported source types
----------------------
  pdf / docx  — file_bytes + file_name → upload to bucket → chunk → embed → store
  xlsx / xls  — file_bytes + file_name → upload to bucket → pandas parse → chunk → embed → store
  vtt         — file_bytes + file_name → upload to bucket → parse WebVTT → chunk → embed → store
  web         — web_url → scrape subprocess → chunk → embed → store

Import
------
    from src.services.ingest_service import IngestService, IngestInput, IngestOutput
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from supabase import Client

from src.processing.tokenization import document_bytes_to_chunks, web_scraped_json_to_chunks
from src.processing.helpers import embed_texts

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
PDF_BUCKET = "pdf"
_SUPPORTED_FILE_TYPES = {"pdf", "docx", "vtt", "xlsx", "xls"}


# ─────────────────────────────────────────────────────────────────────────────
# DTOs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestInput:
    tenant_id: UUID
    client_id: UUID

    # File ingest — provide both
    file_bytes: Optional[bytes] = None
    file_name: Optional[str] = None

    # Web ingest — provide this
    web_url: Optional[str] = None

    title: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64
    prune_after_ingest: bool = False


@dataclass
class IngestOutput:
    document_id: UUID
    source_type: str
    source_uri: str
    chunks_upserted: int
    chunk_ids: List[UUID]
    warnings: List[str]
    prune_result: Optional[JsonDict] = None


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class IngestService:
    def __init__(self, supabase: Client):
        self.sb = supabase

    # ── Storage ───────────────────────────────────────────────────────────────

    def upload_to_bucket(self, file_bytes: bytes, file_name: str, bucket: str = PDF_BUCKET) -> str:
        path = file_name.lstrip("/")
        self.sb.storage.from_(bucket).upload(path, file_bytes, file_options={"upsert": "true"})
        logger.info("Uploaded %d bytes → bucket '%s' path '%s'", len(file_bytes), bucket, path)
        return path

    def download_from_storage(self, source_uri: str) -> Tuple[bytes, str, str, str]:
        """
        Download from Supabase Storage by source_uri ("bucket:pdf/file.pdf").
        Returns (file_bytes, file_type, bucket, path).
        """
        uri = source_uri.removeprefix("bucket:")
        bucket, path = uri.split("/", 1)
        file_type = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        data = self.sb.storage.from_(bucket).download(path)
        if not isinstance(data, (bytes, bytearray)):
            raise RuntimeError(f"Unexpected storage download type: {type(data)}")
        return bytes(data), file_type, bucket, path

    def _storage_uri(self, bucket: str, path: str) -> str:
        return f"bucket:{bucket}/{path}"

    # ── Documents ─────────────────────────────────────────────────────────────

    def _upsert_document(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        source_type: str,
        source_uri: str,
        title: Optional[str],
        metadata: JsonDict,
    ) -> UUID:
        payload = {
            "tenant_id": str(tenant_id),
            "client_id": str(client_id),
            "source_type": source_type,
            "source_uri": source_uri,
            "title": title,
            "metadata": metadata or {},
        }
        res = (
            self.sb.table("documents")
            .insert(payload)
            .execute()
        )
        if not res.data:
            raise RuntimeError("documents insert returned no rows")
        return UUID(res.data[0]["id"])

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_in_batches(self, texts: List[str], model: str, batch_size: int) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(embed_texts(texts[i : i + batch_size], model=model))
        return out

    # ── Chunks ────────────────────────────────────────────────────────────────

    def _upsert_chunk(
        self,
        *,
        tenant_id: UUID,
        document_id: UUID,
        chunk_index: int,
        start_page: Optional[int],
        end_page: Optional[int],
        text: str,
        token_count: Optional[int],
        metadata: JsonDict,
        embedding: Optional[List[float]],
    ) -> UUID:
        res = self.sb.rpc(
            "upsert_chunk",
            {
                "p_tenant_id": str(tenant_id),
                "p_document_id": str(document_id),
                "p_chunk_index": chunk_index,
                "p_page_start": start_page,
                "p_page_end": end_page,
                "p_content": text,
                "p_content_tokens": token_count,
                "p_metadata": metadata or {},
                "p_embedding": embedding,
            },
        ).execute()
        return UUID(str(res.data))

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune_kg(self, *, tenant_id: UUID, client_id: UUID) -> JsonDict:
        res = self.sb.rpc(
            "prune_kg",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_edge_stale_days": 90,
                "p_node_stale_days": 180,
                "p_min_degree": 3,
                "p_keep_edge_evidence": 5,
                "p_keep_node_evidence": 10,
            },
        ).execute()
        return res.data or {}

    # ── Shared embed + store ──────────────────────────────────────────────────

    def _store_chunks(
        self,
        *,
        chunks: List[JsonDict],
        tenant_id: UUID,
        document_id: UUID,
        source_uri: str,
        source_type: str,
        extra_metadata: JsonDict,
        embed_model: str,
        embed_batch_size: int,
    ) -> Tuple[List[UUID], List[str]]:
        warnings: List[str] = []
        chunk_ids: List[UUID] = []

        if not chunks:
            return chunk_ids, warnings

        texts = [c["text"] for c in chunks]
        try:
            embeddings = self._embed_in_batches(texts, model=embed_model, batch_size=embed_batch_size)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks"
            )

        for idx, (chunk_data, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                chunk_id = self._upsert_chunk(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    chunk_index=idx,
                    start_page=chunk_data.get("start_page"),
                    end_page=chunk_data.get("end_page"),
                    text=chunk_data["text"],
                    token_count=chunk_data.get("token_count"),
                    metadata={
                        "source_uri": source_uri,
                        "source_type": source_type,
                        "chunk_start_page": chunk_data.get("start_page"),
                        "chunk_end_page": chunk_data.get("end_page"),
                        **extra_metadata,
                    },
                    embedding=embedding,
                )
                chunk_ids.append(chunk_id)
            except Exception as e:
                warnings.append(f"chunk {idx} upsert failed: {e}")
                logger.warning("chunk %d upsert failed: %s", idx, e)

        return chunk_ids, warnings

    # ── File ingest ───────────────────────────────────────────────────────────

    def _ingest_file(self, inp: IngestInput) -> IngestOutput:
        if not inp.file_bytes:
            raise ValueError("file_bytes is required for PDF/DOCX ingest")
        if not inp.file_name:
            raise ValueError("file_name is required for PDF/DOCX ingest")

        file_name = inp.file_name
        file_type = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        if file_type not in _SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type '{file_type}'. Supported: pdf, docx, vtt, xlsx.")

        storage_path = self.upload_to_bucket(inp.file_bytes, file_name)
        source_uri = self._storage_uri(PDF_BUCKET, storage_path)

        document_id = self._upsert_document(
            tenant_id=inp.tenant_id,
            client_id=inp.client_id,
            source_type=file_type,
            source_uri=source_uri,
            title=inp.title or file_name,
            metadata={
                **(inp.metadata or {}),
                "bucket": PDF_BUCKET,
                "object_path": storage_path,
                "file_type": file_type,
                "file_name": file_name,
            },
        )
        logger.info("Upserted document %s (%s)", document_id, file_name)

        chunks = document_bytes_to_chunks(inp.file_bytes, file_type=file_type)
        logger.info("Tokenized %d chunks from %s", len(chunks), file_name)

        if not chunks:
            return IngestOutput(
                document_id=document_id,
                source_type=file_type,
                source_uri=source_uri,
                chunks_upserted=0,
                chunk_ids=[],
                warnings=["Tokenizer produced no chunks — document may be empty or unreadable."],
            )

        chunk_ids, warnings = self._store_chunks(
            chunks=chunks,
            tenant_id=inp.tenant_id,
            document_id=document_id,
            source_uri=source_uri,
            source_type=file_type,
            extra_metadata={"file_name": file_name},
            embed_model=inp.embed_model,
            embed_batch_size=inp.embed_batch_size,
        )

        return IngestOutput(
            document_id=document_id,
            source_type=file_type,
            source_uri=source_uri,
            chunks_upserted=len(chunk_ids),
            chunk_ids=chunk_ids,
            warnings=warnings,
        )

    # ── Web ingest ────────────────────────────────────────────────────────────

    def _ingest_web(self, inp: IngestInput) -> IngestOutput:
        if not inp.web_url:
            raise ValueError("web_url is required for web ingest")

        url = inp.web_url
        source_type = "web"

        logger.info("Starting web scrape of %s", url)
        scraped_json = _run_spider_subprocess(url)
        total_pages = scraped_json.get("total_pages", 0)
        logger.info("Spider collected %d pages from %s", total_pages, url)

        document_id = self._upsert_document(
            tenant_id=inp.tenant_id,
            client_id=inp.client_id,
            source_type=source_type,
            source_uri=url,
            title=inp.title or (scraped_json.get("pages") or [{}])[0].get("title") or url,
            metadata={
                **(inp.metadata or {}),
                "scraped_pages": total_pages,
                "scraped_at": scraped_json.get("scraped_at"),
            },
        )

        if total_pages == 0:
            return IngestOutput(
                document_id=document_id,
                source_type=source_type,
                source_uri=url,
                chunks_upserted=0,
                chunk_ids=[],
                warnings=["Spider returned no pages — site may block crawling."],
            )

        chunks = web_scraped_json_to_chunks(scraped_json)
        logger.info("Tokenized %d chunks from %s", len(chunks), url)

        if not chunks:
            return IngestOutput(
                document_id=document_id,
                source_type=source_type,
                source_uri=url,
                chunks_upserted=0,
                chunk_ids=[],
                warnings=["Tokenizer produced no chunks from scraped content."],
            )

        chunk_ids, warnings = self._store_chunks(
            chunks=chunks,
            tenant_id=inp.tenant_id,
            document_id=document_id,
            source_uri=url,
            source_type=source_type,
            extra_metadata={"scraped_url": url},
            embed_model=inp.embed_model,
            embed_batch_size=inp.embed_batch_size,
        )

        return IngestOutput(
            document_id=document_id,
            source_type=source_type,
            source_uri=url,
            chunks_upserted=len(chunk_ids),
            chunk_ids=chunk_ids,
            warnings=warnings,
        )

    # ── Entry point ───────────────────────────────────────────────────────────

    def ingest(self, inp: IngestInput) -> IngestOutput:
        if inp.file_bytes is not None and inp.file_name is not None:
            result = self._ingest_file(inp)
        elif inp.web_url is not None:
            result = self._ingest_web(inp)
        else:
            raise ValueError(
                "IngestInput requires either (file_bytes + file_name) or web_url."
            )

        if inp.prune_after_ingest:
            try:
                result.prune_result = self._prune_kg(
                    tenant_id=inp.tenant_id,
                    client_id=inp.client_id,
                )
            except Exception as e:
                result.warnings.append(f"prune_kg failed: {e}")

        logger.info(
            "Ingest complete — document=%s chunks=%d warnings=%d",
            result.document_id, result.chunks_upserted, len(result.warnings),
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Spider subprocess runner
# Runs src/processing/run_scraper.py in a fresh subprocess on every call so
# Scrapy's CrawlerProcess single-run-per-process limit is never hit inside
# the long-running FastAPI server process.
# ─────────────────────────────────────────────────────────────────────────────

def _run_spider_subprocess(url: str) -> JsonDict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        subprocess.run(
            ["python", "src/processing/run_scraper.py", url, out_path],
            check=True,
            capture_output=True,
        )
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        logger.error("Spider subprocess failed: %s", e.stderr.decode())
        return {"source_url": url, "scraped_at": "", "total_pages": 0, "pages": []}
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)

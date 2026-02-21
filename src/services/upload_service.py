"""
src/services/upload_service.py
-------------------------------
High-level helpers for upload + ingest. Thin wrappers around IngestService
that accept file paths or raw bytes and handle the boilerplate.

Use these from scripts, CLI tools, or anywhere you want a one-liner
to get a file or website into the system.

Import
------
    from src.services.upload_service import (
        upload_and_ingest,
        upload_and_ingest_bytes,
        ingest_website,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from supabase import Client

from src.services.ingest_service import IngestService, IngestInput, IngestOutput

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def upload_and_ingest(
    sb: Client,
    file_path: str | Path,
    *,
    tenant_id: UUID,
    client_id: UUID,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    embed_model: str = "text-embedding-3-small",
    embed_batch_size: int = 64,
    prune_after_ingest: bool = False,
) -> IngestOutput:
    """
    Read a PDF or DOCX from disk and run the full ingest pipeline.

    Steps:
      1. Read file bytes from disk
      2. Upload to Supabase "pdf" bucket
      3. Tokenize with spaCy + tiktoken
      4. Embed chunks with OpenAI
      5. Upsert chunks into Supabase

    Args:
        sb:                  Supabase client (service role key).
        file_path:           Local path to the PDF or DOCX.
        tenant_id:           Tenant UUID.
        client_id:           Client UUID.
        title:               Display title (defaults to filename stem).
        metadata:            Extra JSON stored on the document row.
        embed_model:         OpenAI embedding model.
        embed_batch_size:    Chunks per OpenAI API call.
        prune_after_ingest:  Run prune_kg after storing chunks.

    Returns:
        IngestOutput
    """
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")

    ext = fp.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}")

    return IngestService(sb).ingest(
        IngestInput(
            tenant_id=tenant_id,
            client_id=client_id,
            file_bytes=fp.read_bytes(),
            file_name=fp.name,
            title=title or fp.stem,
            metadata=metadata or {},
            embed_model=embed_model,
            embed_batch_size=embed_batch_size,
            prune_after_ingest=prune_after_ingest,
        )
    )


def upload_and_ingest_bytes(
    sb: Client,
    file_bytes: bytes,
    file_name: str,
    *,
    tenant_id: UUID,
    client_id: UUID,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    embed_model: str = "text-embedding-3-small",
    embed_batch_size: int = 64,
    prune_after_ingest: bool = False,
) -> IngestOutput:
    """
    Same as upload_and_ingest but accepts raw bytes — use in FastAPI endpoints
    where the file is already in memory from UploadFile.

    Example
    -------
        @router.post("/upload")
        async def upload(file: UploadFile, tenant_id: UUID, client_id: UUID):
            result = upload_and_ingest_bytes(
                sb=get_supabase(),
                file_bytes=await file.read(),
                file_name=file.filename,
                tenant_id=tenant_id,
                client_id=client_id,
            )
            return {"document_id": str(result.document_id)}
    """
    ext = Path(file_name).suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}")

    return IngestService(sb).ingest(
        IngestInput(
            tenant_id=tenant_id,
            client_id=client_id,
            file_bytes=file_bytes,
            file_name=file_name,
            title=title or Path(file_name).stem,
            metadata=metadata or {},
            embed_model=embed_model,
            embed_batch_size=embed_batch_size,
            prune_after_ingest=prune_after_ingest,
        )
    )


def ingest_website(
    sb: Client,
    url: str,
    *,
    tenant_id: UUID,
    client_id: UUID,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    embed_model: str = "text-embedding-3-small",
    embed_batch_size: int = 64,
    prune_after_ingest: bool = False,
) -> IngestOutput:
    """
    Scrape a website and run the full ingest pipeline.

    Steps:
      1. Run Scrapy SiteSpider via subprocess (safe for long-running servers)
      2. Tokenize scraped pages with spaCy + tiktoken
      3. Embed chunks with OpenAI
      4. Upsert chunks into Supabase

    Args:
        sb:                  Supabase client.
        url:                 Root URL to crawl (e.g. "https://example.com").
        tenant_id:           Tenant UUID.
        client_id:           Client UUID.
        title:               Display title (defaults to first page title or URL).
        metadata:            Extra JSON stored on the document row.
        embed_model:         OpenAI embedding model.
        embed_batch_size:    Chunks per OpenAI API call.
        prune_after_ingest:  Run prune_kg after storing chunks.

    Returns:
        IngestOutput
    """
    return IngestService(sb).ingest(
        IngestInput(
            tenant_id=tenant_id,
            client_id=client_id,
            web_url=url,
            title=title,
            metadata=metadata or {},
            embed_model=embed_model,
            embed_batch_size=embed_batch_size,
            prune_after_ingest=prune_after_ingest,
        )
    )

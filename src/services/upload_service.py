"""
src/services/upload_service.py
-------------------------------
High-level wrappers for upload + ingest. Thin layer around IngestService
that accepts file paths or raw bytes and handles the boilerplate.

Import
------
    from src.services.upload_service import UploadService
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


class UploadService:
    """Convenience wrapper around IngestService for file/web uploads."""

    def __init__(self, supabase: Client):
        self.sb = supabase
        self._ingest = IngestService(supabase)

    def upload_and_ingest(
        self,
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
        """Read a PDF or DOCX from disk and run the full ingest pipeline."""
        fp = Path(file_path)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")

        ext = fp.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
            )

        return self._ingest.ingest(
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
        self,
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
        """Ingest raw bytes (e.g. from FastAPI UploadFile)."""
        ext = Path(file_name).suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
            )

        return self._ingest.ingest(
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
        self,
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
        """Scrape a website and run the full ingest pipeline."""
        return self._ingest.ingest(
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


# ── Module-level convenience functions for backward compat ────────────────────

def upload_and_ingest(sb: Client, file_path: str | Path, **kwargs) -> IngestOutput:
    return UploadService(sb).upload_and_ingest(file_path, **kwargs)


def upload_and_ingest_bytes(
    sb: Client, file_bytes: bytes, file_name: str, **kwargs
) -> IngestOutput:
    return UploadService(sb).upload_and_ingest_bytes(file_bytes, file_name, **kwargs)


def ingest_website(sb: Client, url: str, **kwargs) -> IngestOutput:
    return UploadService(sb).ingest_website(url, **kwargs)

"""
upload_helper.py
----------------
Standalone helper for uploading documents to the Supabase "pdf" bucket
and then kicking off IngestController.

Use this when you have a file on disk (or uploaded via HTTP) and want to:
  1. Push it to Supabase Storage under the "pdf" bucket
  2. Hand it off to IngestController for chunking, embedding, and storage

Typical usage
-------------
    # From a script
    from upload_helper import upload_and_ingest, upload_file_to_bucket
    from supabase import create_client
    import uuid

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    tenant_id = uuid.UUID("your-tenant-uuid")
    client_id = uuid.UUID("your-client-uuid")

    result = upload_and_ingest(
        sb=sb,
        file_path="docs/brochure.pdf",
        tenant_id=tenant_id,
        client_id=client_id,
        title="Product Brochure 2024",
    )
    print(result)

    # From a FastAPI endpoint (bytes already in memory)
    result = upload_and_ingest_bytes(
        sb=sb,
        file_bytes=request_body,
        file_name="uploaded.pdf",
        tenant_id=tenant_id,
        client_id=client_id,
    )

Bucket setup
------------
Make sure the "pdf" bucket exists in your Supabase project.
You can create it via the Supabase dashboard (Storage → New bucket)
or programmatically:

    sb.storage.create_bucket("pdf", options={"public": False})
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

import dotenv
from supabase import Client, create_client

from ingest_controller import IngestController, IngestInput, IngestOutput

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# The one bucket all documents go to
PDF_BUCKET = "pdf"

# Supported extensions → how they're labelled in the pipeline
_SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level: just upload, no ingest
# ─────────────────────────────────────────────────────────────────────────────

def upload_file_to_bucket(
    sb: Client,
    file_path: str | Path,
    *,
    storage_path: Optional[str] = None,
    bucket: str = PDF_BUCKET,
) -> str:
    """
    Upload a file from disk to Supabase Storage.

    Args:
        sb:            Supabase client (service role key recommended).
        file_path:     Local path to the file.
        storage_path:  Path inside the bucket (default: the file's name).
        bucket:        Supabase Storage bucket name (default: "pdf").

    Returns:
        The storage path the file was uploaded to (e.g. "brochure.pdf").

    Raises:
        FileNotFoundError: if file_path does not exist.
        ValueError: if the file extension is not supported.
    """
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")

    ext = fp.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    dest = storage_path or fp.name
    file_bytes = fp.read_bytes()

    sb.storage.from_(bucket).upload(
        dest,
        file_bytes,
        file_options={"upsert": "true"},
    )
    logger.info("Uploaded '%s' → bucket '%s' at path '%s'", fp.name, bucket, dest)
    return dest


def upload_bytes_to_bucket(
    sb: Client,
    file_bytes: bytes,
    file_name: str,
    *,
    bucket: str = PDF_BUCKET,
) -> str:
    """
    Upload raw bytes to Supabase Storage.

    Args:
        sb:          Supabase client.
        file_bytes:  Raw file content.
        file_name:   Filename including extension (used as storage path).
        bucket:      Storage bucket (default: "pdf").

    Returns:
        The storage path the file was uploaded to.
    """
    ext = Path(file_name).suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    path = file_name.lstrip("/")
    sb.storage.from_(bucket).upload(
        path,
        file_bytes,
        file_options={"upsert": "true"},
    )
    logger.info("Uploaded %d bytes → bucket '%s' at path '%s'", len(file_bytes), bucket, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# High-level: upload + full ingest pipeline
# ─────────────────────────────────────────────────────────────────────────────

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
    Upload a PDF or DOCX from disk, then run the full ingest pipeline.

    This is the primary entry point for file-based ingest. It:
      1. Reads the file from disk
      2. Uploads it to the Supabase "pdf" bucket
      3. Tokenizes it with tokenization.document_bytes_to_chunks()
      4. Embeds each chunk with OpenAI text-embedding-3-small
      5. Upserts every chunk (with embedding) into the chunks table

    Args:
        sb:                  Supabase client (use service role key).
        file_path:           Path to the PDF or DOCX on disk.
        tenant_id:           Your tenant UUID.
        client_id:           Your client UUID.
        title:               Optional display title (defaults to filename).
        metadata:            Extra JSON metadata stored on the document row.
        embed_model:         OpenAI embedding model.
        embed_batch_size:    Number of chunks to embed per API call.
        prune_after_ingest:  Run prune_kg after chunks are stored.

    Returns:
        IngestOutput with document_id, chunk_ids, warnings, etc.
    """
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")

    file_bytes = fp.read_bytes()
    controller = IngestController(sb)

    return controller.ingest(
        IngestInput(
            tenant_id=tenant_id,
            client_id=client_id,
            file_bytes=file_bytes,
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
    Same as upload_and_ingest but accepts raw bytes instead of a file path.
    Use this when the file is already in memory (e.g. from a FastAPI upload).

    Example in a FastAPI endpoint
    ------------------------------
        @app.post("/upload")
        async def upload_doc(
            file: UploadFile,
            tenant_id: UUID,
            client_id: UUID,
        ):
            result = upload_and_ingest_bytes(
                sb=get_supabase(),
                file_bytes=await file.read(),
                file_name=file.filename,
                tenant_id=tenant_id,
                client_id=client_id,
            )
            return {"document_id": str(result.document_id), "chunks": result.chunks_upserted}
    """
    controller = IngestController(sb)

    return controller.ingest(
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
    Scrape a website with SiteSpider and run the full ingest pipeline.

    This is the primary entry point for web-based ingest. It:
      1. Runs the Scrapy SiteSpider on the URL
      2. Tokenizes scraped pages with tokenization.web_scraped_json_to_chunks()
      3. Embeds each chunk with OpenAI
      4. Upserts every chunk into the chunks table

    Args:
        sb:                  Supabase client.
        url:                 Root URL to crawl (e.g. "https://example.com").
        tenant_id:           Your tenant UUID.
        client_id:           Your client UUID.
        title:               Optional title override (defaults to first page title or URL).
        metadata:            Extra JSON metadata stored on the document row.
        embed_model:         OpenAI embedding model.
        embed_batch_size:    Chunks per OpenAI API call.
        prune_after_ingest:  Run prune_kg after chunks are stored.

    Returns:
        IngestOutput with document_id, chunk_ids, warnings, etc.
    """
    controller = IngestController(sb)

    return controller.ingest(
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI convenience — run directly to test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import uuid

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
    TENANT_ID    = uuid.UUID(os.environ["TENANT_ID"])
    CLIENT_ID    = uuid.UUID(os.environ["CLIENT_ID"])

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python upload_helper.py path/to/file.pdf")
        print("  python upload_helper.py https://example.com")
        sys.exit(1)

    target = sys.argv[1]

    if target.startswith("http://") or target.startswith("https://"):
        print(f"Ingesting website: {target}")
        result = ingest_website(sb, target, tenant_id=TENANT_ID, client_id=CLIENT_ID)
    else:
        print(f"Ingesting file: {target}")
        result = upload_and_ingest(sb, target, tenant_id=TENANT_ID, client_id=CLIENT_ID)

    print(f"\n✓ document_id : {result.document_id}")
    print(f"  source_type  : {result.source_type}")
    print(f"  source_uri   : {result.source_uri}")
    print(f"  chunks stored: {result.chunks_upserted}")
    if result.warnings:
        print(f"  warnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"    - {w}")

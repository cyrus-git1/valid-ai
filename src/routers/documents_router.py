"""
/documents router
-----------------
CRUD over the documents table and their associated chunks.

GET    /documents                      — List documents (paginated, scoped to tenant+client)
GET    /documents/{document_id}        — Get a single document
PATCH  /documents/{document_id}        — Update title or metadata
DELETE /documents/{document_id}        — Delete document + cascade chunks + KG nodes
GET    /documents/{document_id}/chunks — List chunks for a document
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from src.supabase.supabase_client import get_supabase
from src.models.api.documents import (
    ChunkListResponse,
    ChunkResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdateRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _require_document(sb, document_id: str, tenant_id: UUID) -> dict:
    """Fetch a document row or raise 404."""
    res = (
        sb.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("tenant_id", str(tenant_id))
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return res.data[0]


@router.get("", response_model=DocumentListResponse)
def list_documents(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    """
    List all documents for a tenant + client, newest first.
    Supports pagination via limit/offset.
    """
    sb = get_supabase()

    res = (
        sb.table("documents")
        .select("*", count="exact")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )

    items = [DocumentResponse(**row) for row in (res.data or [])]
    total = res.count or 0

    return DocumentListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    tenant_id: UUID = Query(...),
) -> DocumentResponse:
    """Get a single document by ID."""
    sb = get_supabase()
    row = _require_document(sb, document_id, tenant_id)
    return DocumentResponse(**row)


@router.patch("/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: str,
    body: DocumentUpdateRequest,
    tenant_id: UUID = Query(...),
) -> DocumentResponse:
    """
    Update a document's title and/or metadata.
    Metadata is merged (patched), not replaced.
    """
    sb = get_supabase()
    existing = _require_document(sb, document_id, tenant_id)

    patch: dict = {}
    if body.title is not None:
        patch["title"] = body.title
    if body.metadata is not None:
        # merge — keep existing keys, overlay with new keys
        patch["metadata"] = {**existing.get("metadata", {}), **body.metadata}

    if not patch:
        return DocumentResponse(**existing)

    res = (
        sb.table("documents")
        .update(patch)
        .eq("id", document_id)
        .eq("tenant_id", str(tenant_id))
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=500, detail="Update failed — no rows returned.")

    return DocumentResponse(**res.data[0])


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    tenant_id: UUID = Query(...),
) -> None:
    """
    Delete a document and all its associated data.

    Cascade deletes (via ON DELETE CASCADE in SQL schema):
      - chunks (03_chunks.sql)
      - kg_nodes whose properties.document_id matches (handled by chunk cascade)
      - kg_node_evidence, kg_edge_evidence (via chunk foreign keys)

    KG edges between deleted nodes are also cleaned up by Postgres cascade.
    """
    sb = get_supabase()
    _require_document(sb, document_id, tenant_id)  # 404 if not found

    sb.table("documents").delete().eq("id", document_id).eq("tenant_id", str(tenant_id)).execute()
    logger.info("Deleted document %s", document_id)


@router.get("/{document_id}/chunks", response_model=ChunkListResponse)
def list_chunks(
    document_id: str,
    tenant_id: UUID = Query(...),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ChunkListResponse:
    """
    List all chunks for a document, ordered by chunk_index.
    Includes a has_embedding flag so you can see which chunks are ready for KG build.
    """
    sb = get_supabase()
    _require_document(sb, document_id, tenant_id)  # 404 guard

    res = (
        sb.table("chunks")
        .select(
            "id, document_id, chunk_index, page_start, page_end, "
            "content, content_tokens, metadata, embedding",
            count="exact",
        )
        .eq("tenant_id", str(tenant_id))
        .eq("document_id", document_id)
        .order("chunk_index")
        .range(offset, offset + limit - 1)
        .execute()
    )

    items = [
        ChunkResponse(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            content=row["content"],
            content_tokens=row.get("content_tokens"),
            metadata=row.get("metadata") or {},
            has_embedding=row.get("embedding") is not None,
        )
        for row in (res.data or [])
    ]

    return ChunkListResponse(items=items, total=res.count or 0, limit=limit, offset=offset)

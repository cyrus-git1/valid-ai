"""
/documents router
-----------------
GET    /documents         — List all documents + chunks for a tenant
DELETE /documents         — Bulk-delete documents + cascade chunks by tenant + document IDs
"""
from __future__ import annotations

import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from src.supabase.supabase_client import get_supabase
from src.models.api.documents import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    DocumentWithChunksResponse,
    DocumentWithChunksListResponse,
    ChunkResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentWithChunksListResponse)
def list_documents(
    tenant_id: UUID = Query(...),
) -> DocumentWithChunksListResponse:
    """
    Return every document for a tenant (across all clients) with all
    associated chunks embedded in each document object.
    """
    sb = get_supabase()

    # Fetch all documents for the tenant
    doc_res = (
        sb.table("documents")
        .select("*")
        .eq("tenant_id", str(tenant_id))
        .order("created_at", desc=True)
        .execute()
    )
    docs = doc_res.data or []

    if not docs:
        return DocumentWithChunksListResponse(items=[], total=0)

    # Fetch all chunks for those documents in one query
    doc_ids = [d["id"] for d in docs]
    chunk_res = (
        sb.table("chunks")
        .select(
            "id, document_id, chunk_index, page_start, page_end, "
            "content, content_tokens, metadata, embedding"
        )
        .eq("tenant_id", str(tenant_id))
        .in_("document_id", doc_ids)
        .order("chunk_index")
        .execute()
    )

    # Group chunks by document_id
    chunks_by_doc: dict[str, list[ChunkResponse]] = {}
    for row in (chunk_res.data or []):
        cr = ChunkResponse(
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
        chunks_by_doc.setdefault(row["document_id"], []).append(cr)

    items = [
        DocumentWithChunksResponse(
            id=d["id"],
            tenant_id=d["tenant_id"],
            client_id=d.get("client_id"),
            source_type=d["source_type"],
            source_uri=d.get("source_uri"),
            title=d.get("title"),
            metadata=d.get("metadata") or {},
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            chunks=chunks_by_doc.get(d["id"], []),
        )
        for d in docs
    ]

    return DocumentWithChunksListResponse(items=items, total=len(items))


@router.delete("", response_model=BulkDeleteResponse)
def delete_documents(
    body: BulkDeleteRequest,
    tenant_id: UUID = Query(...),
) -> BulkDeleteResponse:
    """
    Delete documents by tenant_id + list of document_ids.

    Cascade deletes (via ON DELETE CASCADE in SQL schema):
      - chunks
      - kg_node_evidence, kg_edge_evidence (via chunk foreign keys)
      - KG edges between deleted nodes
    """
    sb = get_supabase()
    deleted = 0
    not_found: list[str] = []

    for doc_id in body.document_ids:
        res = (
            sb.table("documents")
            .select("id")
            .eq("id", doc_id)
            .eq("tenant_id", str(tenant_id))
            .limit(1)
            .execute()
        )
        if not res.data:
            not_found.append(doc_id)
            continue

        sb.table("documents").delete().eq("id", doc_id).eq("tenant_id", str(tenant_id)).execute()
        deleted += 1
        logger.info("Deleted document %s (tenant %s)", doc_id, tenant_id)

    return BulkDeleteResponse(deleted=deleted, not_found=not_found)

"""
/data router
------------
Internal data endpoints for the agent service.

This router exposes only generic document metadata helpers and durable
context-summary reads/writes. Transcript- and survey-specific artifacts
should be serialized by the agent and ingested through the generic
processed ingest endpoints instead.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.services.audit_service import AuditService
from src.services.memory_state_service import MemoryStateService
from src.services.redaction import apply_redaction, caller_can_reveal
from src.supabase.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])


# ── Documents listing ────────────────────────────────────────────────────────


@router.get("/documents")
def list_documents(
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
):
    """
    Return every document for a tenant+client with all associated chunks.
    Raw data endpoint for the agent service.

    Chunk content is PII-redacted (via chunks.pii_annotations) unless the
    caller's API key has the `pii:reveal` scope.
    """
    sb = get_supabase()
    can_reveal = caller_can_reveal(request)

    doc_res = (
        sb.table("documents")
        .select("*")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .order("created_at", desc=True)
        .execute()
    )
    docs = doc_res.data or []

    if not docs:
        return {"items": [], "total": 0}

    doc_ids = [d["id"] for d in docs]
    chunk_res = (
        sb.table("chunks")
        .select(
            "id, document_id, chunk_index, page_start, page_end, "
            "content, content_tokens, metadata, embedding, pii_annotations"
        )
        .eq("tenant_id", str(tenant_id))
        .in_("document_id", doc_ids)
        .order("chunk_index")
        .execute()
    )

    chunks_by_doc: Dict[str, list] = {}
    for row in (chunk_res.data or []):
        content = row["content"]
        annotations = row.get("pii_annotations") or []
        if not can_reveal and annotations:
            content = apply_redaction(content, annotations)
        chunk = {
            "id": row["id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "page_start": row.get("page_start"),
            "page_end": row.get("page_end"),
            "content": content,
            "content_tokens": row.get("content_tokens"),
            "metadata": row.get("metadata") or {},
            "has_embedding": row.get("embedding") is not None,
            "has_pii": bool(annotations),
        }
        chunks_by_doc.setdefault(row["document_id"], []).append(chunk)

    if can_reveal:
        logger.warning(
            "data.documents PII_REVEAL tenant=%s client=%s key_id=%s",
            tenant_id, client_id,
            getattr(getattr(request, "state", None), "key_id", None),
        )

    items = [
        {
            "id": d["id"],
            "tenant_id": d["tenant_id"],
            "client_id": d.get("client_id"),
            "source_type": d["source_type"],
            "source_uri": d.get("source_uri"),
            "title": d.get("title"),
            "source_timestamp": d.get("source_timestamp"),
            "is_pinned": d.get("is_pinned", False),
            "is_canonical": d.get("is_canonical", False),
            "status": d.get("status", "active"),
            "metadata": d.get("metadata") or {},
            "created_at": d["created_at"],
            "updated_at": d["updated_at"],
            "chunks": chunks_by_doc.get(d["id"], []),
        }
        for d in docs
    ]

    return {"items": items, "total": len(items)}


# ── Document update (flags/status) ───────────────────────────────────────────


class _DocumentPatchRequest(BaseModel):
    status: Optional[str] = Field(default=None, description="active | draft | deprecated | archived | flagged")
    is_pinned: Optional[bool] = None
    is_canonical: Optional[bool] = None


@router.patch("/documents/{document_id}")
def patch_document(
    document_id: str,
    body: _DocumentPatchRequest,
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
):
    """
    Update document flags (status, is_pinned, is_canonical).
    Bumps memory state on change.
    """
    sb = get_supabase()

    # Verify document exists
    res = (
        sb.table("documents")
        .select("id")
        .eq("id", document_id)
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")

    updates: Dict[str, Any] = {}
    if body.status is not None:
        if body.status not in ("active", "draft", "deprecated", "archived", "flagged"):
            raise HTTPException(status_code=400, detail=f"Invalid status: {body.status}")
        updates["status"] = body.status
    if body.is_pinned is not None:
        updates["is_pinned"] = body.is_pinned
    if body.is_canonical is not None:
        updates["is_canonical"] = body.is_canonical

    if not updates:
        return {"updated": False, "document_id": document_id}

    sb.table("documents").update(updates).eq("id", document_id).eq("tenant_id", str(tenant_id)).eq("client_id", str(client_id)).execute()

    memory_state = MemoryStateService(sb)
    memory_state.bump_dual(
        tenant_id=tenant_id,
        client_id=client_id,
        change_type="update",
        metadata={"document_id": document_id, **updates},
    )

    AuditService(sb).record(
        request=request,
        action="document.patch",
        resource_type="document",
        resource_id=document_id,
        metadata={"updates": updates},
    )

    logger.info(
        "data.documents.patch tenant=%s client=%s document=%s updates=%s",
        tenant_id, client_id, document_id, updates,
    )
    return {"updated": True, "document_id": document_id, **updates}


# ── Documents deletion ───────────────────────────────────────────────────────


class _BulkDeleteRequest(BaseModel):
    document_ids: List[str] = Field(min_length=1)


@router.post("/documents/delete")
def delete_documents(
    body: _BulkDeleteRequest,
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
):
    """
    Delete documents by tenant_id + list of document_ids.
    Bumps memory state and cleans up orphaned KG nodes.
    Raw data endpoint for the agent service.
    """
    sb = get_supabase()
    memory_state = MemoryStateService(sb)
    deleted = 0
    not_found: list[str] = []
    affected_client_ids: set[str] = set()

    for doc_id in body.document_ids:
        res = (
            sb.table("documents")
            .select("id, client_id")
            .eq("id", doc_id)
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .limit(1)
            .execute()
        )
        if not res.data:
            not_found.append(doc_id)
            continue

        row_client_id = res.data[0].get("client_id")
        if row_client_id:
            affected_client_ids.add(row_client_id)

        sb.table("documents").delete().eq("id", doc_id).eq("tenant_id", str(tenant_id)).eq("client_id", str(client_id)).execute()
        deleted += 1
        logger.info("Deleted document %s (tenant %s, client %s)", doc_id, tenant_id, client_id)

    if deleted > 0:
        versions = memory_state.bump_dual(
            tenant_id=tenant_id,
            client_id=client_id,
            change_type="delete",
            metadata={"deleted_documents": deleted, "client_id": str(client_id)},
        )
        for affected_client_id in affected_client_ids:
            try:
                cleanup = sb.rpc(
                    "cleanup_orphaned_kg_nodes",
                    {"p_tenant_id": str(tenant_id), "p_client_id": affected_client_id},
                ).execute()
                logger.info(
                    "KG cleanup for tenant=%s client=%s: %s",
                    tenant_id, affected_client_id, cleanup.data,
                )
            except Exception as e:
                logger.warning("KG orphan cleanup failed for client %s: %s", affected_client_id, e)
        logger.info(
            "data.documents.delete tenant=%s client=%s deleted=%d not_found=%d client_version=%s tenant_version=%s",
            tenant_id,
            client_id,
            deleted,
            len(not_found),
            versions["client_version"],
            versions["tenant_version"],
        )

    AuditService(sb).record(
        request=request,
        action="document.delete",
        resource_type="document",
        resource_id=",".join(body.document_ids),
        metadata={"deleted": deleted, "not_found_count": len(not_found)},
    )
    return {"deleted": deleted, "not_found": not_found}


# ── Document titles ──────────────────────────────────────────────────────────


class DocumentTitlesRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    document_ids: List[str]


class DocumentTitlesResponse(BaseModel):
    titles: Dict[str, str]


@router.post("/document-titles", response_model=DocumentTitlesResponse)
def get_document_titles(req: DocumentTitlesRequest) -> DocumentTitlesResponse:
    """Resolve document IDs to titles."""
    if not req.document_ids:
        return DocumentTitlesResponse(titles={})

    sb = get_supabase()
    try:
        res = (
            sb.table("documents")
            .select("id, title, source_uri, source_type")
            .eq("tenant_id", str(req.tenant_id))
            .eq("client_id", str(req.client_id))
            .in_("id", req.document_ids)
            .execute()
        )
        titles = {}
        for row in (res.data or []):
            if row.get("title"):
                titles[row["id"]] = row["title"]
            elif row.get("source_type") == "web" and row.get("source_uri"):
                titles[row["id"]] = row["source_uri"]
        return DocumentTitlesResponse(titles=titles)
    except Exception as e:
        logger.exception("Failed to resolve document titles")
        raise HTTPException(status_code=500, detail=str(e))


class ContextSummaryReadResponse(BaseModel):
    document_id: str
    tenant_id: str
    client_id: str
    source_type: str
    summary: str
    topics: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_stats: Dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: List[str] = Field(default_factory=list)
    memory_version_at_generation: int = 0
    current_memory_version: int = 0
    is_stale: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def _summary_read_from_docrow(row: Dict[str, Any], chunk_content: str, current_version: int) -> ContextSummaryReadResponse:
    md = row.get("metadata") or {}
    gen_version = int(md.get("memory_version_at_generation") or 0)
    return ContextSummaryReadResponse(
        document_id=row["id"],
        tenant_id=row["tenant_id"],
        client_id=row.get("client_id") or "",
        source_type=row.get("source_type") or "ContextSummary",
        summary=chunk_content,
        topics=md.get("topics") or [],
        metadata=md,
        source_stats=md.get("source_stats") or {},
        source_chunk_ids=md.get("source_chunk_ids") or [],
        memory_version_at_generation=gen_version,
        current_memory_version=current_version,
        is_stale=gen_version > 0 and gen_version < current_version,
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


@router.get("/context/summary/get", response_model=ContextSummaryReadResponse)
def get_context_summary(
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> ContextSummaryReadResponse:
    """Fetch the canonical ContextSummary document + its chunk content.

    Summary content is PII-redacted unless caller has `pii:reveal` scope.
    """
    sb = get_supabase()
    can_reveal = caller_can_reveal(request)
    try:
        doc = (
            sb.table("documents")
            .select("id, tenant_id, client_id, source_type, metadata, created_at, updated_at")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .eq("source_type", "ContextSummary")
            .eq("is_canonical", True)
            .eq("status", "active")
            .limit(1)
            .execute()
        )
        rows = doc.data or []
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No context summary found for tenant={tenant_id}, client={client_id}.",
            )
        row = rows[0]

        # Fetch the single chunk for this summary doc (with annotations)
        chunk_res = (
            sb.table("chunks")
            .select("content, pii_annotations")
            .eq("tenant_id", str(tenant_id))
            .eq("document_id", row["id"])
            .order("chunk_index")
            .limit(1)
            .execute()
        )
        chunk_rows = chunk_res.data or []
        if chunk_rows:
            raw_content = chunk_rows[0]["content"]
            annotations = chunk_rows[0].get("pii_annotations") or []
            content = raw_content if can_reveal else apply_redaction(raw_content, annotations)
        else:
            content = ""

        # Staleness: compare stored memory_version vs current client-scoped version
        state = MemoryStateService(sb).get_state(tenant_id=tenant_id, client_id=client_id)
        current_version = int(state.get("memory_version") or 0)
        return _summary_read_from_docrow(row, content, current_version)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch context summary")
        raise HTTPException(status_code=500, detail=str(e))


# ── /data/summaries — list summaries across granularities ───────────────────


class SummaryListItem(BaseModel):
    document_id: str
    source_type: str
    tenant_id: str
    client_id: Optional[str] = None
    scope_ref: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    memory_version_at_generation: int = 0
    is_stale: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SummaryListResponse(BaseModel):
    items: List[SummaryListItem]
    total: int


@router.get("/summaries", response_model=SummaryListResponse)
def list_summaries(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    source_type: Optional[str] = Query(None, description="ContextSummary | DocumentSummary | TopicSummary"),
) -> SummaryListResponse:
    """List canonical summaries for a (tenant, client), optionally filtered by type."""
    sb = get_supabase()
    q = (
        sb.table("documents")
        .select("id, tenant_id, client_id, source_type, metadata, created_at, updated_at")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .eq("is_canonical", True)
        .eq("status", "active")
        .in_("source_type", [source_type] if source_type else ["ContextSummary", "DocumentSummary", "TopicSummary"])
        .order("updated_at", desc=True)
    )
    rows = q.execute().data or []

    state = MemoryStateService(sb).get_state(tenant_id=tenant_id, client_id=client_id)
    current_version = int(state.get("memory_version") or 0)

    items: List[SummaryListItem] = []
    for r in rows:
        md = r.get("metadata") or {}
        gen_version = int(md.get("memory_version_at_generation") or 0)
        scope_ref = md.get("topic") or md.get("document_id")
        items.append(SummaryListItem(
            document_id=r["id"],
            source_type=r["source_type"],
            tenant_id=r["tenant_id"],
            client_id=r.get("client_id"),
            scope_ref=scope_ref,
            topics=md.get("topics") or [],
            memory_version_at_generation=gen_version,
            is_stale=gen_version > 0 and gen_version < current_version,
            created_at=r.get("created_at"),
            updated_at=r.get("updated_at"),
        ))

    return SummaryListResponse(items=items, total=len(items))


@router.get("/summaries/document/{document_id}", response_model=ContextSummaryReadResponse)
def get_document_summary(
    document_id: str,
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> ContextSummaryReadResponse:
    """Fetch the canonical DocumentSummary for a given source document_id."""
    return _fetch_scoped_summary(
        tenant_id, client_id, "DocumentSummary", "document_id", document_id,
        can_reveal=caller_can_reveal(request),
    )


@router.get("/summaries/topic", response_model=ContextSummaryReadResponse)
def get_topic_summary(
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    topic: str = Query(..., min_length=1),
) -> ContextSummaryReadResponse:
    """Fetch the canonical TopicSummary for a given topic string."""
    return _fetch_scoped_summary(
        tenant_id, client_id, "TopicSummary", "topic", topic,
        can_reveal=caller_can_reveal(request),
    )


def _fetch_scoped_summary(
    tenant_id: UUID,
    client_id: UUID,
    source_type: str,
    scope_field: str,
    scope_value: str,
    can_reveal: bool = False,
) -> ContextSummaryReadResponse:
    sb = get_supabase()
    q = (
        sb.table("documents")
        .select("id, tenant_id, client_id, source_type, metadata, created_at, updated_at")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .eq("source_type", source_type)
        .eq("is_canonical", True)
        .eq("status", "active")
    )
    rows = q.execute().data or []
    # Scope match is done in Python — Supabase-py doesn't expose jsonb key equality cleanly.
    row = None
    for r in rows:
        if (r.get("metadata") or {}).get(scope_field) == scope_value:
            row = r
            break
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No {source_type} found for {scope_field}={scope_value}.",
        )

    chunk_res = (
        sb.table("chunks")
        .select("content, pii_annotations")
        .eq("tenant_id", str(tenant_id))
        .eq("document_id", row["id"])
        .order("chunk_index")
        .limit(1)
        .execute()
    )
    chunk_rows = chunk_res.data or []
    if chunk_rows:
        raw_content = chunk_rows[0]["content"]
        annotations = chunk_rows[0].get("pii_annotations") or []
        content = raw_content if can_reveal else apply_redaction(raw_content, annotations)
    else:
        content = ""

    state = MemoryStateService(sb).get_state(tenant_id=tenant_id, client_id=client_id)
    current_version = int(state.get("memory_version") or 0)
    return _summary_read_from_docrow(row, content, current_version)

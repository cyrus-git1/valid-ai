"""
src/services/data_service.py
----------------------------
Document metadata + durable summary-read logic for the /data router, extracted
so the router stays HTTP-only.

The router owns HTTP concerns: it resolves `can_reveal` from the API key
(`caller_can_reveal(request)`) and passes it in as a bool, and passes `request`
through for audit records. This service owns the Supabase queries, RPCs,
PII redaction, memory-state bumps, and audit writes.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import HTTPException

from src.models.api.data import (
    ContextSummaryReadResponse,
    DocumentTitlesRequest,
    DocumentTitlesResponse,
    KgEntitiesResponse,
    KgEntity,
    PreviewUrlResponse,
    SummaryListItem,
    SummaryListResponse,
    SurveyOutputsResponse,
    SurveyQuestions,
    _BulkDeleteRequest,
    _DocumentPatchRequest,
)
from src.services.audit_service import AuditService
from src.services.memory_state_service import MemoryStateService
from src.services.redaction import apply_redaction

logger = logging.getLogger(__name__)

_SUMMARY_SOURCE_TYPES = ("ContextSummary", "DocumentSummary", "TopicSummary")
_PREVIEW_URL_TTL = 300  # 5 minutes


class DataService:
    """Wraps Supabase for /data document + summary operations."""

    def __init__(self, sb):
        self.sb = sb

    # ── documents ────────────────────────────────────────────────────────────

    def list_documents(self, tenant_id: UUID, client_id: UUID, include_summaries: bool,
                       can_reveal: bool, key_id=None) -> dict:
        sb = self.sb
        doc_q = (
            sb.table("documents")
            .select("*")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
        )
        if not include_summaries:
            doc_q = doc_q.not_.in_("source_type", list(_SUMMARY_SOURCE_TYPES))
        doc_res = doc_q.order("created_at", desc=True).execute()
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
                tenant_id, client_id, key_id,
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

    def patch_document(self, document_id: str, body: _DocumentPatchRequest,
                       tenant_id: UUID, client_id: UUID, request) -> dict:
        sb = self.sb

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

    def delete_documents(self, body: _BulkDeleteRequest, tenant_id: UUID,
                         client_id: UUID, request) -> dict:
        sb = self.sb
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

    def get_document_preview_url(self, document_id: str, tenant_id: UUID, client_id: UUID,
                                 expires_in: int) -> PreviewUrlResponse:
        sb = self.sb
        try:
            res = (
                sb.table("documents")
                .select("id, source_uri, source_type, title")
                .eq("id", document_id)
                .eq("tenant_id", str(tenant_id))
                .eq("client_id", str(client_id))
                .limit(1)
                .execute()
            )
        except Exception as e:
            logger.exception("preview-url document lookup failed")
            raise HTTPException(status_code=500, detail=str(e))

        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="document not found for tenant/client")
        doc = rows[0]
        source_uri = doc.get("source_uri") or ""
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()

        # Web-ingested documents point at a real URL, not a bucket object.
        if source_uri.startswith("http://") or source_uri.startswith("https://"):
            return PreviewUrlResponse(
                url=source_uri, expires_at=expires_at,
                source_type=doc.get("source_type"), filename=doc.get("title"),
            )
        if not source_uri.startswith("bucket:"):
            raise HTTPException(status_code=409, detail="document has no retrievable file in storage")

        # source_uri = "bucket:{bucket}/{tenant}/{client}/{filename}"
        bucket, _, object_path = source_uri[len("bucket:"):].partition("/")
        if not bucket or not object_path:
            raise HTTPException(status_code=500, detail="malformed source_uri")

        try:
            signed = sb.storage.from_(bucket).create_signed_url(object_path, expires_in)
        except Exception as e:
            logger.exception("create_signed_url failed for document %s", document_id)
            raise HTTPException(status_code=500, detail=str(e))

        url = None
        if isinstance(signed, dict):
            url = signed.get("signedURL") or signed.get("signedUrl") or signed.get("signed_url")
        elif isinstance(signed, str):
            url = signed
        if not url:
            raise HTTPException(status_code=500, detail="failed to sign url")
        if url.startswith("/"):  # some supabase-py versions return a relative path
            url = os.environ.get("SUPABASE_URL", "").rstrip("/") + url

        logger.info("preview-url minted document=%s tenant=%s ttl=%ds", document_id, tenant_id, expires_in)
        return PreviewUrlResponse(
            url=url, expires_at=expires_at,
            source_type=doc.get("source_type"),
            filename=object_path.rsplit("/", 1)[-1],
        )

    # ── survey outputs ───────────────────────────────────────────────────────

    def get_survey_outputs(self, tenant_id: UUID, client_id, study_id,
                           output_type: Optional[str]) -> SurveyOutputsResponse:
        sb = self.sb
        try:
            q = (
                sb.table("survey_outputs")
                .select("id, output_type, questions, metadata, created_at")
                .eq("tenant_id", str(tenant_id))
                .gte("expires_at", datetime.now(timezone.utc).isoformat())
                .order("created_at", desc=True)
                .limit(200)
            )
            if client_id is not None:
                q = q.eq("client_id", str(client_id))
            if output_type:
                q = q.eq("output_type", output_type)
            rows = q.execute().data or []
        except Exception as e:
            logger.exception("survey-outputs read failed")
            raise HTTPException(status_code=500, detail=str(e))

        surveys: List[SurveyQuestions] = []
        for r in rows:
            meta = r.get("metadata") or {}
            sid = meta.get("study_id")
            if study_id is not None and str(sid) != str(study_id):
                continue
            surveys.append(SurveyQuestions(
                survey_id=str(r.get("id", "")),
                output_type=r.get("output_type", ""),
                study_id=str(sid) if sid else None,
                questions=r.get("questions") or [],
                created_at=str(r["created_at"]) if r.get("created_at") else None,
            ))
        return SurveyOutputsResponse(surveys=surveys)

    # ── KG entities glossary ─────────────────────────────────────────────────

    def get_kg_entities(self, tenant_id: UUID, client_id: UUID, limit: int) -> KgEntitiesResponse:
        sb = self.sb
        try:
            rows = (
                sb.table("kg_nodes")
                .select("name, type, description")
                .eq("tenant_id", str(tenant_id))
                .eq("client_id", str(client_id))
                .eq("status", "active")
                .limit(limit)
                .execute()
            ).data or []
        except Exception as e:
            logger.warning("kg-entities read failed: %s", e)
            rows = []
        return KgEntitiesResponse(entities=[KgEntity(**r) for r in rows])

    # ── document titles ──────────────────────────────────────────────────────

    def get_document_titles(self, req: DocumentTitlesRequest) -> DocumentTitlesResponse:
        if not req.document_ids:
            return DocumentTitlesResponse(titles={})

        sb = self.sb
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

    # ── summaries ────────────────────────────────────────────────────────────

    @staticmethod
    def _summary_read_from_docrow(row: Dict[str, Any], chunk_content: str,
                                  current_version: int) -> ContextSummaryReadResponse:
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

    def get_context_summary(self, tenant_id: UUID, client_id: UUID,
                            can_reveal: bool) -> ContextSummaryReadResponse:
        sb = self.sb
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
            return self._summary_read_from_docrow(row, content, current_version)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to fetch context summary")
            raise HTTPException(status_code=500, detail=str(e))

    def list_summaries(self, tenant_id: UUID, client_id: UUID,
                       source_type: Optional[str]) -> SummaryListResponse:
        sb = self.sb
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

    def _fetch_scoped_summary(self, tenant_id: UUID, client_id: UUID, source_type: str,
                              scope_field: str, scope_value: str,
                              can_reveal: bool = False) -> ContextSummaryReadResponse:
        sb = self.sb
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
        return self._summary_read_from_docrow(row, content, current_version)

    def get_document_summary(self, tenant_id: UUID, client_id: UUID, document_id: str,
                             can_reveal: bool) -> ContextSummaryReadResponse:
        return self._fetch_scoped_summary(
            tenant_id, client_id, "DocumentSummary", "document_id", document_id,
            can_reveal=can_reveal,
        )

    def get_topic_summary(self, tenant_id: UUID, client_id: UUID, topic: str,
                          can_reveal: bool) -> ContextSummaryReadResponse:
        return self._fetch_scoped_summary(
            tenant_id, client_id, "TopicSummary", "topic", topic,
            can_reveal=can_reveal,
        )

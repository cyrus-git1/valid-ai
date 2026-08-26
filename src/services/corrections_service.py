"""
src/services/corrections_service.py
-----------------------------------
Tenant-scoped context corrections (overrides): durable, non-destructive "going
forward" fixes applied where context is READ, never by editing ingested text.

CRUD lives here; the pure text transform lives in retrieval_postprocess.apply_
corrections. On every write we bump the tenant/client memory-state version (cache
invalidation, exactly like patch_document) and record an audit row.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import HTTPException

from src.models.api.corrections import (
    CorrectionCreateRequest,
    CorrectionCreateResponse,
    CorrectionDeleteResponse,
    CorrectionItem,
    CorrectionsListResponse,
)
from src.services.audit_service import AuditService
from src.services.memory_state_service import MemoryStateService
from src.services.retrieval_postprocess import apply_corrections

logger = logging.getLogger(__name__)


class CorrectionsService:
    def __init__(self, sb):
        self.sb = sb

    # ── writes ───────────────────────────────────────────────────────────────

    def create(self, tenant_id: UUID, client_id: UUID,
               body: CorrectionCreateRequest, request) -> CorrectionCreateResponse:
        applies_to: Any = body.applies_to
        if isinstance(applies_to, list):
            applies_to = [str(d) for d in applies_to]

        row = {
            "tenant_id": str(tenant_id),
            "client_id": str(client_id) if client_id else None,
            "kind": body.kind,
            "from_term": body.from_,
            "to_term": body.to if body.kind == "term_replace" else None,
            "note": body.note,
            "applies_to": applies_to,
            "status": "active",
        }
        res = self.sb.table("context_corrections").insert(row).execute()
        created = (res.data or [{}])[0]
        correction_id = str(created.get("id", ""))

        self._invalidate(tenant_id, client_id, "create",
                         {"correction_id": correction_id, "kind": body.kind, "from": body.from_})
        AuditService(self.sb).record(
            request=request, action="context.correction.create",
            resource_type="context_correction", resource_id=correction_id,
            metadata={"kind": body.kind, "from": body.from_, "to": body.to},
        )
        logger.info("data.corrections.create tenant=%s client=%s id=%s kind=%s",
                    tenant_id, client_id, correction_id, body.kind)
        return CorrectionCreateResponse(correction_id=correction_id, applied=True)

    def delete(self, correction_id: str, tenant_id: UUID, client_id: UUID,
               request) -> CorrectionDeleteResponse:
        res = (
            self.sb.table("context_corrections")
            .delete()
            .eq("id", correction_id)
            .eq("tenant_id", str(tenant_id))
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail=f"Correction {correction_id} not found.")
        self._invalidate(tenant_id, client_id, "delete", {"correction_id": correction_id})
        AuditService(self.sb).record(
            request=request, action="context.correction.delete",
            resource_type="context_correction", resource_id=correction_id, metadata={},
        )
        logger.info("data.corrections.delete tenant=%s id=%s", tenant_id, correction_id)
        return CorrectionDeleteResponse(deleted=True, correction_id=correction_id)

    # ── reads ────────────────────────────────────────────────────────────────

    def list_corrections(self, tenant_id: UUID, client_id: UUID) -> CorrectionsListResponse:
        rows = self._active_rows(tenant_id, client_id)
        items = [
            CorrectionItem(
                correction_id=str(r.get("id", "")),
                kind=r.get("kind", ""),
                from_=r.get("from_term", ""),
                to=r.get("to_term"),
                note=r.get("note"),
                applies_to=r.get("applies_to", "all"),
                created_at=str(r["created_at"]) if r.get("created_at") else None,
            )
            for r in rows
        ]
        return CorrectionsListResponse(corrections=items)

    def _active_rows(self, tenant_id: UUID, client_id: Optional[UUID]) -> List[Dict[str, Any]]:
        q = (
            self.sb.table("context_corrections")
            .select("id, kind, from_term, to_term, note, applies_to, created_at, client_id")
            .eq("tenant_id", str(tenant_id))
            .eq("status", "active")
        )
        if client_id is not None:
            # tenant-wide (null client) + this client's own corrections
            q = q.or_(f"client_id.eq.{client_id},client_id.is.null")
        rows = q.execute().data or []
        return rows

    # ── apply at read time ───────────────────────────────────────────────────

    def _active_rows_safe(self, tenant_id: UUID, client_id: Optional[UUID]) -> List[Dict[str, Any]]:
        """Like _active_rows but never raises — corrections are a best-effort
        read-time enhancement and must not break the summary/retrieval path."""
        try:
            return self._active_rows(tenant_id, client_id)
        except Exception:
            logger.warning("context-correction fetch failed; skipping", exc_info=True)
            return []

    def apply_to_summary(self, text: Optional[str], tenant_id: UUID,
                         client_id: Optional[UUID]) -> Optional[str]:
        """Fold 'all'-scoped corrections into the aggregate context summary."""
        rows = [r for r in self._active_rows_safe(tenant_id, client_id)
                if r.get("applies_to") == "all"]
        return apply_corrections(text, rows) if rows else text

    def apply_to_documents(self, docs, tenant_id: UUID, client_id: Optional[UUID]):
        """Swap terms in retrieved Documents' page_content; doc-scoped corrections
        apply only to their listed document_ids. Returns Documents (rebuilt)."""
        rows = self._active_rows_safe(tenant_id, client_id)
        if not rows or not docs:
            return docs
        from langchain_core.documents import Document
        out = []
        for d in docs:
            doc_id = str((d.metadata or {}).get("document_id") or "")
            relevant = [
                r for r in rows
                if r.get("applies_to") == "all"
                or (isinstance(r.get("applies_to"), list) and doc_id in r["applies_to"])
            ]
            if relevant:
                out.append(Document(page_content=apply_corrections(d.page_content, relevant),
                                    metadata=d.metadata))
            else:
                out.append(d)
        return out

    # ── internals ────────────────────────────────────────────────────────────

    def _invalidate(self, tenant_id: UUID, client_id: Optional[UUID],
                    change: str, metadata: Dict[str, Any]) -> None:
        ms = MemoryStateService(self.sb)
        try:
            if client_id is not None:
                ms.bump_dual(tenant_id=tenant_id, client_id=client_id,
                             change_type="context_correction", metadata={"change": change, **metadata})
            else:
                ms.bump(tenant_id=tenant_id, client_id=None,
                        change_type="context_correction", metadata={"change": change, **metadata})
        except Exception:
            logger.warning("memory-state bump failed for correction", exc_info=True)

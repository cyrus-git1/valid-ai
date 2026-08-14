"""
src/services/spine_service.py
-----------------------------
Observation / Concept / App-entity "memory spine" logic, extracted from
spine_router so the router stays HTTP-only.

The router owns HTTP concerns (auth/tenant resolution via _check_tenant_match,
query params); this service owns the embedding orchestration, Supabase RPCs, and
response shaping. Methods take the already-resolved tenant_id and raise
HTTPException with identical status codes (500 on RPC failure, 502 on embedding
provider failure, 400 for missing embedding input).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from src.logging_config import get_logger
from src.models.api.concepts import (
    ConceptByStudyItem,
    ConceptCandidate,
    ConceptCreateRequest,
    ConceptCreateResponse,
    ConceptMergeRequest,
    ConceptMergeResponse,
    ConceptNearestRequest,
    ConceptNearestResponse,
    ConceptRelation,
    ConceptRelationsComputeRequest,
    ConceptRelationsComputeResponse,
    ConceptsByStudyResponse,
    GraduateRequest,
    GraduateResponse,
    LinkTagRequest,
    LinkTagResponse,
    MirrorTagRequest,
    MirrorTagResponse,
)
from src.models.api.app_entities import (
    AppEntityMatch,
    AppEntityNearestRequest,
    AppEntityNearestResponse,
    AppEntityUpsertRequest,
    AppEntityUpsertResponse,
)
from src.models.api.observations import (
    ObservationRecord,
    ObservationsByConceptResponse,
    ObservationsByIdsRequest,
    ObservationsRollupRequest,
    ObservationsRollupResponse,
    ObservationUpsertRequest,
    ObservationUpsertResponse,
)
from src.services.embedding_service import (
    embed_in_batches as _embed_in_batches,
    EMBED_MODEL as _EMBED_MODEL,
)

logger = get_logger(__name__)


class SpineService:
    """Wraps Supabase for observation / concept / app-entity spine operations."""

    def __init__(self, sb):
        self.sb = sb

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rpc_scalar(res_data: Any) -> Dict[str, Any]:
        """Unwrap a jsonb-returning RPC (dict, or single-element list)."""
        ret = res_data or {}
        if isinstance(ret, list):
            ret = ret[0] if ret else {}
        return ret if isinstance(ret, dict) else {}

    @staticmethod
    def _resolve_embedding(
        *, tenant_id: str, supplied: Optional[List[float]], text: Optional[str]
    ) -> Optional[List[float]]:
        if supplied:
            return supplied
        if text:
            try:
                return _embed_in_batches([text], tenant_id=tenant_id)[0]
            except HTTPException:
                raise
            except Exception as ex:
                logger.error("concept embedding failed (provider): %s", ex, exc_info=True)
                raise HTTPException(
                    status_code=502, detail=f"embedding provider unavailable: {ex}"
                )
        return None

    @staticmethod
    def _row_to_observation(row: Dict[str, Any]) -> ObservationRecord:
        return ObservationRecord(
            node_id=str(row.get("node_id", "")),
            observation_id=row.get("observation_id"),
            nl_text=row.get("nl_text"),
            value=row.get("value"),
            modality=row.get("modality"),
            signal_type=row.get("signal_type"),
            direction=row.get("direction"),
            prevalence=row.get("prevalence"),
            confidence=row.get("confidence"),
            reliability=row.get("reliability"),
            segment=row.get("segment"),
            occurred_at=row.get("occurred_at"),
            source=row.get("source"),
            study_id=str(row["study_id"]) if row.get("study_id") else None,
            evidence_chunk_ids=[str(c) for c in (row.get("evidence_chunk_ids") or [])],
            evidence=row.get("evidence"),
        )

    # ── observations ─────────────────────────────────────────────────────────

    def upsert_observation(self, tenant_id: str, body: ObservationUpsertRequest) -> ObservationUpsertResponse:
        client_id = str(body.client_id) if body.client_id else None
        sb = self.sb

        try:
            embedding = _embed_in_batches([body.nl_text], tenant_id=tenant_id)[0]
        except Exception as ex:
            logger.warning("observation embedding failed (storing null embedding): %s", ex)
            embedding = None

        properties: Dict[str, Any] = {"value": body.value}
        for field in ("modality", "signal_type", "direction", "occurred_at", "confidence"):
            val = getattr(body, field)
            if val is not None:
                properties[field] = val
        if body.prevalence is not None:
            properties["prevalence"] = body.prevalence.model_dump(exclude_none=True)
        if body.reliability is not None:
            properties["reliability"] = body.reliability.model_dump(exclude_none=True)
        if body.segment is not None:
            properties["segment"] = body.segment.model_dump(exclude_none=True)
        if body.source is not None:
            properties["source"] = body.source.model_dump(exclude_none=True)

        try:
            res = sb.rpc(
                "upsert_observation",
                {
                    "p_tenant_id":         tenant_id,
                    "p_client_id":         client_id,
                    "p_observation_id":    body.observation_id,
                    "p_nl_text":           body.nl_text,
                    "p_properties":        properties,
                    "p_embedding":         embedding,
                    "p_embedding_model":   _EMBED_MODEL,
                    "p_study_id":          str(body.study_id) if body.study_id else None,
                    "p_evidence_chunk_id": str(body.evidence_chunk_id) if body.evidence_chunk_id else None,
                    "p_concept_id":        str(body.concept_id) if body.concept_id else None,
                    "p_evidence":          body.evidence.model_dump(exclude_none=True) if body.evidence else None,
                    "p_evidence_chunk_ids": [str(c) for c in body.evidence_chunk_ids] if body.evidence_chunk_ids else None,
                },
            ).execute()
        except Exception as ex:
            logger.exception("observation upsert failed for %s", body.observation_id)
            raise HTTPException(status_code=500, detail=str(ex))

        ret = self._rpc_scalar(res.data)
        return ObservationUpsertResponse(
            observation_id=ret.get("observation_id", body.observation_id),
            node_id=str(ret.get("node_id", "")),
            evidence_linked=bool(ret.get("evidence_linked", False)),
            concept_linked=bool(ret.get("concept_linked", False)),
        )

    def observations_by_concept(self, tenant_id: str, concept_id, study_ids, modality,
                                persona, variant_key) -> ObservationsByConceptResponse:
        sb = self.sb
        try:
            res = sb.rpc(
                "observations_by_concept",
                {
                    "p_tenant_id":   tenant_id,
                    "p_concept_id":  str(concept_id),
                    "p_study_ids":   [str(s) for s in study_ids] if study_ids else None,
                    "p_modality":    modality,
                    "p_persona":     persona,
                    "p_variant_key": variant_key,
                },
            ).execute()
        except Exception as ex:
            logger.exception("observations_by_concept failed for concept %s", concept_id)
            raise HTTPException(status_code=500, detail=str(ex))

        rows = res.data or []
        if isinstance(rows, dict):
            rows = [rows]

        observations: List[ObservationRecord] = [self._row_to_observation(row) for row in rows]
        return ObservationsByConceptResponse(
            concept_id=str(concept_id),
            observations=observations,
        )

    def observations_by_ids(self, tenant_id: str, body: ObservationsByIdsRequest) -> Dict[str, ObservationRecord]:
        sb = self.sb
        try:
            res = sb.rpc(
                "observations_by_ids",
                {
                    "p_tenant_id":       tenant_id,
                    "p_observation_ids": body.ids or [],
                    "p_study_ids":       [str(s) for s in body.study_ids] if body.study_ids else None,
                },
            ).execute()
        except Exception as ex:
            logger.exception("observations_by_ids failed")
            raise HTTPException(status_code=500, detail=str(ex))

        rows = res.data or []
        if isinstance(rows, dict):
            rows = [rows]
        out: Dict[str, ObservationRecord] = {}
        for row in rows:
            rec = self._row_to_observation(row)
            if rec.observation_id:
                out[rec.observation_id] = rec
        return out

    def observations_rollup(self, tenant_id: str, body: ObservationsRollupRequest) -> ObservationsRollupResponse:
        sb = self.sb
        rng = body.range
        try:
            res = sb.rpc(
                "observations_rollup",
                {
                    "p_tenant_id":     tenant_id,
                    "p_study_ids":     [str(s) for s in body.study_ids] if body.study_ids else None,
                    "p_range_from":    rng.from_ if rng else None,
                    "p_range_to":      rng.to if rng else None,
                    "p_top_evidence":  body.top_evidence,
                },
            ).execute()
        except Exception as ex:
            logger.exception("observations_rollup failed")
            raise HTTPException(status_code=500, detail=str(ex))

        ret = self._rpc_scalar(res.data)
        return ObservationsRollupResponse(
            cube=ret.get("cube") or [],
            evidence=ret.get("evidence") or {},
        )

    # ── concepts ─────────────────────────────────────────────────────────────

    def concepts_nearest(self, tenant_id: str, body: ConceptNearestRequest) -> ConceptNearestResponse:
        sb = self.sb
        embedding = self._resolve_embedding(tenant_id=tenant_id, supplied=body.embedding, text=body.query_text)
        if embedding is None:
            raise HTTPException(status_code=400, detail="provide query_text or embedding")

        try:
            res = sb.rpc(
                "nearest_concepts",
                {
                    "p_tenant_id":       tenant_id,
                    "p_embedding":       embedding,
                    "p_hints":           body.hints or None,
                    "p_top_k":           body.top_k,
                    "p_embedding_model": _EMBED_MODEL,
                },
            ).execute()
        except Exception as ex:
            logger.exception("nearest_concepts failed")
            raise HTTPException(status_code=500, detail=str(ex))

        rows = res.data or []
        if isinstance(rows, dict):
            rows = [rows]

        candidates = [
            ConceptCandidate(
                id=str(row.get("id", "")),
                canonical_id=row.get("canonical_id"),
                canonical_label=row.get("canonical_label"),
                alias_set=row.get("alias_set"),
                merge_confidence=row.get("merge_confidence"),
                external_ref=row.get("external_ref"),
                similarity=row.get("similarity"),
                score=row.get("final_score"),
            )
            for row in rows
        ]
        return ConceptNearestResponse(candidates=candidates)

    def concepts_create(self, tenant_id: str, body: ConceptCreateRequest) -> ConceptCreateResponse:
        client_id = str(body.client_id) if body.client_id else None
        sb = self.sb

        embedding = self._resolve_embedding(
            tenant_id=tenant_id,
            supplied=body.embedding,
            text=body.embedding_text or body.canonical_label,
        )

        try:
            res = sb.rpc(
                "create_concept",
                {
                    "p_tenant_id":        tenant_id,
                    "p_client_id":        client_id,
                    "p_canonical_label":  body.canonical_label,
                    "p_aliases":          body.aliases or [],
                    "p_embedding":        embedding,
                    "p_embedding_model":  _EMBED_MODEL,
                    "p_merge_confidence": body.merge_confidence,
                    "p_canonical_id":     body.canonical_id,
                },
            ).execute()
        except Exception as ex:
            logger.exception("create_concept failed")
            raise HTTPException(status_code=500, detail=str(ex))

        ret = self._rpc_scalar(res.data)
        return ConceptCreateResponse(
            concept_id=str(ret.get("concept_id", "")),
            canonical_id=str(ret.get("canonical_id", "")),
            node_key=str(ret.get("node_key", "")),
            created=bool(ret.get("created", False)),
            redirected=bool(ret.get("redirected", False)),
        )

    def concepts_merge(self, tenant_id: str, body: ConceptMergeRequest) -> ConceptMergeResponse:
        sb = self.sb
        try:
            res = sb.rpc(
                "merge_concepts",
                {
                    "p_tenant_id":            tenant_id,
                    "p_surviving_concept_id": body.surviving_concept_id,
                    "p_source_concept_id":    body.source_concept_id,
                },
            ).execute()
        except Exception as ex:
            logger.exception("merge_concepts failed")
            raise HTTPException(status_code=500, detail=str(ex))

        ret = self._rpc_scalar(res.data)
        return ConceptMergeResponse(
            merged=bool(ret.get("merged", False)),
            rewired_count=int(ret.get("rewired_count", 0)),
            surviving_member_count=int(ret.get("surviving_member_count", 0)),
        )

    def concepts_by_study(self, tenant_id: str, study_ids, client_id) -> ConceptsByStudyResponse:
        sb = self.sb
        try:
            res = sb.rpc(
                "concepts_by_study",
                {
                    "p_tenant_id": tenant_id,
                    "p_study_ids": [str(s) for s in study_ids] if study_ids else None,
                    "p_client_id": str(client_id) if client_id else None,
                },
            ).execute()
        except Exception as ex:
            logger.exception("concepts_by_study failed")
            raise HTTPException(status_code=500, detail=str(ex))

        rows = res.data or []
        if isinstance(rows, dict):
            rows = [rows]
        concepts = [
            ConceptByStudyItem(
                concept_id=str(row.get("concept_id", "")),
                label=row.get("label"),
                external_ref=row.get("external_ref"),
            )
            for row in rows
        ]
        return ConceptsByStudyResponse(concepts=concepts)

    def mirror_tag(self, tenant_id: str, body: MirrorTagRequest) -> MirrorTagResponse:
        client_id = str(body.client_id) if body.client_id else None
        sb = self.sb
        text = body.embedding_text or " ".join(filter(None, [body.label, body.description]))
        embedding = self._resolve_embedding(tenant_id=tenant_id, supplied=body.embedding, text=text)
        try:
            res = sb.rpc(
                "mirror_tag_concept",
                {
                    "p_tenant_id":       tenant_id,
                    "p_client_id":       client_id,
                    "p_tag_id":          str(body.tag_id),
                    "p_label":           body.label,
                    "p_description":     body.description,
                    "p_embedding":       embedding,
                    "p_embedding_model": _EMBED_MODEL,
                },
            ).execute()
        except Exception as ex:
            logger.exception("mirror_tag_concept failed")
            raise HTTPException(status_code=500, detail=str(ex))
        ret = self._rpc_scalar(res.data)
        return MirrorTagResponse(
            concept_id=str(ret.get("concept_id", "")),
            external_ref=str(ret.get("external_ref", "")),
            node_key=str(ret.get("node_key", "")),
        )

    def link_tag(self, tenant_id: str, body: LinkTagRequest) -> LinkTagResponse:
        sb = self.sb
        try:
            res = sb.rpc(
                "link_concept_tag",
                {
                    "p_tenant_id":  tenant_id,
                    "p_concept_id": str(body.concept_id),
                    "p_tag_id":     str(body.tag_id),
                },
            ).execute()
        except Exception as ex:
            logger.exception("link_concept_tag failed")
            raise HTTPException(status_code=500, detail=str(ex))
        ret = self._rpc_scalar(res.data)
        return LinkTagResponse(
            linked=bool(ret.get("linked", False)),
            concept_id=str(ret.get("concept_id", body.concept_id)),
            external_ref=str(ret.get("external_ref", "")),
        )

    def graduate(self, tenant_id: str, body: GraduateRequest) -> GraduateResponse:
        client_id = str(body.client_id) if body.client_id else None
        sb = self.sb
        try:
            res = sb.rpc(
                "graduate_concept",
                {
                    "p_tenant_id":    tenant_id,
                    "p_concept_id":   str(body.concept_id),
                    "p_label":        body.label,
                    "p_description":  body.description,
                    "p_client_id":    client_id,
                    "p_cluster_id":   body.cluster_id,
                    "p_evidence_ids": body.evidence_ids or [],
                },
            ).execute()
        except Exception as ex:
            logger.exception("graduate_concept failed")
            raise HTTPException(status_code=500, detail=str(ex))
        ret = self._rpc_scalar(res.data) or {}
        return GraduateResponse(
            tag_id=str(ret.get("tag_id", "")),
            concept_id=str(ret.get("concept_id", body.concept_id)),
            external_ref=str(ret.get("external_ref", "")),
            created=bool(ret.get("created", False)),
        )

    def compute_concept_relations(self, tenant_id: str, body: ConceptRelationsComputeRequest) -> ConceptRelationsComputeResponse:
        sb = self.sb
        try:
            res = sb.rpc(
                "compute_concept_relations",
                {
                    "p_tenant_id":   tenant_id,
                    "p_study_ids":   [str(s) for s in body.study_ids] if body.study_ids else None,
                    "p_min_cooccur": body.min_cooccur,
                },
            ).execute()
        except Exception as ex:
            logger.exception("compute_concept_relations failed")
            raise HTTPException(status_code=500, detail=str(ex))
        ret = self._rpc_scalar(res.data)
        return ConceptRelationsComputeResponse(relations_written=int(ret.get("relations_written", 0)))

    def concept_relations(self, tenant_id: str, study_ids, rel_types) -> List[ConceptRelation]:
        sb = self.sb
        try:
            res = sb.rpc(
                "concept_relations",
                {
                    "p_tenant_id": tenant_id,
                    "p_study_ids": [str(s) for s in study_ids] if study_ids else None,
                    "p_rel_types": rel_types or ["supports", "contradicts"],
                },
            ).execute()
        except Exception as ex:
            logger.exception("concept_relations failed")
            raise HTTPException(status_code=500, detail=str(ex))
        rows = res.data or []
        if isinstance(rows, dict):
            rows = [rows]
        return [
            ConceptRelation(
                src_concept_id=str(row.get("src_concept_id", "")),
                dst_concept_id=str(row.get("dst_concept_id", "")),
                rel_type=str(row.get("rel_type", "")),
                weight=row.get("weight"),
            )
            for row in rows
        ]

    # ── app entities ─────────────────────────────────────────────────────────

    def upsert_app_entity(self, tenant_id: str, body: AppEntityUpsertRequest) -> AppEntityUpsertResponse:
        client_id = str(body.client_id) if body.client_id else None
        sb = self.sb

        node_key = f"app:{body.kind}:{body.external_id}"

        # Re-embed only when the stored text changed (name holds the embedded text).
        unchanged = False
        try:
            q = (
                sb.table("kg_nodes")
                .select("name")
                .eq("tenant_id", tenant_id)
                .eq("node_key", node_key)
                .eq("type", "AppEntity")
            )
            q = q.eq("client_id", client_id) if client_id else q.is_("client_id", "null")
            rows = (q.limit(1).execute().data) or []
            unchanged = bool(rows) and rows[0].get("name") == body.text
        except Exception as ex:
            logger.debug("app-entity text-change check failed (will re-embed): %s", ex)

        embedding = None
        if not unchanged:
            try:
                embedding = _embed_in_batches([body.text], tenant_id=tenant_id)[0]
            except Exception as ex:
                logger.warning("app-entity embedding failed (storing null embedding): %s", ex)
                embedding = None

        try:
            res = sb.rpc(
                "upsert_app_entity",
                {
                    "p_tenant_id":       tenant_id,
                    "p_client_id":       client_id,
                    "p_study_id":        str(body.study_id),
                    "p_kind":            body.kind,
                    "p_external_id":     body.external_id,
                    "p_text":            body.text,
                    "p_embedding":       embedding,
                    "p_embedding_model": _EMBED_MODEL,
                    "p_status":          body.status,
                },
            ).execute()
        except Exception as ex:
            logger.exception("upsert_app_entity failed for %s:%s", body.kind, body.external_id)
            raise HTTPException(status_code=500, detail=str(ex))

        ret = self._rpc_scalar(res.data)
        return AppEntityUpsertResponse(
            node_id=str(ret.get("node_id", "")),
            external_id=ret.get("external_id", body.external_id),
            kind=ret.get("kind", body.kind),
            created=bool(ret.get("created", False)),
        )

    def nearest_app_entities(self, tenant_id: str, body: AppEntityNearestRequest) -> AppEntityNearestResponse:
        sb = self.sb
        embedding = self._resolve_embedding(tenant_id=tenant_id, supplied=body.embedding, text=body.query_text)
        if embedding is None:
            raise HTTPException(status_code=400, detail="provide query_text or embedding")

        try:
            res = sb.rpc(
                "nearest_app_entities",
                {
                    "p_tenant_id":       tenant_id,
                    "p_embedding":       embedding,
                    "p_study_ids":       [str(s) for s in body.study_ids] if body.study_ids else None,
                    "p_kinds":           body.kinds or None,
                    "p_top_k":           body.top_k,
                    "p_embedding_model": _EMBED_MODEL,
                },
            ).execute()
        except Exception as ex:
            logger.exception("nearest_app_entities failed")
            raise HTTPException(status_code=500, detail=str(ex))

        rows = res.data or []
        if isinstance(rows, dict):
            rows = [rows]

        matches = [
            AppEntityMatch(
                external_id=str(row.get("external_id", "")),
                kind=str(row.get("kind", "")),
                study_id=str(row["study_id"]) if row.get("study_id") else None,
                similarity=float(row.get("similarity") or 0.0),
            )
            for row in rows
        ]
        return AppEntityNearestResponse(matches=matches)

    # ── canvas blocks ────────────────────────────────────────────────────────

    def upsert_canvas_block(self, tenant_id: str, body: "CanvasBlockUpsertRequest") -> "CanvasBlockUpsertResponse":
        from src.models.api.canvas_blocks import CanvasBlockUpsertResponse

        client_id = str(body.client_id) if body.client_id else None
        study_id = str(body.study_id) if body.study_id else None
        sb = self.sb

        node_key = (
            f"canvas:org:{body.block_key}" if study_id is None
            else f"canvas:study:{study_id}:{body.block_key}"
        )

        # Re-embed only when the statement (embedded text = name) changed.
        embedding = None
        if body.statement:
            unchanged = False
            try:
                q = (
                    sb.table("kg_nodes")
                    .select("name")
                    .eq("tenant_id", tenant_id)
                    .eq("node_key", node_key)
                    .eq("type", "CanvasBlock")
                )
                q = q.eq("client_id", client_id) if client_id else q.is_("client_id", "null")
                rows = (q.limit(1).execute().data) or []
                unchanged = bool(rows) and rows[0].get("name") == body.statement
            except Exception as ex:
                logger.debug("canvas-block text-change check failed (will re-embed): %s", ex)
            if not unchanged:
                try:
                    embedding = _embed_in_batches([body.statement], tenant_id=tenant_id)[0]
                except Exception as ex:
                    logger.warning("canvas-block embedding failed (storing null embedding): %s", ex)
                    embedding = None

        try:
            res = sb.rpc(
                "upsert_canvas_block",
                {
                    "p_tenant_id":       tenant_id,
                    "p_client_id":       client_id,
                    "p_study_id":        study_id,
                    "p_block_key":       body.block_key,
                    "p_statement":       body.statement,
                    "p_stated":          body.stated,
                    "p_evidenced":       body.evidenced,
                    "p_source":          body.source,
                    "p_status":          body.status,
                    "p_confidence":      body.confidence,
                    "p_pinned":          body.pinned,
                    "p_divergence":      body.divergence,
                    "p_evidence_refs":   body.evidence_refs,
                    "p_embedding":       embedding,
                    "p_embedding_model": _EMBED_MODEL,
                },
            ).execute()
        except Exception as ex:
            logger.exception("upsert_canvas_block failed for %s (%s)", body.block_key, node_key)
            raise HTTPException(status_code=500, detail=str(ex))

        ret = self._rpc_scalar(res.data)
        return CanvasBlockUpsertResponse(
            node_id=str(ret.get("node_id", "")),
            block_key=ret.get("block_key", body.block_key),
            scope=ret.get("scope", "org" if study_id is None else "study"),
            created=bool(ret.get("created", False)),
            pinned=bool(ret.get("pinned", False)),
            divergence=bool(ret.get("divergence", False)),
        )

    def canvas_by_scope(self, tenant_id: str, body: "CanvasByScopeRequest") -> "CanvasByScopeResponse":
        from src.models.api.canvas_blocks import CanvasBlockRow, CanvasByScopeResponse

        client_id = str(body.client_id) if body.client_id else None
        study_id = str(body.study_id) if body.study_id else None

        try:
            res = self.sb.rpc(
                "canvas_by_scope",
                {"p_tenant_id": tenant_id, "p_client_id": client_id, "p_study_id": study_id},
            ).execute()
        except Exception as ex:
            logger.exception("canvas_by_scope failed")
            raise HTTPException(status_code=500, detail=str(ex))

        rows = res.data or []
        if isinstance(rows, dict):
            rows = [rows]

        blocks = [
            CanvasBlockRow(
                node_id=str(row.get("node_id", "")),
                block_key=str(row.get("block_key", "")),
                statement=row.get("statement"),
                stated=row.get("stated"),
                evidenced=row.get("evidenced"),
                source=row.get("source"),
                status=row.get("status"),
                confidence=row.get("confidence"),
                pinned=bool(row.get("pinned", False)),
                divergence=bool(row.get("divergence", False)),
                study_id=str(row["study_id"]) if row.get("study_id") else None,
                evidence_refs=row.get("evidence_refs") or [],
            )
            for row in rows
        ]
        return CanvasByScopeResponse(blocks=blocks)

    # ── hypotheses ───────────────────────────────────────────────────────────

    def upsert_hypothesis(self, tenant_id: str, body: "HypothesisUpsertRequest") -> "HypothesisUpsertResponse":
        from src.models.api.hypotheses import HypothesisUpsertResponse

        client_id = str(body.client_id) if body.client_id else None
        study_id = str(body.study_id) if body.study_id else None
        sb = self.sb

        node_key = (
            f"hyp:org:{body.external_id}" if study_id is None
            else f"hyp:study:{study_id}:{body.external_id}"
        )

        # Re-embed only when the claim text (embedded = name) changed.
        embedding = None
        if body.text:
            unchanged = False
            try:
                q = (
                    sb.table("kg_nodes")
                    .select("name")
                    .eq("tenant_id", tenant_id)
                    .eq("node_key", node_key)
                    .eq("type", "Hypothesis")
                )
                q = q.eq("client_id", client_id) if client_id else q.is_("client_id", "null")
                rows = (q.limit(1).execute().data) or []
                unchanged = bool(rows) and rows[0].get("name") == body.text
            except Exception as ex:
                logger.debug("hypothesis text-change check failed (will re-embed): %s", ex)
            if not unchanged:
                try:
                    embedding = _embed_in_batches([body.text], tenant_id=tenant_id)[0]
                except Exception as ex:
                    logger.warning("hypothesis embedding failed (storing null embedding): %s", ex)
                    embedding = None

        try:
            res = sb.rpc(
                "upsert_hypothesis",
                {
                    "p_tenant_id":       tenant_id,
                    "p_client_id":       client_id,
                    "p_study_id":        study_id,
                    "p_external_id":     body.external_id,
                    "p_text":            body.text,
                    "p_block_key":       body.block_key,
                    "p_status":          body.status,
                    "p_confidence":      body.confidence,
                    "p_reasoning":       body.reasoning,
                    "p_theme_ids":       body.theme_ids,
                    "p_evidence_refs":   body.evidence_refs,
                    "p_embedding":       embedding,
                    "p_embedding_model": _EMBED_MODEL,
                    "p_origin":          body.origin,
                },
            ).execute()
        except Exception as ex:
            logger.exception("upsert_hypothesis failed for %s", body.external_id)
            raise HTTPException(status_code=500, detail=str(ex))

        ret = self._rpc_scalar(res.data)
        return HypothesisUpsertResponse(
            node_id=str(ret.get("node_id", "")),
            external_id=ret.get("external_id", body.external_id),
            scope=ret.get("scope", "org" if study_id is None else "study"),
            created=bool(ret.get("created", False)),
        )

    def hypotheses_by_scope(self, tenant_id: str, body: "HypothesesByScopeRequest") -> "HypothesesByScopeResponse":
        from src.models.api.hypotheses import HypothesesByScopeResponse, HypothesisRow

        client_id = str(body.client_id) if body.client_id else None
        study_id = str(body.study_id) if body.study_id else None

        try:
            res = self.sb.rpc(
                "hypotheses_by_scope",
                {"p_tenant_id": tenant_id, "p_client_id": client_id, "p_study_id": study_id},
            ).execute()
        except Exception as ex:
            logger.exception("hypotheses_by_scope failed")
            raise HTTPException(status_code=500, detail=str(ex))

        rows = res.data or []
        if isinstance(rows, dict):
            rows = [rows]

        hyps = [
            HypothesisRow(
                node_id=str(row.get("node_id", "")),
                external_id=str(row.get("external_id", "")),
                text=row.get("text"),
                block_key=row.get("block_key"),
                status=row.get("status"),
                confidence=row.get("confidence"),
                reasoning=row.get("reasoning"),
                origin=row.get("origin"),
                theme_ids=row.get("theme_ids") or [],
                evidence_refs=row.get("evidence_refs") or [],
                study_id=str(row["study_id"]) if row.get("study_id") else None,
                seen_count=row.get("seen_count"),
                created_at=str(row["created_at"]) if row.get("created_at") else None,
                updated_at=str(row["updated_at"]) if row.get("updated_at") else None,
            )
            for row in rows
        ]
        return HypothesesByScopeResponse(hypotheses=hyps)

"""
src/services/entity_service.py
------------------------------
Entity extraction-cache + entity KG (nodes/edges) logic, extracted from
entities_router so the router stays HTTP-only.

The router owns HTTP concerns (auth/tenant checks, input length guards, status
codes for missing bodies); this service owns the Supabase queries, RPC calls,
and embedding orchestration. Methods raise HTTPException for not-found / bad
-request / upstream-failure cases so behavior is identical to the pre-extraction
router.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from src.logging_config import get_logger
from src.models.api.entities import (
    CacheUpsertRequest,
    CacheUpsertResponse,
    CachedEntitiesResponse,
    EntityInput,
    EntityMergeRequest,
    EntityMergeResponse,
    EntityRedirect,
    EntityRollup,
    EntityRollupResponse,
    EntitySearchResponse,
    EntitySearchResultItem,
    EntitySentiment,
    EntityUpsertRequest,
    EntityUpsertResponse,
    MentionItem,
)
from src.routers.ingest_router import _embed_in_batches, _EMBED_MODEL

logger = get_logger(__name__)

# Cache TTL — matches migration 36 default (30 days). Held here too so the
# API layer can compute expires_at without round-tripping to Postgres defaults.
_CACHE_TTL = timedelta(days=30)


class EntityService:
    """Wraps Supabase for entity extraction-cache + entity KG operations."""

    def __init__(self, sb):
        self.sb = sb

    # ── payload helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _entity_payload_to_props(e: EntityInput) -> Dict[str, Any]:
        """Pack an EntityInput into the jsonb 'properties' shape on the entity node."""
        props: Dict[str, Any] = {
            "source_origin": e.source,
            "confidence":    e.confidence,
        }
        if e.sentiment:
            props["sentiment_aggregate"] = e.sentiment.dict(exclude_none=True)
        return props

    @staticmethod
    def _mention_payload(e: EntityInput, extractor_version: str,
                         survey_id) -> Dict[str, Any]:
        """Pack per-mention edge payload."""
        p: Dict[str, Any] = {
            "count":              e.count,
            "source":             e.source,
            "confidence":         e.confidence,
            "cue_indices":        e.cue_indices,
            "sample_contexts":    e.sample_contexts[:3],   # cap per the brief
            "extractor_version":  extractor_version,
        }
        if e.sentiment:
            p["sentiment"] = e.sentiment.dict(exclude_none=True)
        if survey_id:
            p["survey_id"] = str(survey_id)
        return p

    def _resolve_canonical(self, tenant_id: str, canonical_id: str,
                           label: Optional[str]) -> List[Dict[str, Any]]:
        """Find Entity kg_nodes for a canonical_id (+ optional label)."""
        # GIN expression index on properties->>'canonical_id' makes this fast
        q = (
            self.sb.table("kg_nodes")
            .select("id, node_key, name, description, properties, status, embedding_model, last_seen_at, created_at, updated_at")
            .eq("tenant_id", tenant_id)
            .eq("type", "Entity")
            .filter("properties->>canonical_id", "eq", canonical_id)
        )
        res = q.execute()
        rows = res.data or []
        if label:
            rows = [r for r in rows if (r.get("properties") or {}).get("label") == label]
        return rows

    # ── Tier 1: extraction cache ─────────────────────────────────────────────

    def get_cached(self, tenant_id: str, transcript_sha256: str,
                   prompt_version: Optional[str]) -> CachedEntitiesResponse:
        """Fetch cached LLM extraction output. 404 on miss or expired."""
        try:
            q = (
                self.sb.table("entity_extraction_cache")
                .select("transcript_sha256, entities, model_name, prompt_version, created_at, expires_at, tenant_id")
                .eq("transcript_sha256", transcript_sha256)
                .eq("tenant_id", tenant_id)
                .limit(1)
            )
            res = q.execute()
        except Exception as e:
            logger.exception("entity_cache read failed")
            raise HTTPException(status_code=500, detail=str(e))

        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="cache miss")

        row = rows[0]

        # Expiry check (defensive — daily cleanup may not have run yet)
        expires_at = row["expires_at"]
        if expires_at:
            # Parse ISO-ish timestamp from supabase
            try:
                exp_dt = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
                if exp_dt < datetime.now(timezone.utc):
                    raise HTTPException(status_code=404, detail="cache expired")
            except (ValueError, TypeError):
                pass  # If unparseable, fall through and serve it

        # Optional prompt_version filter: cache invalidates if the caller's
        # expected prompt version doesn't match what was stored. Lets the agent
        # ship prompt changes without stale data leaking.
        if prompt_version and row.get("prompt_version") != prompt_version:
            raise HTTPException(status_code=404, detail="prompt_version mismatch")

        return CachedEntitiesResponse(
            transcript_sha256=row["transcript_sha256"],
            entities=row["entities"] or [],
            model_name=row["model_name"],
            prompt_version=row["prompt_version"],
            created_at=str(row["created_at"]),
            expires_at=str(row["expires_at"]),
        )

    def upsert_cache(self, tenant_id: str, body: CacheUpsertRequest) -> CacheUpsertResponse:
        """Upsert cached LLM extraction output. Idempotent on transcript_sha256."""
        expires_at = (datetime.now(timezone.utc) + _CACHE_TTL).isoformat()
        payload: Dict[str, Any] = {
            "transcript_sha256": body.transcript_sha256,
            "tenant_id": tenant_id,
            "client_id": str(body.client_id) if body.client_id else None,
            "entities": body.entities,
            "model_name": body.model_name,
            "prompt_version": body.prompt_version,
            "expires_at": expires_at,
        }

        try:
            self.sb.table("entity_extraction_cache").upsert(
                payload, on_conflict="transcript_sha256"
            ).execute()
        except Exception as e:
            logger.exception("entity_cache upsert failed")
            raise HTTPException(status_code=500, detail=str(e))

        return CacheUpsertResponse(transcript_sha256=body.transcript_sha256)

    # ── Tier 2: entity KG upsert + merge ─────────────────────────────────────

    def upsert_mentions(self, tenant_id: str, body: EntityUpsertRequest) -> EntityUpsertResponse:
        """Upsert a batch of entities + their MENTIONS edges for one session."""
        client_id = str(body.client_id) if body.client_id else None

        # Pre-compute embeddings only for entities we've never seen.
        # The RPC preserves the old embedding when p_embedding is null, so we
        # can pass null for entities that already have one.
        existing_keys: Dict[str, bool] = {}
        for e in body.entities:
            node_key = f"entity:{e.canonical_id}:{e.label}"
            try:
                r = (
                    self.sb.table("kg_nodes")
                    .select("id, embedding")
                    .eq("tenant_id", tenant_id)
                    .eq("node_key", node_key)
                    .limit(1)
                    .execute()
                )
                existing_keys[node_key] = bool(r.data and r.data[0].get("embedding") is not None)
            except Exception:
                existing_keys[node_key] = False

        new_entities = [e for e in body.entities
                        if not existing_keys.get(f"entity:{e.canonical_id}:{e.label}", False)]

        embeddings: Dict[str, Optional[List[float]]] = {}
        if new_entities:
            embed_input = [f"{e.label}: {e.canonical_text}" for e in new_entities]
            try:
                vecs = _embed_in_batches(embed_input, tenant_id=tenant_id)
                for e, v in zip(new_entities, vecs):
                    embeddings[f"entity:{e.canonical_id}:{e.label}"] = v
            except Exception as ex:
                logger.warning("entity embedding failed (proceeding with null embeddings): %s", ex)

        upserted_entities = 0
        upserted_mentions = 0
        redirects: List[EntityRedirect] = []
        warnings: List[str] = []

        for e in body.entities:
            node_key = f"entity:{e.canonical_id}:{e.label}"
            emb = embeddings.get(node_key)  # None if entity already had embedding
            try:
                res = self.sb.rpc(
                    "upsert_entity_with_mention",
                    {
                        "p_tenant_id":           tenant_id,
                        "p_client_id":           client_id,
                        "p_document_id":         str(body.document_id),
                        "p_canonical_id":        e.canonical_id,
                        "p_canonical_text":      e.canonical_text,
                        "p_label":               e.label,
                        "p_source_origin":       e.source,
                        "p_embedding":           emb,
                        "p_entity_properties":   self._entity_payload_to_props(e),
                        "p_mention_properties":  self._mention_payload(e, body.extractor_version, body.survey_id),
                    },
                ).execute()
                ret = res.data or {}
                if isinstance(ret, list) and ret:
                    ret = ret[0]

                upserted_entities += 1
                upserted_mentions += 1

                if ret.get("redirected_from"):
                    redirects.append(EntityRedirect(
                        canonical_id=e.canonical_id,
                        surviving_id=str(ret.get("entity_id")),
                    ))
            except Exception as ex:
                logger.exception("entity upsert failed for %s:%s", e.canonical_id, e.label)
                warnings.append(f"{e.canonical_id}:{e.label} → {str(ex)[:120]}")

        return EntityUpsertResponse(
            upserted_entities=upserted_entities,
            upserted_mentions=upserted_mentions,
            merged_into_existing=redirects,
            warnings=warnings,
        )

    def merge(self, tenant_id: str, canonical_id: str, label: Optional[str],
              body: EntityMergeRequest) -> EntityMergeResponse:
        """Merge (canonical_id, label) into the survivor. Idempotent."""
        # Resolve source entity(ies)
        source_rows = self._resolve_canonical(tenant_id, canonical_id, label)
        if not source_rows:
            raise HTTPException(status_code=404, detail=f"source entity {canonical_id} not found")
        if len(source_rows) > 1 and not label:
            raise HTTPException(
                status_code=400,
                detail="canonical_id matches multiple labels; pass ?label= to disambiguate",
            )

        surviving_rows = self._resolve_canonical(
            tenant_id, body.surviving_canonical_id, body.surviving_label,
        )
        if not surviving_rows:
            raise HTTPException(status_code=404, detail="surviving entity not found")

        try:
            res = self.sb.rpc("merge_entities", {
                "p_tenant_id":            tenant_id,
                "p_source_entity_id":     source_rows[0]["id"],
                "p_surviving_entity_id":  surviving_rows[0]["id"],
            }).execute()
            ret = res.data or {}
            if isinstance(ret, list) and ret:
                ret = ret[0]
            return EntityMergeResponse(
                merged=bool(ret.get("merged", True)),
                redirect_count=int(ret.get("redirect_count", 0)),
            )
        except HTTPException:
            raise
        except Exception as ex:
            logger.exception("entity merge failed")
            raise HTTPException(status_code=500, detail=str(ex))

    # ── Tier 3: search + rollup ──────────────────────────────────────────────

    def search(self, tenant_id: str, client_id, q: Optional[str], label: Optional[str],
               valence: Optional[str], min_mentions: int, top_k: int) -> EntitySearchResponse:
        """List or semantic-search tenant entities. Tenant-scoped."""
        # Semantic path: embed the query and rank via existing search_kg_nodes RPC.
        if q:
            try:
                qvec = _embed_in_batches([q], tenant_id=tenant_id)[0]
                res = self.sb.rpc("search_kg_nodes", {
                    "p_tenant_id":       tenant_id,
                    "p_client_id":       str(client_id) if client_id else None,
                    "p_embedding":       qvec,
                    "p_top_k":           top_k,
                    "p_types":           ["Entity"],
                    "p_embedding_model": _EMBED_MODEL,
                }).execute()
                items: List[EntitySearchResultItem] = []
                for row in (res.data or []):
                    props = row.get("properties") or {}
                    if label and props.get("label") != label:
                        continue
                    tm = int(props.get("total_mentions") or 0)
                    if tm < min_mentions:
                        continue
                    sentaggr = props.get("sentiment_aggregate") or {}
                    if valence:
                        vd = (sentaggr.get("valence_distribution") or {})
                        if int(vd.get(valence) or 0) == 0:
                            continue
                    items.append(EntitySearchResultItem(
                        canonical_id=props.get("canonical_id", ""),
                        canonical_text=props.get("canonical_text") or row.get("name", ""),
                        label=props.get("label", ""),
                        total_mentions=tm,
                        sentiment_aggregate=sentaggr or None,
                        score=float(row.get("final_score") or row.get("similarity") or 0.0),
                    ))
                return EntitySearchResponse(entities=items[:top_k])
            except HTTPException:
                raise
            except Exception as ex:
                logger.exception("entity semantic search failed")
                raise HTTPException(status_code=500, detail=str(ex))

        # List path: filter by tenant + Entity type, paginate by mentions.
        try:
            q1 = (
                self.sb.table("kg_nodes")
                .select("id, name, properties, status")
                .eq("tenant_id", tenant_id)
                .eq("type", "Entity")
                .eq("status", "active")
                .limit(top_k * 5)   # over-fetch then filter in Python
            )
            if client_id:
                q1 = q1.eq("client_id", str(client_id))
            res = q1.execute()

            items: List[EntitySearchResultItem] = []
            for row in (res.data or []):
                props = row.get("properties") or {}
                if label and props.get("label") != label:
                    continue
                tm = int(props.get("total_mentions") or 0)
                if tm < min_mentions:
                    continue
                sentaggr = props.get("sentiment_aggregate") or {}
                if valence:
                    vd = (sentaggr.get("valence_distribution") or {})
                    if int(vd.get(valence) or 0) == 0:
                        continue
                items.append(EntitySearchResultItem(
                    canonical_id=props.get("canonical_id", ""),
                    canonical_text=props.get("canonical_text") or row.get("name", ""),
                    label=props.get("label", ""),
                    total_mentions=tm,
                    sentiment_aggregate=sentaggr or None,
                ))
            items.sort(key=lambda i: i.total_mentions, reverse=True)
            return EntitySearchResponse(entities=items[:top_k])
        except HTTPException:
            raise
        except Exception as ex:
            logger.exception("entity list failed")
            raise HTTPException(status_code=500, detail=str(ex))

    def rollup(self, tenant_id: str, canonical_id: str, client_id,
               label: Optional[str]) -> EntityRollupResponse:
        """Full entity profile + all MENTIONS."""
        rows = self._resolve_canonical(tenant_id, canonical_id, label)
        if not rows:
            raise HTTPException(status_code=404, detail=f"entity {canonical_id} not found")
        if len(rows) > 1 and not label:
            raise HTTPException(
                status_code=400,
                detail="canonical_id matches multiple labels; pass label= to disambiguate",
            )

        row = rows[0]
        props = row.get("properties") or {}

        entity = EntityRollup(
            canonical_id=props.get("canonical_id", canonical_id),
            canonical_text=props.get("canonical_text") or row.get("name", ""),
            label=props.get("label", label or ""),
            total_mentions=int(props.get("total_mentions") or 0),
            distinct_session_count=int(props.get("distinct_session_count") or 0),
            sentiment_aggregate=props.get("sentiment_aggregate"),
            first_seen_at=props.get("first_seen_at"),
            last_seen_at=props.get("last_seen_at"),
            status=row.get("status") or "active",
            merged_into=props.get("merged_into"),
        )

        # Pull MENTIONS edges → join through src kg_node (Document type) to get document_id
        try:
            edges_res = (
                self.sb.table("kg_edges")
                .select("src_id, properties, last_seen_at")
                .eq("tenant_id", tenant_id)
                .eq("dst_id", row["id"])
                .eq("rel_type", "mentions")
                .eq("is_active", True)
                .order("last_seen_at", desc=True)
                .limit(500)
                .execute()
            )
            edge_rows = edges_res.data or []
            src_ids = list({e["src_id"] for e in edge_rows})
            # Bulk-fetch the source Document nodes to map node_id → document_id
            doc_map: Dict[str, str] = {}
            if src_ids:
                docs_res = (
                    self.sb.table("kg_nodes")
                    .select("id, properties")
                    .in_("id", src_ids)
                    .execute()
                )
                for d in (docs_res.data or []):
                    doc_id = (d.get("properties") or {}).get("document_id")
                    if doc_id:
                        doc_map[d["id"]] = doc_id

            mentions: List[MentionItem] = []
            for e in edge_rows:
                ep = e.get("properties") or {}
                sentiment = ep.get("sentiment")
                mentions.append(MentionItem(
                    document_id=doc_map.get(e["src_id"], ""),
                    survey_id=ep.get("survey_id"),
                    count=int(ep.get("count") or 1),
                    sentiment=EntitySentiment(**sentiment) if sentiment else None,
                    sample_contexts=list(ep.get("sample_contexts") or [])[:3],
                    extracted_at=ep.get("extracted_at"),
                ))
        except Exception:
            logger.exception("entity rollup mentions fetch failed")
            mentions = []

        return EntityRollupResponse(entity=entity, mentions=mentions)

"""
/ingest router (data plane)
----------------------------
Receives pre-processed documents from the agent service and handles storage.

POST /ingest/processed      — Store a pre-processed file (chunks + entities)
POST /ingest/processed-web  — Store a pre-processed web scrape (chunks + entities)

The agent service does: parsing, chunking, language filtering, NER extraction.
This service does: file storage, document rows, embedding, chunk storage,
KG node/edge creation, entity linking.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from openai import OpenAI
from pydantic import BaseModel, Field

from src.config.plan_limits import get_limit
from src.services.audit_service import AuditService
from src.services.memory_state_service import MemoryStateService
from src.services.tenant_plan_service import (
    EmbeddingQuotaService,
    TenantPlanService,
    estimate_tokens,
)
from src.supabase.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

_EMBED_MODEL = "text-embedding-3-small"
_EMBED_BATCH_SIZE = 64
_PDF_BUCKET = "pdf"
_VALID_DOCS_BUCKET = "valid-docs"


def _assert_chunks_within_plan(sb, tenant_id, chunk_count: int, *, scopes=None) -> str:
    """Reject ingest if it exceeds the tenant plan's chunk-count cap.

    Admin scope bypasses. Returns the plan name for logging.
    """
    if scopes and "admin" in scopes:
        return "admin"
    plan = TenantPlanService(sb).get_plan(str(tenant_id))
    cap = get_limit(plan, "max_chunks_per_ingest")
    if chunk_count > cap:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "payload.too_many_chunks",
                "plan": plan,
                "chunk_count": chunk_count,
                "plan_max": cap,
                "message": (
                    f"Ingest has {chunk_count} chunks; plan '{plan}' allows "
                    f"up to {cap} chunks per request. Upgrade your plan or "
                    f"split into multiple ingest calls."
                ),
            },
        )
    return plan
_SIMILARITY_THRESHOLD = 0.82
# Step 4 — write-time entity reconciliation. Conservative by design: only route a
# mention onto an existing near-duplicate entity when cosine >= this floor;
# otherwise mint a new node. A wrong merge silently corrupts the graph and is
# worse than a duplicate, so keep this high and tune down against real data.
_ENTITY_RECONCILE_THRESHOLD = 0.93
_MAX_EDGES_PER_CHUNK = 5
_ADJACENCY_WINDOW = 1
_PAGE_WINDOW = 2
_LEXICAL_TOKEN_LIMIT = 12
_MAX_TOKEN_DOC_FREQ = 20
_MAX_CANDIDATES_PER_CHUNK = 32


# -- Models --


class ChunkItem(BaseModel):
    text: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    token_count: Optional[int] = None


class EntityItem(BaseModel):
    name: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class _ProvenanceMixin(BaseModel):
    """Optional provenance fields. Agents may pass these to associate an
    ingest with a specific actor / source app / request for audit + history.
    """
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None  # 'user' | 'service' | 'agent' | 'anonymous'
    source_app: Optional[str] = None
    request_id: Optional[str] = None
    previous_version_id: Optional[str] = None


class ProcessedDocumentRequest(_ProvenanceMixin):
    tenant_id: UUID
    client_id: UUID
    file_name: str
    file_bytes_b64: str
    source_type: str
    title: str
    source_timestamp: Optional[datetime] = None
    is_pinned: bool = False
    is_canonical: bool = False
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[ChunkItem] = Field(default_factory=list)
    entities: List[EntityItem] = Field(default_factory=list)


class ProcessedWebRequest(_ProvenanceMixin):
    tenant_id: UUID
    client_id: UUID
    url: str
    title: str
    source_timestamp: Optional[datetime] = None
    is_pinned: bool = False
    is_canonical: bool = False
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[ChunkItem] = Field(default_factory=list)
    entities: List[EntityItem] = Field(default_factory=list)


class IngestProcessedResponse(BaseModel):
    document_id: str
    source_uri: str
    chunks_upserted: int
    entities_linked: int = 0
    warnings: List[str] = Field(default_factory=list)


class SummaryIngestRequest(_ProvenanceMixin):
    """Thin, synchronous ingest path for LLM-generated summary chunks.

    Produces exactly one document + one chunk + one KG node. Skips the
    O(n^2) semantic-linking phase because there's only one chunk. Entity
    mentions (if provided) use weight=0.5 so source evidence always
    outranks summary evidence during graph expansion.
    """
    tenant_id: UUID
    client_id: UUID
    source_type: str = Field(
        description="ContextSummary | DocumentSummary | TopicSummary",
    )
    summary_text: str
    topics: List[str] = Field(default_factory=list)
    # Scope-identity: exactly one of document_id or topic may be set depending
    # on source_type. For ContextSummary, both are None.
    document_id: Optional[str] = None
    topic: Optional[str] = None
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_stats: Dict[str, Any] = Field(default_factory=dict)
    memory_version_at_generation: Optional[int] = None
    entities: List[EntityItem] = Field(default_factory=list)
    # Optional arbitrary metadata from the agent (model, prompt version, etc.)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class SummaryIngestResponse(BaseModel):
    document_id: str
    chunk_id: str
    node_id: str
    superseded_document_id: Optional[str] = None
    memory_version: int


# -- Embedding helpers --


def _embed_texts(texts: List[str], *, tenant_id=None) -> List[List[float]]:
    """Call OpenAI embeddings. If `tenant_id` is set, charge tokens against
    that tenant's daily quota (and reconcile against actual usage).
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    quota_svc: Optional[EmbeddingQuotaService] = None
    estimate = 0
    if tenant_id:
        quota_svc = EmbeddingQuotaService(TenantPlanService(get_supabase()))
        estimate = estimate_tokens(texts)
        quota_svc.check_and_consume(str(tenant_id), estimate)

    resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)

    # Reconcile actual vs estimate (OpenAI reports prompt_tokens)
    if quota_svc and tenant_id:
        try:
            actual = int(getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0)
            if actual:
                quota_svc.reconcile_actual(str(tenant_id), estimate, actual)
        except Exception:
            pass

    return [d.embedding for d in resp.data]


def _embed_in_batches(texts: List[str], *, tenant_id=None) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        out.extend(_embed_texts(texts[i:i + _EMBED_BATCH_SIZE], tenant_id=tenant_id))
    return out


# -- Storage helpers --


def _sanitize_storage_key(name: str) -> str:
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = re.sub(r"[\s\[\]\(\),!@#$%^&+={}|;:'\"<>?]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return f"{stem}.{ext}" if ext else stem


def _upload_to_bucket(sb, *, tenant_id: UUID, client_id: UUID, file_bytes: bytes, file_name: str) -> str:
    safe_name = _sanitize_storage_key(file_name.lstrip("/"))
    path = f"{tenant_id}/{client_id}/{safe_name}"
    sb.storage.from_(_PDF_BUCKET).upload(path, file_bytes, file_options={"upsert": "true"})
    return f"bucket:{_PDF_BUCKET}/{path}"


def _set_audit_context(sb, *, actor_id: Optional[str], request_id: Optional[str], reason: Optional[str] = None) -> None:
    """Set per-transaction GUCs that the audit triggers read.

    Best-effort: failures here must never block the underlying mutation.
    """
    try:
        sb.rpc("set_audit_context", {
            "p_actor_id": actor_id,
            "p_request_id": request_id,
            "p_reason": reason,
        }).execute()
    except Exception as e:
        logger.debug("set_audit_context failed: %s", e)


def _upsert_document(
    sb,
    *,
    tenant_id,
    client_id,
    source_type,
    source_uri,
    title,
    metadata,
    source_timestamp: Optional[datetime] = None,
    is_pinned: bool = False,
    is_canonical: bool = False,
    status: str = "active",
    actor_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    source_app: Optional[str] = None,
    request_id: Optional[str] = None,
    ingest_job_id: Optional[str] = None,
    previous_version_id: Optional[str] = None,
) -> UUID:
    existing = (
        sb.table("documents").select("id")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .eq("source_uri", source_uri)
        .limit(1).execute()
    )
    base = {
        "source_type": source_type,
        "title": title,
        "metadata": metadata or {},
        "source_timestamp": source_timestamp.isoformat() if source_timestamp else None,
        "is_pinned": is_pinned,
        "is_canonical": is_canonical,
        "status": status,
    }
    # Provenance is INSERT-only. Once set, the columns are frozen — they
    # describe who created this version of the row. Subsequent mutations
    # (status changes, metadata edits) are captured in audit_log with their
    # own actor via set_audit_context().
    prov = {
        "actor_id": actor_id,
        "actor_type": actor_type,
        "source_app": source_app,
        "request_id": request_id,
        "ingest_job_id": ingest_job_id,
        "previous_version_id": previous_version_id,
    }
    prov = {k: v for k, v in prov.items() if v is not None}

    if existing.data:
        # UPDATE path: never touch provenance columns. Just refresh the
        # editable fields. The audit_log trigger will capture WHO made this
        # update via the per-request audit GUC.
        doc_id = existing.data[0]["id"]
        sb.table("documents").update(base).eq("id", doc_id).eq(
            "tenant_id", str(tenant_id)
        ).eq("client_id", str(client_id)).execute()
        return UUID(doc_id)

    # INSERT path: stamp provenance once, forever.
    res = sb.table("documents").insert({
        "tenant_id": str(tenant_id), "client_id": str(client_id),
        "source_uri": source_uri,
        **base,
        **prov,
    }).execute()
    if not res.data:
        raise RuntimeError("documents insert returned no rows")
    return UUID(res.data[0]["id"])


def _upsert_chunk(sb, *, tenant_id, document_id, chunk_index, start_page, end_page,
                  text, token_count, metadata, embedding) -> UUID:
    res = sb.rpc("upsert_chunk", {
        "p_tenant_id": str(tenant_id), "p_document_id": str(document_id),
        "p_chunk_index": chunk_index, "p_page_start": start_page, "p_page_end": end_page,
        "p_content": text, "p_content_tokens": token_count,
        "p_metadata": metadata or {}, "p_embedding": embedding,
        "p_embedding_model": _EMBED_MODEL,
    }).execute()
    return UUID(str(res.data))


def _chunk_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]{4,}", text.lower())
    if not tokens:
        return set()
    return set(tokens[:_LEXICAL_TOKEN_LIMIT])


def _page_distance(chunk_a: Dict[str, Any], chunk_b: Dict[str, Any]) -> Optional[int]:
    a_start = chunk_a.get("start_page")
    a_end = chunk_a.get("end_page")
    b_start = chunk_b.get("start_page")
    b_end = chunk_b.get("end_page")
    if a_start is None and a_end is None:
        return None
    if b_start is None and b_end is None:
        return None
    a_lo = a_start if a_start is not None else a_end
    a_hi = a_end if a_end is not None else a_start
    b_lo = b_start if b_start is not None else b_end
    b_hi = b_end if b_end is not None else b_start
    if a_hi is None or b_lo is None or b_hi is None or a_lo is None:
        return None
    if a_hi < b_lo:
        return b_lo - a_hi
    if b_hi < a_lo:
        return a_lo - b_hi
    return 0


def _build_semantic_candidates(chunk_texts: List[str], chunk_items: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    token_sets = [_chunk_tokens(text) for text in chunk_texts]
    token_to_indices: Dict[str, set[int]] = {}
    for idx, tokens in enumerate(token_sets):
        for token in tokens:
            token_to_indices.setdefault(token, set()).add(idx)

    candidates: Dict[int, List[int]] = {}
    total_chunks = len(chunk_texts)
    for idx in range(total_chunks):
        candidate_ids: set[int] = set()

        # Always preserve local continuity.
        for offset in range(1, _ADJACENCY_WINDOW + 1):
            if idx - offset >= 0:
                candidate_ids.add(idx - offset)
            if idx + offset < total_chunks:
                candidate_ids.add(idx + offset)

        # Add nearby chunks by page proximity when page metadata exists.
        for other_idx in range(total_chunks):
            if other_idx == idx:
                continue
            distance = _page_distance(chunk_items[idx], chunk_items[other_idx])
            if distance is not None and distance <= _PAGE_WINDOW:
                candidate_ids.add(other_idx)

        # Add lexically related chunks, but only through reasonably selective tokens.
        for token in token_sets[idx]:
            related = token_to_indices.get(token, set())
            if len(related) > _MAX_TOKEN_DOC_FREQ:
                continue
            candidate_ids.update(related)

        candidate_ids.discard(idx)
        ordered = sorted(
            candidate_ids,
            key=lambda other_idx: (
                abs(other_idx - idx),
                _page_distance(chunk_items[idx], chunk_items[other_idx]) or 0,
                other_idx,
            ),
        )
        candidates[idx] = ordered[:_MAX_CANDIDATES_PER_CHUNK]

    return candidates


# -- KG build --


def _build_kg_nodes_and_edges(sb, *, tenant_id, client_id, document_id, chunk_ids, chunk_texts, chunk_embeddings, chunk_items):
    """Create Chunk KG nodes plus adjacency and bounded semantic edges."""
    if not chunk_ids:
        return 0, 0

    chunk_id_to_node_id: Dict[str, UUID] = {}
    nodes = 0
    for cid, text, emb in zip(chunk_ids, chunk_texts, chunk_embeddings):
        preview = text[:80].strip().replace("\n", " ")
        try:
            res = sb.rpc("upsert_kg_node", {
                "p_tenant_id": str(tenant_id), "p_client_id": str(client_id),
                "p_node_key": f"chunk:{cid}", "p_type": "Chunk",
                "p_name": f"Chunk", "p_description": preview + ("…" if len(text) > 80 else ""),
                "p_properties": {
                    "chunk_id": str(cid),
                    "document_id": str(document_id),
                    "tenant_id": str(tenant_id),
                    "client_id": str(client_id),
                },
                "p_embedding": emb, "p_status": "active",
                "p_embedding_model": _EMBED_MODEL,
            }).execute()
            node_id = UUID(str(res.data))
            chunk_id_to_node_id[str(cid)] = node_id
            nodes += 1

            sb.table("kg_node_evidence").upsert({
                "tenant_id": str(tenant_id), "client_id": str(client_id),
                "node_id": str(node_id), "chunk_id": str(cid),
                "quote": text[:200].strip() or None, "score": 1.0,
            }, on_conflict="tenant_id,client_id,node_id,chunk_id").execute()
        except Exception as e:
            logger.warning("Chunk node upsert failed for %s: %s", cid, e)

    # Adjacency edges preserve local context even when semantic labeling misses it.
    edges = 0
    for idx in range(len(chunk_ids) - 1):
        src_nid = chunk_id_to_node_id.get(str(chunk_ids[idx]))
        dst_nid = chunk_id_to_node_id.get(str(chunk_ids[idx + 1]))
        if not src_nid or not dst_nid:
            continue
        try:
            sb.rpc("upsert_kg_edge", {
                "p_tenant_id": str(tenant_id), "p_client_id": str(client_id),
                "p_src_id": str(src_nid), "p_dst_id": str(dst_nid),
                "p_rel_type": "adjacent_to", "p_weight": 1.0,
                "p_properties": {"method": "chunk_sequence", "document_id": str(document_id)},
            }).execute()
            edges += 1
        except Exception as e:
            logger.warning("Adjacency edge failed: %s", e)

    # Bounded semantic edges avoid the old all-pairs O(n^2) similarity matrix.
    if len(chunk_embeddings) >= 2:
        vectors = np.array(chunk_embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = vectors / norms
        candidates_by_idx = _build_semantic_candidates(chunk_texts, chunk_items)

        for i, candidate_indices in candidates_by_idx.items():
            if not candidate_indices:
                continue
            src_nid = chunk_id_to_node_id.get(str(chunk_ids[i]))
            if not src_nid:
                continue
            candidate_vectors = normed[candidate_indices]
            sims = candidate_vectors @ normed[i]
            ranked_pairs = [
                (candidate_indices[pos], float(score))
                for pos, score in enumerate(sims)
                if float(score) >= _SIMILARITY_THRESHOLD and abs(candidate_indices[pos] - i) > _ADJACENCY_WINDOW
            ]
            ranked_pairs.sort(key=lambda item: item[1], reverse=True)
            for j, score in ranked_pairs[:_MAX_EDGES_PER_CHUNK]:
                dst_nid = chunk_id_to_node_id.get(str(chunk_ids[j]))
                if not dst_nid:
                    continue
                try:
                    sb.rpc("upsert_kg_edge", {
                        "p_tenant_id": str(tenant_id), "p_client_id": str(client_id),
                        "p_src_id": str(src_nid), "p_dst_id": str(dst_nid),
                        "p_rel_type": "related_to", "p_weight": score,
                        "p_properties": {"method": "bounded_chunk_similarity", "document_id": str(document_id)},
                    }).execute()
                    edges += 1
                except Exception as e:
                    logger.warning("Similarity edge failed: %s", e)

    return nodes, edges


def _link_entities(sb, *, tenant_id, client_id, entities, chunk_ids, chunk_texts):
    """Create Entity nodes and mentions/co_occurs edges to chunks."""
    if not entities or not chunk_ids:
        return 0

    # Embed entity names
    embed_input = [f"{e.type}: {e.name}" for e in entities]
    try:
        entity_embeddings = _embed_in_batches(embed_input)
    except Exception as e:
        logger.warning("Entity embedding failed: %s", e)
        entity_embeddings = [None] * len(entities)

    # Upsert entity nodes.
    #
    # Step 4 (write-time reconciliation): before minting a new string-keyed node,
    # look for an existing near-duplicate Entity of the SAME type via
    # nearest_entity_candidate (which — unlike search_kg_nodes — also considers
    # provisional 'pending_linking' nodes, so the first two wordings of a concept
    # can pool and cross the trust gate). If a candidate is at/above the
    # conservative similarity floor, route this mention onto it (bumping its
    # seen_count) instead of creating a duplicate. When uncertain we mint a new
    # node — a wrong merge silently corrupts the graph and is worse than a dup.
    entity_node_ids: Dict[str, UUID] = {}
    for ent, emb in zip(entities, entity_embeddings):
        normalized_name = ent.name.lower().strip()
        string_key = f"entity:{tenant_id}:{client_id}:{ent.type.lower()}:{normalized_name}"

        # Defaults = mint a fresh node at the deterministic string key.
        upsert_key = string_key
        upsert_name = ent.name
        upsert_desc = f"{ent.type}: {ent.name}"
        upsert_props: Dict[str, Any] = {
            "entity_type": ent.type,
            "tenant_id": str(tenant_id),
            "client_id": str(client_id),
            **ent.properties,
        }
        upsert_emb = emb
        reconciled_onto: Optional[str] = None

        if emb is not None:
            try:
                cand = sb.rpc("nearest_entity_candidate", {
                    "p_tenant_id":       str(tenant_id),
                    "p_client_id":       str(client_id),
                    "p_embedding":       emb,
                    "p_entity_type":     ent.type,
                    "p_min_similarity":  _ENTITY_RECONCILE_THRESHOLD,
                    "p_top_k":           1,
                    "p_embedding_model": _EMBED_MODEL,
                }).execute()
                rows = cand.data or []
                top = rows[0] if rows else None
                # A reconcile only when the nearest above-floor node is a DIFFERENT
                # node_key (a different wording). An exact-key hit is just this same
                # entity being re-ingested — fall through to the normal upsert.
                if top and top.get("node_key") and top["node_key"] != string_key:
                    upsert_key = top["node_key"]
                    upsert_name = top.get("name") or upsert_name        # keep survivor label — no drift
                    upsert_desc = top.get("description") or upsert_desc
                    upsert_props = {}                                    # {} merges as a no-op — don't drift survivor props
                    upsert_emb = None                                    # keep survivor embedding
                    reconciled_onto = top["node_key"]
            except Exception as e:
                logger.warning("entity reconcile lookup failed for '%s' (minting new): %s", ent.name, e)

        try:
            res = sb.rpc("upsert_kg_node", {
                "p_tenant_id": str(tenant_id), "p_client_id": str(client_id),
                "p_node_key": upsert_key, "p_type": "Entity",
                "p_name": upsert_name, "p_description": upsert_desc,
                "p_properties": upsert_props,
                "p_embedding": upsert_emb, "p_status": "active",
                "p_embedding_model": _EMBED_MODEL,
            }).execute()
            entity_node_ids[normalized_name] = UUID(str(res.data))
            if reconciled_onto:
                logger.info(
                    "entity reconcile: '%s' (%s) routed onto %s (sim>=%.2f)",
                    ent.name, ent.type, reconciled_onto, _ENTITY_RECONCILE_THRESHOLD,
                )
        except Exception as e:
            logger.warning("Entity upsert failed for '%s': %s", ent.name, e)

    if not entity_node_ids:
        return 0

    # Fetch chunk node IDs
    chunk_node_ids: Dict[int, UUID] = {}
    for idx, cid in enumerate(chunk_ids):
        try:
            res = (sb.table("kg_nodes").select("id")
                   .eq("tenant_id", str(tenant_id))
                   .eq("client_id", str(client_id))
                   .eq("node_key", f"chunk:{cid}")
                   .limit(1).execute())
            if res.data:
                chunk_node_ids[idx] = UUID(res.data[0]["id"])
        except Exception:
            pass

    # Scan for mentions + co-occurrence
    total_mentions = 0
    for idx, (text, cid) in enumerate(zip(chunk_texts, chunk_ids)):
        text_lower = text.lower()
        chunk_nid = chunk_node_ids.get(idx)
        entities_in_chunk: List[str] = []

        for name_lower, entity_nid in entity_node_ids.items():
            if name_lower in text_lower:
                entities_in_chunk.append(name_lower)
                if chunk_nid:
                    try:
                        sb.rpc("upsert_kg_edge", {
                            "p_tenant_id": str(tenant_id), "p_client_id": str(client_id),
                            "p_src_id": str(chunk_nid), "p_dst_id": str(entity_nid),
                            "p_rel_type": "mentions", "p_weight": 1.0, "p_properties": {},
                        }).execute()
                        total_mentions += 1
                    except Exception:
                        pass

                    # Evidence quote
                    try:
                        pos = text_lower.find(name_lower)
                        start = max(0, pos - 50)
                        end = min(len(text), pos + len(name_lower) + 50)
                        sb.table("kg_node_evidence").upsert({
                            "tenant_id": str(tenant_id), "client_id": str(client_id),
                            "node_id": str(entity_nid), "chunk_id": str(cid),
                            "quote": text[start:end].strip(), "score": 1.0,
                        }, on_conflict="tenant_id,client_id,node_id,chunk_id").execute()
                    except Exception:
                        pass

        # Co-occurrence
        if len(entities_in_chunk) > 1:
            for i in range(len(entities_in_chunk)):
                for j in range(i + 1, len(entities_in_chunk)):
                    try:
                        sb.rpc("upsert_kg_edge", {
                            "p_tenant_id": str(tenant_id), "p_client_id": str(client_id),
                            "p_src_id": str(entity_node_ids[entities_in_chunk[i]]),
                            "p_dst_id": str(entity_node_ids[entities_in_chunk[j]]),
                            "p_rel_type": "co_occurs", "p_weight": 1.0,
                            "p_properties": {"source_chunk_id": str(cid)},
                        }).execute()
                    except Exception:
                        pass

    return total_mentions


# -- Endpoints --


def _execute_processed_ingest(req: ProcessedDocumentRequest) -> IngestProcessedResponse:
    """Synchronous implementation of /ingest/processed. Called by the arq worker."""
    sb = get_supabase()
    memory_state = MemoryStateService(sb)
    warnings: List[str] = []
    started = time.perf_counter()

    # Plan-gated chunk-count check (rejects oversized payloads before embedding cost)
    _assert_chunks_within_plan(sb, req.tenant_id, len(req.chunks or []))

    # Decode and upload file
    try:
        file_bytes = base64.b64decode(req.file_bytes_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    source_uri = _upload_to_bucket(
        sb,
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        file_bytes=file_bytes,
        file_name=req.file_name,
    )

    # Set audit context (read by triggers) and capture provenance
    _set_audit_context(sb, actor_id=req.actor_id, request_id=req.request_id)

    # Document row (with provenance)
    document_id = _upsert_document(
        sb, tenant_id=req.tenant_id, client_id=req.client_id,
        source_type=req.source_type, source_uri=source_uri,
        title=req.title, metadata={
            **req.metadata,
            "file_name": req.file_name,
            "file_type": req.source_type,
            "tenant_id": str(req.tenant_id),
            "client_id": str(req.client_id),
        },
        source_timestamp=req.source_timestamp,
        is_pinned=req.is_pinned,
        is_canonical=req.is_canonical,
        status=req.status,
        actor_id=req.actor_id,
        actor_type=req.actor_type,
        source_app=req.source_app,
        request_id=req.request_id,
        previous_version_id=req.previous_version_id,
    )

    if not req.chunks:
        return IngestProcessedResponse(
            document_id=str(document_id), source_uri=source_uri,
            chunks_upserted=0, warnings=["No chunks provided."],
        )

    # Embed chunks (charges tenant's daily embedding-token quota)
    chunk_texts = [c.text for c in req.chunks]
    try:
        embeddings = _embed_in_batches(chunk_texts, tenant_id=req.tenant_id)
    except HTTPException:
        raise  # quota errors should surface as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # Store chunks
    chunk_ids: List[UUID] = []
    for idx, (chunk, emb) in enumerate(zip(req.chunks, embeddings)):
        try:
            cid = _upsert_chunk(
                sb, tenant_id=req.tenant_id, document_id=document_id,
                chunk_index=idx, start_page=chunk.start_page, end_page=chunk.end_page,
                text=chunk.text, token_count=chunk.token_count,
                metadata={
                    "source_uri": source_uri,
                    "source_type": req.source_type,
                    "tenant_id": str(req.tenant_id),
                    "client_id": str(req.client_id),
                    "document_id": str(document_id),
                    "chunk_index": idx,
                },
                embedding=emb,
            )
            chunk_ids.append(cid)
        except Exception as e:
            warnings.append(f"Chunk {idx} failed: {e}")

    # KG build
    try:
        nodes, edges = _build_kg_nodes_and_edges(
            sb, tenant_id=req.tenant_id, client_id=req.client_id,
            document_id=document_id,
            chunk_ids=chunk_ids, chunk_texts=chunk_texts, chunk_embeddings=embeddings,
            chunk_items=[c.model_dump() for c in req.chunks],
        )
        logger.info("KG build: %d nodes, %d edges", nodes, edges)
    except Exception as e:
        warnings.append(f"KG build failed: {e}")

    # Entity linking
    entities_linked = 0
    if req.entities:
        try:
            entities_linked = _link_entities(
                sb, tenant_id=req.tenant_id, client_id=req.client_id,
                entities=req.entities, chunk_ids=chunk_ids, chunk_texts=chunk_texts,
            )
        except Exception as e:
            warnings.append(f"Entity linking failed: {e}")

    versions = memory_state.bump_dual(
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        change_type="ingest",
        metadata={"document_id": str(document_id), "client_id": str(req.client_id), "source_type": req.source_type},
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "ingest.processed tenant=%s client=%s document=%s chunks=%d entities=%d warnings=%d client_version=%s tenant_version=%s elapsed_ms=%s",
        req.tenant_id,
        req.client_id,
        document_id,
        len(chunk_ids),
        entities_linked,
        len(warnings),
        versions["client_version"],
        versions["tenant_version"],
        elapsed_ms,
    )

    return IngestProcessedResponse(
        document_id=str(document_id), source_uri=source_uri,
        chunks_upserted=len(chunk_ids), entities_linked=entities_linked,
        warnings=warnings,
    )


def _execute_processed_web_ingest(req: ProcessedWebRequest) -> IngestProcessedResponse:
    """Synchronous implementation of /ingest/processed-web. Called by the arq worker."""
    sb = get_supabase()
    memory_state = MemoryStateService(sb)
    warnings: List[str] = []
    started = time.perf_counter()

    _assert_chunks_within_plan(sb, req.tenant_id, len(req.chunks or []))
    _set_audit_context(sb, actor_id=req.actor_id, request_id=req.request_id)

    document_id = _upsert_document(
        sb, tenant_id=req.tenant_id, client_id=req.client_id,
        source_type="web", source_uri=req.url,
        title=req.title, metadata={
            **req.metadata,
            "tenant_id": str(req.tenant_id),
            "client_id": str(req.client_id),
            "source_uri": req.url,
        },
        source_timestamp=req.source_timestamp,
        is_pinned=req.is_pinned,
        is_canonical=req.is_canonical,
        status=req.status,
        actor_id=req.actor_id,
        actor_type=req.actor_type,
        source_app=req.source_app,
        request_id=req.request_id,
        previous_version_id=req.previous_version_id,
    )

    if not req.chunks:
        return IngestProcessedResponse(
            document_id=str(document_id), source_uri=req.url,
            chunks_upserted=0, warnings=["No chunks provided."],
        )

    chunk_texts = [c.text for c in req.chunks]
    try:
        embeddings = _embed_in_batches(chunk_texts, tenant_id=req.tenant_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    chunk_ids: List[UUID] = []
    for idx, (chunk, emb) in enumerate(zip(req.chunks, embeddings)):
        try:
            cid = _upsert_chunk(
                sb, tenant_id=req.tenant_id, document_id=document_id,
                chunk_index=idx, start_page=chunk.start_page, end_page=chunk.end_page,
                text=chunk.text, token_count=chunk.token_count,
                metadata={
                    "source_uri": req.url,
                    "source_type": "web",
                    "tenant_id": str(req.tenant_id),
                    "client_id": str(req.client_id),
                    "document_id": str(document_id),
                    "chunk_index": idx,
                },
                embedding=emb,
            )
            chunk_ids.append(cid)
        except Exception as e:
            warnings.append(f"Chunk {idx} failed: {e}")

    try:
        nodes, edges = _build_kg_nodes_and_edges(
            sb, tenant_id=req.tenant_id, client_id=req.client_id,
            document_id=document_id,
            chunk_ids=chunk_ids, chunk_texts=chunk_texts, chunk_embeddings=embeddings,
            chunk_items=[c.model_dump() for c in req.chunks],
        )
        logger.info("KG build: %d nodes, %d edges", nodes, edges)
    except Exception as e:
        warnings.append(f"KG build failed: {e}")

    entities_linked = 0
    if req.entities:
        try:
            entities_linked = _link_entities(
                sb, tenant_id=req.tenant_id, client_id=req.client_id,
                entities=req.entities, chunk_ids=chunk_ids, chunk_texts=chunk_texts,
            )
        except Exception as e:
            warnings.append(f"Entity linking failed: {e}")

    versions = memory_state.bump_dual(
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        change_type="ingest",
        metadata={"document_id": str(document_id), "client_id": str(req.client_id), "source_type": "web"},
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "ingest.processed_web tenant=%s client=%s document=%s chunks=%d entities=%d warnings=%d client_version=%s tenant_version=%s elapsed_ms=%s",
        req.tenant_id,
        req.client_id,
        document_id,
        len(chunk_ids),
        entities_linked,
        len(warnings),
        versions["client_version"],
        versions["tenant_version"],
        elapsed_ms,
    )

    return IngestProcessedResponse(
        document_id=str(document_id), source_uri=req.url,
        chunks_upserted=len(chunk_ids), entities_linked=entities_linked,
        warnings=warnings,
    )


# ── Async enqueue endpoints + job status ─────────────────────────────────────


class IngestJobAck(BaseModel):
    job_id: str
    status: str = "queued"


class IngestJobStatus(BaseModel):
    job_id: str
    status: str
    job_type: str
    tenant_id: str
    client_id: Optional[str] = None
    document_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    # Provenance
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None
    source_app: Optional[str] = None
    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    request_id: Optional[str] = None


class IngestJobsListResponse(BaseModel):
    items: List[IngestJobStatus]
    total: int


def _redis_available() -> bool:
    """Check if Redis is reachable (cached for the process lifetime)."""
    try:
        import redis as _redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = _redis.from_url(url, decode_responses=True)
        r.ping()
        return True
    except Exception:
        return False


def _enqueue_job(
    *,
    request: Request,
    job_type: str,
    payload: Dict[str, Any],
    tenant_id: str,
    client_id: Optional[str],
    actor_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    source_app: Optional[str] = None,
    body_request_id: Optional[str] = None,
    source_type: Optional[str] = None,
    source_uri: Optional[str] = None,
) -> str:
    """Create ingest_jobs row, enqueue arq task, return job_id.

    Falls back to synchronous execution if Redis/arq is unavailable.

    If body_request_id is provided AND a job already exists at that
    (tenant, request_id), reuses the existing job_id (idempotency key).
    """
    sb = get_supabase()
    state = getattr(request, "state", None)
    key_id = getattr(state, "key_id", None) if state else None
    # Prefer body request_id (from agent) over middleware request_id, since the
    # agent's request_id is the canonical correlation key for the user-facing op.
    req_id = body_request_id or (getattr(state, "request_id", None) if state else None)

    # Idempotency: if we already have a job for this (tenant, request_id),
    # return it instead of creating a duplicate.
    if req_id:
        try:
            existing = (
                sb.table("ingest_jobs").select("id, status")
                .eq("tenant_id", tenant_id)
                .eq("request_id", req_id)
                .order("enqueued_at", desc=True).limit(1).execute()
            )
            if existing.data:
                logger.info("ingest.idempotency_hit request_id=%s job=%s", req_id, existing.data[0]["id"])
                return existing.data[0]["id"]
        except Exception:
            pass

    row = sb.table("ingest_jobs").insert({
        "tenant_id": tenant_id,
        "client_id": client_id,
        "job_type": job_type,
        "status": "queued",
        "request_id": req_id,
        "key_id": key_id,
        "actor_id": actor_id,
        "actor_type": actor_type,
        "source_app": source_app,
        "source_type": source_type,
        "source_uri": source_uri,
    }).execute()
    job_id = (row.data or [{}])[0].get("id")
    if not job_id:
        raise HTTPException(status_code=500, detail="Failed to create job row")

    enqueued = False
    if _redis_available():
        try:
            import asyncio
            from arq.connections import RedisSettings, create_pool
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

            async def _enqueue():
                pool = await create_pool(RedisSettings.from_dsn(redis_url))
                task_name = "run_processed_ingest_task" if job_type == "processed" else "run_processed_web_ingest_task"
                await pool.enqueue_job(task_name, job_id, payload)

            asyncio.run(_enqueue())
            enqueued = True
        except Exception as e:
            logger.warning("arq enqueue failed, falling back to sync: %s", e)

    if not enqueued:
        # No Redis / arq unavailable — run synchronously (old behavior)
        logger.info("ingest.sync_fallback job=%s type=%s", job_id, job_type)
        try:
            sb.table("ingest_jobs").update({"status": "running", "started_at": datetime.utcnow().isoformat()}).eq("id", job_id).execute()
            if job_type == "processed":
                result = _execute_processed_ingest(ProcessedDocumentRequest.model_validate(payload))
            else:
                result = _execute_processed_web_ingest(ProcessedWebRequest.model_validate(payload))
            sb.table("ingest_jobs").update({
                "status": "complete",
                "completed_at": datetime.utcnow().isoformat(),
                "document_id": result.document_id,
                "result": result.model_dump(mode="json"),
            }).eq("id", job_id).execute()
        except Exception as e:
            sb.table("ingest_jobs").update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e)[:2000],
            }).eq("id", job_id).execute()
            raise HTTPException(status_code=500, detail=str(e))

    AuditService(sb).record(
        request=request,
        action=f"ingest.{job_type}.{'enqueue' if enqueued else 'sync'}",
        resource_type="ingest_job",
        resource_id=job_id,
        metadata={"client_id": client_id, "async": enqueued},
    )
    return job_id


@router.post("/processed", response_model=IngestJobAck, status_code=202)
def ingest_processed(req: ProcessedDocumentRequest, request: Request) -> IngestJobAck:
    """Ingest a processed document. Async via arq if Redis available, else synchronous."""
    job_id = _enqueue_job(
        request=request,
        job_type="processed",
        payload=req.model_dump(mode="json"),
        tenant_id=str(req.tenant_id),
        client_id=str(req.client_id) if req.client_id else None,
        actor_id=req.actor_id,
        actor_type=req.actor_type,
        source_app=req.source_app,
        body_request_id=req.request_id,
        source_type=req.source_type,
        source_uri=req.file_name,
    )
    return IngestJobAck(job_id=job_id)


@router.post("/processed-web", response_model=IngestJobAck, status_code=202)
def ingest_processed_web(req: ProcessedWebRequest, request: Request) -> IngestJobAck:
    """Ingest a processed web scrape. Async via arq if Redis available, else synchronous."""
    job_id = _enqueue_job(
        request=request,
        job_type="processed-web",
        payload=req.model_dump(mode="json"),
        tenant_id=str(req.tenant_id),
        client_id=str(req.client_id) if req.client_id else None,
        actor_id=req.actor_id,
        actor_type=req.actor_type,
        source_app=req.source_app,
        body_request_id=req.request_id,
        source_type="web",
        source_uri=req.url,
    )
    return IngestJobAck(job_id=job_id)


def _row_to_job_status(row: Dict[str, Any]) -> IngestJobStatus:
    return IngestJobStatus(
        job_id=row["id"],
        status=row["status"],
        job_type=row["job_type"],
        tenant_id=row["tenant_id"],
        client_id=row.get("client_id"),
        document_id=row.get("document_id"),
        result=row.get("result"),
        error=row.get("error"),
        enqueued_at=row.get("enqueued_at"),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        actor_id=row.get("actor_id"),
        actor_type=row.get("actor_type"),
        source_app=row.get("source_app"),
        source_type=row.get("source_type"),
        source_uri=row.get("source_uri"),
        request_id=row.get("request_id"),
    )


@router.get("/jobs/{job_id}", response_model=IngestJobStatus)
def get_ingest_job(job_id: str, request: Request) -> IngestJobStatus:
    """Fetch status of a single ingest job. Tenant enforced via auth middleware."""
    sb = get_supabase()
    tenant_id = getattr(request.state, "tenant_id", None)
    q = sb.table("ingest_jobs").select("*").eq("id", job_id)
    if tenant_id:
        q = q.eq("tenant_id", str(tenant_id))
    res = q.limit(1).execute()
    rows = res.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Job not found")
    return _row_to_job_status(rows[0])


@router.get("/jobs", response_model=IngestJobsListResponse)
def list_ingest_jobs(
    request: Request,
    tenant_id: Optional[UUID] = None,
    client_id: Optional[UUID] = None,
    actor_id: Optional[str] = None,
    status: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 50,
) -> IngestJobsListResponse:
    """List ingest jobs with filters. Backs upload-history UI in agent service.

    When auth is enabled, tenant scope is enforced from the API key.
    The `tenant_id` query param is allowed only when it matches the key's tenant
    (auth middleware already rejects mismatches at the request level).
    """
    sb = get_supabase()
    auth_tenant = getattr(request.state, "tenant_id", None)
    effective_tenant = str(auth_tenant) if auth_tenant else (str(tenant_id) if tenant_id else None)

    q = sb.table("ingest_jobs").select("*", count="exact").order("enqueued_at", desc=True)
    if effective_tenant:
        q = q.eq("tenant_id", effective_tenant)
    if client_id:
        q = q.eq("client_id", str(client_id))
    if actor_id:
        q = q.eq("actor_id", actor_id)
    if status:
        q = q.eq("status", status)
    if since:
        q = q.gte("enqueued_at", since.isoformat())
    q = q.limit(max(1, min(limit, 500)))

    res = q.execute()
    rows = res.data or []
    return IngestJobsListResponse(
        items=[_row_to_job_status(r) for r in rows],
        total=res.count or len(rows),
    )


# ── /ingest/summary — fast-path for LLM-generated summary chunks ────────────

_SUMMARY_TYPES = {"ContextSummary", "DocumentSummary", "TopicSummary"}


@router.post("/summary", response_model=SummaryIngestResponse, status_code=201)
def ingest_summary(req: SummaryIngestRequest, request: Request) -> SummaryIngestResponse:
    """
    Store an LLM-generated summary as a single-chunk document. Synchronous —
    the work is small (one embedding + a few upserts) and agents need the
    document_id immediately for their response.
    """
    if req.source_type not in _SUMMARY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"source_type must be one of {_SUMMARY_TYPES}",
        )
    # Scope sanity checks
    if req.source_type == "DocumentSummary" and not req.document_id:
        raise HTTPException(status_code=400, detail="DocumentSummary requires document_id")
    if req.source_type == "TopicSummary" and not req.topic:
        raise HTTPException(status_code=400, detail="TopicSummary requires topic")

    sb = get_supabase()
    memory_state = MemoryStateService(sb)
    started = time.perf_counter()

    # 1. Supersede existing canonical summary at the same scope
    superseded_id: Optional[str] = None
    lookup = (
        sb.table("documents")
        .select("id, metadata")
        .eq("tenant_id", str(req.tenant_id))
        .eq("client_id", str(req.client_id))
        .eq("source_type", req.source_type)
        .eq("is_canonical", True)
        .eq("status", "active")
    )
    existing_rows = lookup.execute().data or []
    # Python-side scope match (can't filter on jsonb keys via supabase-py easily)
    for row in existing_rows:
        md = row.get("metadata") or {}
        if req.source_type == "ContextSummary":
            superseded_id = row["id"]
            break
        if req.source_type == "DocumentSummary" and md.get("document_id") == req.document_id:
            superseded_id = row["id"]
            break
        if req.source_type == "TopicSummary" and md.get("topic") == req.topic:
            superseded_id = row["id"]
            break

    if superseded_id:
        sb.table("documents").update({
            "is_canonical": False,
            "status": "deprecated",
        }).eq("id", superseded_id).execute()

    # 2. Embed the summary text (one call)
    try:
        embedding = _embed_texts([req.summary_text], tenant_id=req.tenant_id)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 3. Insert documents row
    md = {
        **(req.extra_metadata or {}),
        "topics": req.topics,
        "source_stats": req.source_stats,
        "source_chunk_ids": req.source_chunk_ids,
        "memory_version_at_generation": req.memory_version_at_generation,
    }
    if req.document_id:
        md["document_id"] = req.document_id
    if req.topic:
        md["topic"] = req.topic

    # Audit context (so triggers stamp the SOC 2 actor on every row write)
    _set_audit_context(sb, actor_id=req.actor_id, request_id=req.request_id)

    source_uri = _summary_source_uri(req)
    doc_res = sb.table("documents").insert({
        "tenant_id": str(req.tenant_id),
        "client_id": str(req.client_id),
        "source_type": req.source_type,
        "source_uri": source_uri,
        "title": _summary_title(req),
        "is_pinned": False,
        "is_canonical": True,
        "status": "active",
        "metadata": md,
        "actor_id": req.actor_id,
        "actor_type": req.actor_type,
        "source_app": req.source_app,
        "request_id": req.request_id,
        # Chain to the row we just deprecated, if any — this becomes the
        # version history for summary regeneration.
        "previous_version_id": superseded_id or req.previous_version_id,
    }).execute()
    document_id = (doc_res.data or [{}])[0].get("id")
    if not document_id:
        raise HTTPException(status_code=500, detail="Failed to create summary document row")

    # 4. Upsert chunk (one row, chunk_index=0)
    chunk_id = sb.rpc("upsert_chunk", {
        "p_tenant_id":       str(req.tenant_id),
        "p_document_id":     document_id,
        "p_chunk_index":     0,
        "p_content":         req.summary_text,
        "p_content_tokens":  len(req.summary_text.split()),
        "p_metadata": {
            "source_type":  req.source_type,
            "tenant_id":    str(req.tenant_id),
            "client_id":    str(req.client_id),
            "document_id":  document_id,
            "chunk_index":  0,
            "is_summary":   True,
        },
        "p_embedding":        embedding,
        "p_embedding_model":  _EMBED_MODEL,
    }).execute().data
    chunk_id = str(chunk_id)

    # 5. Upsert KG node (type = source_type, so ContextSummary/DocumentSummary/TopicSummary)
    node_key = f"summary:{req.source_type.lower()}:{chunk_id}"
    node_id = sb.rpc("upsert_kg_node", {
        "p_tenant_id":       str(req.tenant_id),
        "p_client_id":       str(req.client_id),
        "p_node_key":        node_key,
        "p_type":            req.source_type,
        "p_name":            _summary_title(req),
        "p_description":     req.summary_text[:80],
        "p_properties": {
            "chunk_id":    chunk_id,
            "document_id": document_id,
            "tenant_id":   str(req.tenant_id),
            "client_id":   str(req.client_id),
            "is_summary":  True,
        },
        "p_embedding":       embedding,
        "p_status":          "active",
        "p_embedding_model": _EMBED_MODEL,
    }).execute().data
    node_id = str(node_id)

    # 6. Entity mentions with weight=0.5 (source evidence stays at 1.0)
    for ent in req.entities:
        normalized = ent.name.lower().strip()
        ent_key = f"entity:{req.tenant_id}:{req.client_id}:{ent.type.lower()}:{normalized}"
        try:
            ent_node_id = sb.rpc("upsert_kg_node", {
                "p_tenant_id":       str(req.tenant_id),
                "p_client_id":       str(req.client_id),
                "p_node_key":        ent_key,
                "p_type":            "Entity",
                "p_name":            ent.name,
                "p_description":     f"{ent.type}: {ent.name}",
                "p_properties": {
                    "entity_type": ent.type,
                    "tenant_id":   str(req.tenant_id),
                    "client_id":   str(req.client_id),
                    **ent.properties,
                },
                "p_embedding":       None,
                "p_status":          "active",
                "p_embedding_model": _EMBED_MODEL,
            }).execute().data
            sb.rpc("upsert_kg_edge", {
                "p_tenant_id":  str(req.tenant_id),
                "p_client_id":  str(req.client_id),
                "p_src_id":     node_id,
                "p_dst_id":     str(ent_node_id),
                "p_rel_type":   "mentions",
                "p_weight":     0.5,          # lower than source-origin (1.0)
                "p_properties": {"origin": "summary"},
            }).execute()
        except Exception as e:
            logger.warning("summary entity link failed: %s", e)

    # 7. Bump memory state
    versions = memory_state.bump_dual(
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        change_type="summary",
        metadata={
            "document_id":  document_id,
            "source_type":  req.source_type,
            "topic":        req.topic,
            "superseded":   superseded_id,
        },
    )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "ingest.summary tenant=%s client=%s type=%s doc=%s chunk=%s node=%s superseded=%s client_version=%s tenant_version=%s elapsed_ms=%s",
        req.tenant_id, req.client_id, req.source_type,
        document_id, chunk_id, node_id, superseded_id,
        versions["client_version"], versions["tenant_version"], elapsed_ms,
    )

    AuditService(sb).record(
        request=request,
        action=f"summary.{req.source_type.lower()}.upsert",
        resource_type="document",
        resource_id=document_id,
        metadata={"source_type": req.source_type, "superseded": superseded_id},
    )

    return SummaryIngestResponse(
        document_id=document_id,
        chunk_id=chunk_id,
        node_id=node_id,
        superseded_document_id=superseded_id,
        memory_version=versions["client_version"],
    )


def _summary_source_uri(req: SummaryIngestRequest) -> str:
    if req.source_type == "ContextSummary":
        return f"summary:context:{req.tenant_id}:{req.client_id}"
    if req.source_type == "DocumentSummary":
        return f"summary:document:{req.tenant_id}:{req.client_id}:{req.document_id}"
    if req.source_type == "TopicSummary":
        slug = re.sub(r"[^a-z0-9]+", "-", (req.topic or "").lower()).strip("-")
        return f"summary:topic:{req.tenant_id}:{req.client_id}:{slug}"
    return f"summary:{req.tenant_id}:{req.client_id}"


def _summary_title(req: SummaryIngestRequest) -> str:
    if req.source_type == "ContextSummary":
        return "Context Summary"
    if req.source_type == "DocumentSummary":
        return f"Document Summary ({req.document_id})"
    if req.source_type == "TopicSummary":
        return f"Topic Summary: {req.topic}"
    return "Summary"


# ── /ingest/valid — ingest into Valid's dedicated KG (sales bot corpus) ──────


_VALID_ALLOWED_TYPES = {"pdf", "docx", "pptx", "xlsx", "csv", "txt", "md"}


class ValidIngestRequest(_ProvenanceMixin):
    """Ingest a file into the valid-docs KG (client-facing sales bot).

    No tenant/client scoping — valid's corpus is a single global collection.
    Accepts pre-chunked file content. Stores the file in the valid-docs bucket.
    Builds valid_chunks + valid_kg_nodes + valid_kg_edges.

    Allowed source_types: pdf, docx, pptx, xlsx, csv, txt, md
    (no web scrapes, webvtt, or json)
    """
    file_name: str
    file_bytes_b64: str
    title: str
    source_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[ChunkItem] = Field(default_factory=list)
    entities: List[EntityItem] = Field(default_factory=list)


class ValidIngestResponse(BaseModel):
    document_id: str
    chunks_upserted: int
    nodes_created: int
    edges_created: int
    entities_linked: int = 0
    warnings: List[str] = Field(default_factory=list)


@router.post("/valid", response_model=ValidIngestResponse, status_code=201)
def ingest_valid(req: ValidIngestRequest, request: Request) -> ValidIngestResponse:
    """
    Ingest a file into Valid's dedicated KG for the landing page sales bot.

    Allowed file types: pdf, docx, pptx, xlsx, csv, txt, md
    Stores the file in the valid-docs bucket.

    Flow:
      1. Validate file type + upload to valid-docs bucket
      2. Create a documents row
      3. Embed + store chunks in valid_chunks
      4. Build valid_kg_nodes (one per chunk + one per entity)
      5. Build valid_kg_edges (adjacency, semantic similarity, mentions, co-occurrence)
      6. Store evidence quotes in valid_kg_node_evidence
    """
    if req.source_type.lower() not in _VALID_ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"source_type must be one of {sorted(_VALID_ALLOWED_TYPES)}, got '{req.source_type}'",
        )

    sb = get_supabase()
    warnings: List[str] = []
    started = time.perf_counter()

    # Use a fixed tenant_id for valid's own content (not multi-tenant)
    VALID_TENANT_ID = "00000000-0000-0000-0000-000000000000"

    # Chunk-count cap (valid's tenant_plans row should be 'enterprise')
    scopes = getattr(getattr(request, "state", None), "scopes", []) or []
    _assert_chunks_within_plan(sb, VALID_TENANT_ID, len(req.chunks or []), scopes=scopes)

    # 0. Decode and upload file to valid-docs bucket
    try:
        file_bytes = base64.b64decode(req.file_bytes_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    safe_name = _sanitize_storage_key(req.file_name.lstrip("/"))
    storage_path = f"{VALID_TENANT_ID}/{safe_name}"
    try:
        sb.storage.from_(_VALID_DOCS_BUCKET).upload(storage_path, file_bytes, file_options={"upsert": "true"})
    except Exception as e:
        warnings.append(f"File upload to valid-docs failed: {e}")

    # Audit context (read by triggers)
    _set_audit_context(sb, actor_id=req.actor_id, request_id=req.request_id)

    # 1. Document row — provenance is INSERT-only, never overwritten on update
    source_uri = f"bucket:{_VALID_DOCS_BUCKET}/{storage_path}"
    base_fields = {
        "source_type": req.source_type,
        "source_uri": source_uri,
        "title": req.title,
        "is_canonical": True,
        "status": "active",
        "metadata": {**req.metadata, "corpus": "valid-docs", "file_name": req.file_name, "file_type": req.source_type},
    }
    existing = (
        sb.table("documents").select("id")
        .eq("tenant_id", VALID_TENANT_ID)
        .is_("client_id", "null")
        .eq("source_uri", source_uri)
        .limit(1).execute()
    )
    if existing.data:
        document_id = existing.data[0]["id"]
        sb.table("documents").update(base_fields).eq("id", document_id).execute()
    else:
        prov = {
            k: v for k, v in {
                "actor_id": req.actor_id,
                "actor_type": req.actor_type,
                "source_app": req.source_app,
                "request_id": req.request_id,
                "previous_version_id": req.previous_version_id,
            }.items() if v is not None
        }
        doc_res = sb.table("documents").insert({
            "tenant_id": VALID_TENANT_ID,
            **base_fields,
            **prov,
        }).execute()
        document_id = (doc_res.data or [{}])[0].get("id")
        if not document_id:
            raise HTTPException(status_code=500, detail="Failed to create document row")

    if not req.chunks:
        return ValidIngestResponse(
            document_id=str(document_id),
            chunks_upserted=0, nodes_created=0, edges_created=0,
            warnings=["No chunks provided."],
        )

    # 2. Embed all chunks (charged to the fixed VALID_TENANT_ID daily quota)
    chunk_texts = [c.text for c in req.chunks]
    try:
        embeddings = _embed_in_batches(chunk_texts, tenant_id=VALID_TENANT_ID)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 3. Store valid_chunks + valid_kg_nodes (one per chunk)
    chunk_ids: List[str] = []
    chunk_node_ids: Dict[int, str] = {}
    nodes_created = 0

    for idx, (chunk, emb) in enumerate(zip(req.chunks, embeddings)):
        # valid_chunk row
        try:
            cid = sb.rpc("upsert_valid_chunk", {
                "p_tenant_id": VALID_TENANT_ID,
                "p_document_id": document_id,
                "p_chunk_index": idx,
                "p_page_start": chunk.start_page,
                "p_page_end": chunk.end_page,
                "p_content": chunk.text,
                "p_content_tokens": chunk.token_count or len(chunk.text.split()),
                "p_metadata": {
                    "source_uri": source_uri,
                    "source_type": req.source_type,
                    "document_id": str(document_id),
                    "chunk_index": idx,
                },
                "p_embedding": emb,
                "p_embedding_model": _EMBED_MODEL,
            }).execute().data
            cid = str(cid)
            chunk_ids.append(cid)
        except Exception as e:
            warnings.append(f"Chunk {idx} storage failed: {e}")
            continue

        # valid_kg_node for this chunk
        preview = chunk.text[:80].strip().replace("\n", " ")
        node_key = f"valid:chunk:{cid}"
        try:
            nid = sb.rpc("upsert_valid_kg_node", {
                "p_node_key": node_key,
                "p_type": "Chunk",
                "p_name": "Chunk",
                "p_description": preview + ("…" if len(chunk.text) > 80 else ""),
                "p_properties": {
                    "chunk_id": cid,
                    "document_id": str(document_id),
                    "chunk_index": idx,
                },
                "p_embedding": emb,
                "p_status": "active",
                "p_embedding_model": _EMBED_MODEL,
            }).execute().data
            nid = str(nid)
            chunk_node_ids[idx] = nid
            nodes_created += 1

            # Evidence quote
            sb.table("valid_kg_node_evidence").upsert({
                "node_id": nid,
                "chunk_id": cid,
                "quote": chunk.text[:200].strip(),
                "score": 1.0,
            }, on_conflict="node_id,chunk_id").execute()
        except Exception as e:
            warnings.append(f"Chunk {idx} KG node failed: {e}")

    # 4. Adjacency edges (chunk[i] → chunk[i+1])
    edges_created = 0
    for i in range(len(chunk_ids) - 1):
        src = chunk_node_ids.get(i)
        dst = chunk_node_ids.get(i + 1)
        if src and dst:
            try:
                sb.rpc("upsert_valid_kg_edge", {
                    "p_src_id": src, "p_dst_id": dst,
                    "p_rel_type": "adjacent_to", "p_weight": 1.0,
                    "p_properties": {"document_id": str(document_id)},
                }).execute()
                edges_created += 1
            except Exception:
                pass

    # 5. Semantic similarity edges (bounded, same as main KG)
    if len(embeddings) > 1:
        try:
            emb_array = np.array(embeddings)
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normed = emb_array / norms

            for i in range(len(normed)):
                src_nid = chunk_node_ids.get(i)
                if not src_nid:
                    continue
                sims = normed @ normed[i]
                pairs = [
                    (j, float(sims[j]))
                    for j in range(len(sims))
                    if j != i and float(sims[j]) >= _SIMILARITY_THRESHOLD and abs(j - i) > 1
                ]
                pairs.sort(key=lambda x: x[1], reverse=True)
                for j, score in pairs[:_MAX_EDGES_PER_CHUNK]:
                    dst_nid = chunk_node_ids.get(j)
                    if not dst_nid:
                        continue
                    try:
                        sb.rpc("upsert_valid_kg_edge", {
                            "p_src_id": src_nid, "p_dst_id": dst_nid,
                            "p_rel_type": "related_to", "p_weight": score,
                            "p_properties": {"method": "chunk_similarity"},
                        }).execute()
                        edges_created += 1
                    except Exception:
                        pass
        except Exception as e:
            warnings.append(f"Semantic edge build failed: {e}")

    # 6. Entity linking
    entities_linked = 0
    if req.entities and chunk_ids:
        embed_input = [f"{e.type}: {e.name}" for e in req.entities]
        try:
            entity_embeddings = _embed_in_batches(embed_input)
        except Exception:
            entity_embeddings = [None] * len(req.entities)

        entity_node_ids: Dict[str, str] = {}
        for ent, ent_emb in zip(req.entities, entity_embeddings):
            normalized = ent.name.lower().strip()
            ent_key = f"valid:entity:{ent.type.lower()}:{normalized}"
            try:
                ent_nid = sb.rpc("upsert_valid_kg_node", {
                    "p_node_key": ent_key,
                    "p_type": "Entity",
                    "p_name": ent.name,
                    "p_description": f"{ent.type}: {ent.name}",
                    "p_properties": {"entity_type": ent.type, **ent.properties},
                    "p_embedding": ent_emb,
                    "p_status": "active",
                    "p_embedding_model": _EMBED_MODEL,
                }).execute().data
                entity_node_ids[normalized] = str(ent_nid)
                nodes_created += 1
            except Exception as e:
                warnings.append(f"Entity '{ent.name}' node failed: {e}")

        # Mentions + co-occurrence
        for idx, (text, cid) in enumerate(zip(chunk_texts, chunk_ids)):
            text_lower = text.lower()
            chunk_nid = chunk_node_ids.get(idx)
            entities_in_chunk: List[str] = []

            for name_lower, ent_nid in entity_node_ids.items():
                if name_lower in text_lower:
                    entities_in_chunk.append(name_lower)
                    if chunk_nid:
                        try:
                            sb.rpc("upsert_valid_kg_edge", {
                                "p_src_id": chunk_nid, "p_dst_id": ent_nid,
                                "p_rel_type": "mentions", "p_weight": 1.0,
                                "p_properties": {},
                            }).execute()
                            entities_linked += 1
                            edges_created += 1
                        except Exception:
                            pass

                        # Evidence quote
                        try:
                            pos = text_lower.find(name_lower)
                            start = max(0, pos - 50)
                            end = min(len(text), pos + len(name_lower) + 50)
                            sb.table("valid_kg_node_evidence").upsert({
                                "node_id": ent_nid, "chunk_id": cid,
                                "quote": text[start:end].strip(), "score": 1.0,
                            }, on_conflict="node_id,chunk_id").execute()
                        except Exception:
                            pass

            # Co-occurrence edges
            if len(entities_in_chunk) > 1:
                for i in range(len(entities_in_chunk)):
                    for j in range(i + 1, len(entities_in_chunk)):
                        try:
                            sb.rpc("upsert_valid_kg_edge", {
                                "p_src_id": entity_node_ids[entities_in_chunk[i]],
                                "p_dst_id": entity_node_ids[entities_in_chunk[j]],
                                "p_rel_type": "co_occurs", "p_weight": 1.0,
                                "p_properties": {"source_chunk_id": cid},
                            }).execute()
                            edges_created += 1
                        except Exception:
                            pass

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "ingest.valid doc=%s chunks=%d nodes=%d edges=%d entities=%d elapsed_ms=%s",
        document_id, len(chunk_ids), nodes_created, edges_created, entities_linked, elapsed_ms,
    )

    return ValidIngestResponse(
        document_id=str(document_id),
        chunks_upserted=len(chunk_ids),
        nodes_created=nodes_created,
        edges_created=edges_created,
        entities_linked=entities_linked,
        warnings=warnings,
    )

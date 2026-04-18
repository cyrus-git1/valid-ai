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
import json
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

from src.services.audit_service import AuditService
from src.services.memory_state_service import MemoryStateService
from src.supabase.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

_EMBED_MODEL = "text-embedding-3-small"
_EMBED_BATCH_SIZE = 64
_PDF_BUCKET = "pdf"
_SIMILARITY_THRESHOLD = 0.82
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


class ProcessedDocumentRequest(BaseModel):
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


class ProcessedWebRequest(BaseModel):
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


class SummaryIngestRequest(BaseModel):
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


def _embed_texts(texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _embed_in_batches(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        out.extend(_embed_texts(texts[i:i + _EMBED_BATCH_SIZE]))
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
) -> UUID:
    existing = (
        sb.table("documents").select("id")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .eq("source_uri", source_uri)
        .limit(1).execute()
    )
    if existing.data:
        doc_id = existing.data[0]["id"]
        sb.table("documents").update({
            "source_type": source_type,
            "title": title,
            "metadata": metadata or {},
            "source_timestamp": source_timestamp.isoformat() if source_timestamp else None,
            "is_pinned": is_pinned,
            "is_canonical": is_canonical,
            "status": status,
        }).eq("id", doc_id).eq("tenant_id", str(tenant_id)).eq("client_id", str(client_id)).execute()
        return UUID(doc_id)

    res = sb.table("documents").insert({
        "tenant_id": str(tenant_id), "client_id": str(client_id),
        "source_type": source_type, "source_uri": source_uri,
        "title": title, "metadata": metadata or {},
        "source_timestamp": source_timestamp.isoformat() if source_timestamp else None,
        "is_pinned": is_pinned,
        "is_canonical": is_canonical,
        "status": status,
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

    # Upsert entity nodes
    entity_node_ids: Dict[str, UUID] = {}
    for ent, emb in zip(entities, entity_embeddings):
        normalized_name = ent.name.lower().strip()
        node_key = f"entity:{tenant_id}:{client_id}:{ent.type.lower()}:{normalized_name}"
        try:
            res = sb.rpc("upsert_kg_node", {
                "p_tenant_id": str(tenant_id), "p_client_id": str(client_id),
                "p_node_key": node_key, "p_type": "Entity",
                "p_name": ent.name, "p_description": f"{ent.type}: {ent.name}",
                "p_properties": {
                    "entity_type": ent.type,
                    "tenant_id": str(tenant_id),
                    "client_id": str(client_id),
                    **ent.properties,
                },
                "p_embedding": emb, "p_status": "active",
                "p_embedding_model": _EMBED_MODEL,
            }).execute()
            entity_node_ids[ent.name.lower().strip()] = UUID(str(res.data))
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

    # Document row
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
    )

    if not req.chunks:
        return IngestProcessedResponse(
            document_id=str(document_id), source_uri=source_uri,
            chunks_upserted=0, warnings=["No chunks provided."],
        )

    # Embed chunks
    chunk_texts = [c.text for c in req.chunks]
    try:
        embeddings = _embed_in_batches(chunk_texts)
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
    )

    if not req.chunks:
        return IngestProcessedResponse(
            document_id=str(document_id), source_uri=req.url,
            chunks_upserted=0, warnings=["No chunks provided."],
        )

    chunk_texts = [c.text for c in req.chunks]
    try:
        embeddings = _embed_in_batches(chunk_texts)
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
) -> str:
    """Create ingest_jobs row, enqueue arq task, return job_id.

    Falls back to synchronous execution if Redis/arq is unavailable.
    """
    sb = get_supabase()
    state = getattr(request, "state", None)
    key_id = getattr(state, "key_id", None) if state else None
    req_id = getattr(state, "request_id", None) if state else None

    row = sb.table("ingest_jobs").insert({
        "tenant_id": tenant_id,
        "client_id": client_id,
        "job_type": job_type,
        "status": "queued",
        "request_id": req_id,
        "key_id": key_id,
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
    )
    return IngestJobAck(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=IngestJobStatus)
def get_ingest_job(job_id: str, request: Request) -> IngestJobStatus:
    """Fetch status of an ingest job. Tenant is enforced via auth middleware."""
    sb = get_supabase()
    tenant_id = getattr(request.state, "tenant_id", None)
    q = sb.table("ingest_jobs").select("*").eq("id", job_id)
    if tenant_id:
        q = q.eq("tenant_id", str(tenant_id))
    res = q.limit(1).execute()
    rows = res.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Job not found")
    row = rows[0]
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
        embedding = _embed_texts([req.summary_text])[0]
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

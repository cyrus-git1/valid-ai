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
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from src.supabase.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

_EMBED_MODEL = "text-embedding-3-small"
_EMBED_BATCH_SIZE = 64
_PDF_BUCKET = "pdf"
_SIMILARITY_THRESHOLD = 0.82
_MAX_EDGES_PER_CHUNK = 10


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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[ChunkItem] = Field(default_factory=list)
    entities: List[EntityItem] = Field(default_factory=list)


class ProcessedWebRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    url: str
    title: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[ChunkItem] = Field(default_factory=list)
    entities: List[EntityItem] = Field(default_factory=list)


class IngestProcessedResponse(BaseModel):
    document_id: str
    source_uri: str
    chunks_upserted: int
    entities_linked: int = 0
    warnings: List[str] = Field(default_factory=list)


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


def _upload_to_bucket(sb, file_bytes: bytes, file_name: str) -> str:
    path = _sanitize_storage_key(file_name.lstrip("/"))
    sb.storage.from_(_PDF_BUCKET).upload(path, file_bytes, file_options={"upsert": "true"})
    return f"bucket:{_PDF_BUCKET}/{path}"


def _upsert_document(sb, *, tenant_id, client_id, source_type, source_uri, title, metadata) -> UUID:
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
            "source_type": source_type, "title": title, "metadata": metadata or {},
        }).eq("id", doc_id).execute()
        return UUID(doc_id)

    res = sb.table("documents").insert({
        "tenant_id": str(tenant_id), "client_id": str(client_id),
        "source_type": source_type, "source_uri": source_uri,
        "title": title, "metadata": metadata or {},
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
    }).execute()
    return UUID(str(res.data))


# -- KG build --


def _build_kg_nodes_and_edges(sb, *, tenant_id, client_id, chunk_ids, chunk_texts, chunk_embeddings):
    """Create Chunk KG nodes and cosine similarity edges."""
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
                "p_properties": {"chunk_id": str(cid)},
                "p_embedding": emb, "p_status": "active",
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

    # Similarity edges
    edges = 0
    if len(chunk_embeddings) >= 2:
        vectors = np.array(chunk_embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = vectors / norms
        sim = normed @ normed.T
        n = len(chunk_ids)

        for i in range(n):
            sims_i = sim[i].copy()
            sims_i[i] = -1.0
            cand = np.where(sims_i >= _SIMILARITY_THRESHOLD)[0]
            if cand.size == 0:
                continue
            top = cand[np.argsort(sims_i[cand])[::-1]][:_MAX_EDGES_PER_CHUNK]
            src_nid = chunk_id_to_node_id.get(str(chunk_ids[i]))
            if not src_nid:
                continue
            for j in top:
                dst_nid = chunk_id_to_node_id.get(str(chunk_ids[j]))
                if not dst_nid:
                    continue
                try:
                    sb.rpc("upsert_kg_edge", {
                        "p_tenant_id": str(tenant_id), "p_client_id": str(client_id),
                        "p_src_id": str(src_nid), "p_dst_id": str(dst_nid),
                        "p_rel_type": "related_to", "p_weight": float(sims_i[j]),
                        "p_properties": {"method": "chunk_embedding_cosine"},
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
        node_key = f"entity:{tenant_id}:{ent.name.lower().strip()}"
        try:
            res = sb.rpc("upsert_kg_node", {
                "p_tenant_id": str(tenant_id), "p_client_id": str(client_id),
                "p_node_key": node_key, "p_type": "Entity",
                "p_name": ent.name, "p_description": f"{ent.type}: {ent.name}",
                "p_properties": {"entity_type": ent.type, **ent.properties},
                "p_embedding": emb, "p_status": "active",
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


@router.post("/processed", response_model=IngestProcessedResponse)
def ingest_processed(req: ProcessedDocumentRequest) -> IngestProcessedResponse:
    """
    Store a pre-processed document from the agent service.

    The agent has already parsed, chunked, filtered, and extracted entities.
    This endpoint handles: file storage, document row, embedding, chunk storage,
    KG build, and entity linking.
    """
    sb = get_supabase()
    warnings: List[str] = []

    # Decode and upload file
    try:
        file_bytes = base64.b64decode(req.file_bytes_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    source_uri = _upload_to_bucket(sb, file_bytes, req.file_name)

    # Document row
    document_id = _upsert_document(
        sb, tenant_id=req.tenant_id, client_id=req.client_id,
        source_type=req.source_type, source_uri=source_uri,
        title=req.title, metadata={
            **req.metadata, "file_name": req.file_name, "file_type": req.source_type,
        },
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
                metadata={"source_uri": source_uri, "source_type": req.source_type},
                embedding=emb,
            )
            chunk_ids.append(cid)
        except Exception as e:
            warnings.append(f"Chunk {idx} failed: {e}")

    # KG build
    try:
        nodes, edges = _build_kg_nodes_and_edges(
            sb, tenant_id=req.tenant_id, client_id=req.client_id,
            chunk_ids=chunk_ids, chunk_texts=chunk_texts, chunk_embeddings=embeddings,
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

    return IngestProcessedResponse(
        document_id=str(document_id), source_uri=source_uri,
        chunks_upserted=len(chunk_ids), entities_linked=entities_linked,
        warnings=warnings,
    )


@router.post("/processed-web", response_model=IngestProcessedResponse)
def ingest_processed_web(req: ProcessedWebRequest) -> IngestProcessedResponse:
    """
    Store a pre-processed web scrape from the agent service.

    Same as /ingest/processed but no file upload — source_uri is the URL.
    """
    sb = get_supabase()
    warnings: List[str] = []

    document_id = _upsert_document(
        sb, tenant_id=req.tenant_id, client_id=req.client_id,
        source_type="web", source_uri=req.url,
        title=req.title, metadata=req.metadata,
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
                metadata={"source_uri": req.url, "source_type": "web"},
                embedding=emb,
            )
            chunk_ids.append(cid)
        except Exception as e:
            warnings.append(f"Chunk {idx} failed: {e}")

    try:
        nodes, edges = _build_kg_nodes_and_edges(
            sb, tenant_id=req.tenant_id, client_id=req.client_id,
            chunk_ids=chunk_ids, chunk_texts=chunk_texts, chunk_embeddings=embeddings,
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

    return IngestProcessedResponse(
        document_id=str(document_id), source_uri=req.url,
        chunks_upserted=len(chunk_ids), entities_linked=entities_linked,
        warnings=warnings,
    )

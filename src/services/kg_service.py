"""
src/services/kg_service.py
---------------------------
Builds and maintains the Knowledge Graph from chunk embeddings.

Responsibilities
----------------
  - upsert_node / upsert_edge  — thin wrappers around SQL RPCs
  - prune                       — archive stale nodes/edges, trim evidence
  - build_kg_from_chunk_embeddings — full KG build pipeline:
      fetch chunks → validate embeddings → upsert nodes → draw similarity edges

Import
------
    from src.services.kg_service import KGService, KGBuildConfig
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
from supabase import Client

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

_VALID_NODE_TYPES = frozenset({
    "WebPage", "PDF", "Image", "PowerPoint", "Docx",
    "VideoTranscript", "ChatTranscript", "ChatSnapshot", "Chunk",
})

_EMBEDDING_DIM = 1536


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return (n, n) cosine similarity matrix for an (n, d) array of vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v = vectors / norms
    return v @ v.T


def _safe_preview(text: str, max_len: int = 80) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:max_len] + ("…" if len(text) > max_len else "")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KGBuildConfig:
    similarity_threshold: float = 0.82
    max_edges_per_chunk: int = 10
    max_chunks: int = 2000
    batch_size: int = 500
    rel_type: str = "related_to"
    edge_properties: Optional[JsonDict] = None


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class KGService:
    def __init__(self, supabase: Client):
        self.sb = supabase

    # ── Node / edge RPCs ──────────────────────────────────────────────────────

    def upsert_node(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        node_key: str,
        type_value: str,
        name: str,
        description: Optional[str] = None,
        properties: Optional[JsonDict] = None,
        embedding: Optional[List[float]] = None,
        status: str = "active",
    ) -> UUID:
        if type_value not in _VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node type '{type_value}'. Must be one of: {sorted(_VALID_NODE_TYPES)}"
            )
        res = self.sb.rpc(
            "upsert_kg_node",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_node_key": node_key,
                "p_type": type_value,
                "p_name": name,
                "p_description": description,
                "p_properties": properties or {},
                "p_embedding": embedding,
                "p_status": status,
            },
        ).execute()
        return UUID(str(res.data))

    def upsert_edge(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        src_id: UUID,
        dst_id: UUID,
        rel_type: str,
        weight: Optional[float] = None,
        properties: Optional[JsonDict] = None,
    ) -> UUID:
        res = self.sb.rpc(
            "upsert_kg_edge",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_src_id": str(src_id),
                "p_dst_id": str(dst_id),
                "p_rel_type": rel_type,
                "p_weight": weight,
                "p_properties": properties or {},
            },
        ).execute()
        return UUID(str(res.data))

    # ── Pruning ───────────────────────────────────────────────────────────────

    def prune(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        edge_stale_days: int = 90,
        node_stale_days: int = 180,
        min_degree: int = 3,
        keep_edge_evidence: int = 5,
        keep_node_evidence: int = 10,
    ) -> JsonDict:
        res = self.sb.rpc(
            "prune_kg",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_edge_stale_days": edge_stale_days,
                "p_node_stale_days": node_stale_days,
                "p_min_degree": min_degree,
                "p_keep_edge_evidence": keep_edge_evidence,
                "p_keep_node_evidence": keep_node_evidence,
            },
        ).execute()
        return res.data or {}

    # ── Chunk fetching ────────────────────────────────────────────────────────

    def fetch_chunks_with_embeddings(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        document_id: Optional[UUID] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[JsonDict]:
        """
        Server-side JOIN via SQL RPC (09b_fetch_chunks_rpc.sql).
        Returns chunks that have embeddings, scoped to tenant + client.
        """
        res = self.sb.rpc(
            "fetch_chunks_with_embeddings",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_document_id": str(document_id) if document_id else None,
                "p_limit": limit,
                "p_offset": offset,
            },
        ).execute()
        return res.data or []

    def _fetch_all_chunks_paginated(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        document_id: Optional[UUID],
        cfg: KGBuildConfig,
    ) -> List[JsonDict]:
        chunks: List[JsonDict] = []
        offset = 0
        while True:
            batch = self.fetch_chunks_with_embeddings(
                tenant_id=tenant_id,
                client_id=client_id,
                document_id=document_id,
                limit=cfg.batch_size,
                offset=offset,
            )
            if not batch:
                break
            chunks.extend(batch)
            logger.debug("Fetched %d chunks (total: %d, offset: %d)", len(batch), len(chunks), offset)
            if len(chunks) >= cfg.max_chunks:
                logger.warning("Reached max_chunks limit (%d). Truncating.", cfg.max_chunks)
                chunks = chunks[: cfg.max_chunks]
                break
            if len(batch) < cfg.batch_size:
                break
            offset += cfg.batch_size
        return chunks

    # ── KG build ──────────────────────────────────────────────────────────────

    def build_kg_from_chunk_embeddings(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        document_id: Optional[UUID] = None,
        config: Optional[KGBuildConfig] = None,
    ) -> Dict[str, Any]:
        """
        Full KG build pipeline:
          1. Fetch all embedded chunks (paginated)
          2. Filter to valid 1536-dim embeddings
          3. Upsert one KG node per chunk
          4. Draw cosine-similarity edges between nodes above threshold

        Returns a summary dict with counts.
        """
        cfg = config or KGBuildConfig()

        all_chunks = self._fetch_all_chunks_paginated(
            tenant_id=tenant_id,
            client_id=client_id,
            document_id=document_id,
            cfg=cfg,
        )

        if not all_chunks:
            return {
                "chunks_fetched": 0,
                "chunks_valid": 0,
                "chunks_skipped": 0,
                "nodes_upserted": 0,
                "edges_upserted": 0,
                "note": "No embedded chunks found.",
            }

        # Validate embeddings
        valid_chunks: List[JsonDict] = []
        valid_embeddings: List[List[float]] = []
        skipped = 0

        for c in all_chunks:
            emb = c.get("embedding")
            if isinstance(emb, list) and len(emb) == _EMBEDDING_DIM:
                valid_chunks.append(c)
                valid_embeddings.append(emb)
            else:
                skipped += 1
                logger.warning(
                    "Skipping chunk %s — bad embedding (got %s, expected %d).",
                    c.get("id"),
                    len(emb) if isinstance(emb, list) else type(emb).__name__,
                    _EMBEDDING_DIM,
                )

        if not valid_chunks:
            return {
                "chunks_fetched": len(all_chunks),
                "chunks_valid": 0,
                "chunks_skipped": skipped,
                "nodes_upserted": 0,
                "edges_upserted": 0,
                "note": "No chunks had valid embeddings.",
            }

        vectors = np.array(valid_embeddings, dtype=np.float32)

        # 1) Upsert chunk nodes
        chunk_id_to_node_id: Dict[str, UUID] = {}
        nodes_upserted = 0

        for c in valid_chunks:
            chunk_id = c["id"]
            node_id = self.upsert_node(
                tenant_id=tenant_id,
                client_id=client_id,
                node_key=f"chunk:{chunk_id}",
                type_value="Chunk",
                name=f"Chunk {c.get('chunk_index', 0)}",
                description=_safe_preview(c.get("content", "")),
                properties={
                    "chunk_id": chunk_id,
                    "document_id": c.get("document_id"),
                    "chunk_index": c.get("chunk_index"),
                    "metadata": c.get("metadata") or {},
                },
                embedding=c.get("embedding"),
                status="active",
            )
            chunk_id_to_node_id[chunk_id] = node_id
            nodes_upserted += 1

        # 2) Similarity edges
        sim = _cosine_sim_matrix(vectors)
        edges_upserted = 0
        n = len(valid_chunks)

        for i in range(n):
            sims_i = sim[i].copy()
            sims_i[i] = -1.0

            cand_idx = np.where(sims_i >= cfg.similarity_threshold)[0]
            if cand_idx.size == 0:
                continue

            cand_sorted = cand_idx[np.argsort(sims_i[cand_idx])[::-1]][: cfg.max_edges_per_chunk]
            src_node_id = chunk_id_to_node_id[valid_chunks[i]["id"]]

            for j in cand_sorted:
                dst_node_id = chunk_id_to_node_id[valid_chunks[j]["id"]]
                self.upsert_edge(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    src_id=src_node_id,
                    dst_id=dst_node_id,
                    rel_type=cfg.rel_type,
                    weight=float(sims_i[j]),
                    properties={
                        **(cfg.edge_properties or {}),
                        "method": "chunk_embedding_cosine",
                        "threshold": cfg.similarity_threshold,
                    },
                )
                edges_upserted += 1

        return {
            "chunks_fetched": len(all_chunks),
            "chunks_valid": len(valid_chunks),
            "chunks_skipped": skipped,
            "nodes_upserted": nodes_upserted,
            "edges_upserted": edges_upserted,
            "similarity_threshold": cfg.similarity_threshold,
            "max_edges_per_chunk": cfg.max_edges_per_chunk,
        }

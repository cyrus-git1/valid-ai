"""
src/services/kg_retriever_service.py
-------------------------------------
SupabaseKGRetriever — LangChain BaseRetriever backed by the Supabase KG.

Retrieval strategy
------------------
  1. Embed the query with OpenAI text-embedding-3-small
  2. Call search_kg_nodes RPC — pgvector similarity search over kg_nodes
  3. For each seed node, walk outgoing edges (graph expansion)
     to pull in structurally related nodes
  4. Deduplicate, annotate source ("vector" vs "graph_expansion")
  5. Fetch full chunk text from chunks table (node description is 80-char preview only)
  6. Return as LangChain Documents

SQL RPCs required
-----------------
  search_kg_nodes             — kg_search_rpc.sql
  fetch_chunks_with_embeddings — 09b_fetch_chunks_rpc.sql

Import
------
    from src.services.kg_retriever_service import KGRetrieverService

    retriever = KGRetrieverService.from_env(
        tenant_id=uuid.UUID("..."),
        client_id=uuid.UUID("..."),
    )
    docs = retriever.invoke("What is the return policy?")
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

import dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from supabase import Client, create_client

from src.models.api.retrieval import KGRetrieverConfig

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


class KGRetrieverService(KGRetrieverConfig, BaseRetriever):
    """
    LangChain-compatible retriever over the Supabase Knowledge Graph.

    Each returned Document has:
        page_content  — full chunk text (fetched from chunks table)
        metadata      — node_id, node_key, node_type, document_id,
                        chunk_id, chunk_index, similarity_score, source

    Quick start
    -----------
        retriever = KGRetrieverService.from_env(
            tenant_id=uuid.UUID("your-tenant-id"),
            client_id=uuid.UUID("your-client-id"),
            top_k=5,
            hop_limit=1,
        )
        docs = retriever.invoke("What is the refund policy?")
    """

    # ── Pydantic fields ───────────────────────────────────────────────────────

    # ── Private clients ───────────────────────────────────────────────────────
    _sb: Optional[Client] = None
    _embeddings: Optional[OpenAIEmbeddings] = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        self._sb = create_client(self.supabase_url, self.supabase_key)
        self._embeddings = OpenAIEmbeddings(
            model=self.embed_model,
            api_key=self.openai_api_key,
        )
        logger.debug(
            "KGRetrieverService ready — tenant=%s client=%s top_k=%d hop_limit=%d",
            self.tenant_id, self.client_id, self.top_k, self.hop_limit,
        )

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> List[float]:
        return self._embeddings.embed_query(query)

    # ── Vector search ─────────────────────────────────────────────────────────

    def _vector_search(self, embedding: List[float]) -> List[JsonDict]:
        """
        Call the search_kg_nodes SQL RPC (kg_search_rpc.sql).
        Returns rows with: id, node_key, name, description, properties, type, similarity
        """
        try:
            params = {
                "p_tenant_id": str(self.tenant_id),
                "p_client_id": str(self.client_id) if self.client_id else None,
                "p_embedding": embedding,
                "p_top_k": self.top_k,
                "p_types": self.node_types,
                "p_document_ids": self.document_ids,
                "p_recency_weight": self.recency_weight,
                "p_boost_pinned": self.boost_pinned,
                "p_embedding_model": self.embed_model,
            }
            if self.exclude_status is not None:
                params["p_exclude_status"] = self.exclude_status
            if self.source_types is not None:
                params["p_source_types"] = self.source_types
            res = self._sb.rpc("search_kg_nodes", params).execute()
            return res.data or []
        except Exception as e:
            logger.error("search_kg_nodes RPC failed: %s", e)
            return []

    # ── Graph expansion ───────────────────────────────────────────────────────

    def _get_neighbour_ids(self, node_id: str) -> List[str]:
        """Fetch connected nodes via both outgoing and incoming edges above min_edge_weight."""
        neighbours: List[str] = []
        try:
            # Outgoing edges (src_id = this node)
            out_q = (
                self._sb.table("kg_edges")
                .select("dst_id, weight")
                .eq("tenant_id", str(self.tenant_id))
                .eq("src_id", node_id)
                .eq("is_active", True)
                .gte("weight", self.min_edge_weight)
            )
            if self.client_id:
                out_q = out_q.eq("client_id", str(self.client_id))
            if self.rel_types:
                out_q = out_q.in_("rel_type", self.rel_types)
            out_res = out_q.order("weight", desc=True).limit(self.max_neighbours).execute()
            neighbours.extend(row["dst_id"] for row in (out_res.data or []))

            # Incoming edges (dst_id = this node) — e.g. chunk→entity "mentions" edges
            in_q = (
                self._sb.table("kg_edges")
                .select("src_id, weight")
                .eq("tenant_id", str(self.tenant_id))
                .eq("dst_id", node_id)
                .eq("is_active", True)
                .gte("weight", self.min_edge_weight)
            )
            if self.client_id:
                in_q = in_q.eq("client_id", str(self.client_id))
            if self.rel_types:
                in_q = in_q.in_("rel_type", self.rel_types)
            in_res = in_q.order("weight", desc=True).limit(self.max_neighbours).execute()
            neighbours.extend(row["src_id"] for row in (in_res.data or []))
        except Exception as e:
            logger.error("Edge fetch failed for node %s: %s", node_id, e)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for nid in neighbours:
            if nid not in seen:
                seen.add(nid)
                unique.append(nid)
        return unique[:self.max_neighbours]

    def _fetch_nodes_by_ids(self, node_ids: List[str]) -> List[JsonDict]:
        """Batch fetch active node rows by ID list, filtered by document scope if set."""
        if not node_ids:
            return []
        try:
            res = (
                self._sb.table("kg_nodes")
                .select("id, node_key, name, description, properties, type")
                .in_("id", node_ids)
                .eq("tenant_id", str(self.tenant_id))
                .eq("status", "active")
            )
            if self.client_id:
                res = res.eq("client_id", str(self.client_id))
            res = res.execute()
            rows = res.data or []

            # Filter expanded nodes to document scope (if set)
            if self.document_ids and rows:
                doc_set = set(self.document_ids)
                rows = [
                    r for r in rows
                    if (r.get("properties") or {}).get("document_id") in doc_set
                    or (r.get("type") == "Entity")  # entities are cross-document, always include
                ]

            return rows
        except Exception as e:
            logger.error("Node batch fetch failed: %s", e)
            return []

    # ── Chunk content ─────────────────────────────────────────────────────────

    def _get_chunk_content(self, chunk_id: str, document_id: Optional[str]) -> Optional[str]:
        """
        Fetch full chunk text from the chunks table.
        Node description is only an 80-char preview — LLM needs the full text.
        """
        try:
            q = (
                self._sb.table("chunks")
                .select("content")
                .eq("id", chunk_id)
                .eq("tenant_id", str(self.tenant_id))
            )
            if document_id:
                q = q.eq("document_id", document_id)
            res = q.limit(1).execute()
            if res.data:
                return res.data[0]["content"]
        except Exception as e:
            logger.warning("Chunk content fetch failed for %s: %s", chunk_id, e)
        return None

    # ── Evidence fetching ────────────────────────────────────────────────────

    def _get_node_evidence(self, node_id: str) -> List[JsonDict]:
        """Fetch evidence rows for a node from kg_node_evidence."""
        try:
            res = (
                self._sb.table("kg_node_evidence")
                .select("chunk_id, quote, score")
                .eq("tenant_id", str(self.tenant_id))
                .eq("node_id", node_id)
                .order("score", desc=True)
                .limit(5)
            )
            if self.client_id:
                res = res.eq("client_id", str(self.client_id))
            res = res.execute()
            return res.data or []
        except Exception as e:
            logger.warning("Node evidence fetch failed for %s: %s", node_id, e)
            return []

    # ── Node → Document ───────────────────────────────────────────────────────

    def _node_to_document(
        self,
        node: JsonDict,
        similarity: Optional[float] = None,
        final_score: Optional[float] = None,
        source_type: Optional[str] = None,
        source: str = "vector",
        retrieval_reason: str = "",
    ) -> Document:
        props = node.get("properties") or {}
        chunk_id = props.get("chunk_id")
        node_type = node.get("type", "")
        entity_type = props.get("entity_type")

        # Fetch evidence for richer context
        node_id = node.get("id", "")
        evidence_rows = self._get_node_evidence(node_id)
        evidence_quote = evidence_rows[0]["quote"] if evidence_rows else None
        evidence_score = evidence_rows[0]["score"] if evidence_rows else None

        # Compose content based on node type
        if entity_type and not chunk_id:
            # Entity node — compose from name + description + evidence quotes
            parts = [f"{entity_type}: {node.get('name', '')}"]
            desc = node.get("description")
            if desc and desc != f"{entity_type}: {node.get('name', '')}":
                parts.append(desc)
            if evidence_rows:
                parts.append("\nEvidence:")
                for ev in evidence_rows:
                    quote = ev.get("quote", "").strip()
                    if quote:
                        parts.append(f"- {quote}")
            content = "\n".join(parts)
        else:
            # Chunk node — fetch full text from chunks table
            content = (self._get_chunk_content(chunk_id, props.get("document_id")) if chunk_id else None) \
                or node.get("description") \
                or node.get("name") \
                or ""

        metadata: JsonDict = {
            "node_id": node_id,
            "node_key": node.get("node_key"),
            "node_type": node_type,
            "entity_type": entity_type,
            "document_id": props.get("document_id"),
            "chunk_id": chunk_id,
            "chunk_index": props.get("chunk_index"),
            "source": source,
            "retrieval_reason": retrieval_reason,
            "evidence_quote": evidence_quote,
            "evidence_score": evidence_score,
            "evidence_count": len(evidence_rows),
        }
        if similarity is not None:
            metadata["similarity_score"] = round(float(similarity), 4)
        if final_score is not None:
            metadata["final_score"] = round(float(final_score), 4)
        if source_type is not None:
            metadata["source_type"] = source_type

        return Document(page_content=content, metadata=metadata)

    # ── BaseRetriever interface ───────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Core LangChain retrieval method.

        1. Embed query
        2. Vector search → seed nodes
        3. Multi-hop graph expansion (up to hop_limit hops)
           - Hop 1: seed → neighbors (similar chunks + entities)
           - Hop 2: neighbors → their neighbors (entity bridging:
             chunk → entity → other chunks mentioning same entity)
        4. Deduplicate → Documents
        """
        logger.debug("Retrieving for query: %r (hop_limit=%d)", query[:80], self.hop_limit)

        embedding = self._embed_query(query)
        seed_nodes = self._vector_search(embedding)
        logger.debug("Vector search returned %d seed nodes", len(seed_nodes))

        seen_ids: set[str] = set()
        documents: List[Document] = []

        # Add seed nodes
        for node in seed_nodes:
            nid = node["id"]
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            sim = node.get("similarity")
            fscore = node.get("final_score")
            reason = f"Vector similarity={sim:.4f}" if sim is not None else "Vector match"
            logger.info(
                "[Retrieval] SEED node=%s key=%s sim=%.4f final=%.4f reason=%s",
                nid, node.get("node_key", "?"), sim or 0, fscore or 0, reason,
            )
            documents.append(self._node_to_document(
                node,
                similarity=sim,
                final_score=fscore,
                source_type=node.get("source_type"),
                source="vector",
                retrieval_reason=reason,
            ))

        # Multi-hop expansion
        if self.hop_limit >= 1:
            # Start with seed node IDs as the frontier
            frontier_ids: List[str] = [n["id"] for n in seed_nodes]

            for hop in range(1, self.hop_limit + 1):
                next_frontier: List[str] = []

                for nid in frontier_ids:
                    neighbour_ids = [
                        n for n in self._get_neighbour_ids(nid)
                        if n not in seen_ids
                    ]
                    if not neighbour_ids:
                        continue

                    neighbours = self._fetch_nodes_by_ids(neighbour_ids)
                    for nb in neighbours:
                        nb_id = nb["id"]
                        if nb_id in seen_ids:
                            continue
                        seen_ids.add(nb_id)
                        next_frontier.append(nb_id)

                        exp_reason = (
                            f"Hop {hop} from {nid[:8]}… "
                            f"(edge weight >= {self.min_edge_weight})"
                        )
                        logger.info(
                            "[Retrieval] HOP-%d node=%s key=%s type=%s via=%s",
                            hop, nb_id, nb.get("node_key", "?")[:30],
                            nb.get("type", "?"), nid[:8],
                        )
                        documents.append(self._node_to_document(
                            nb, source=f"graph_hop_{hop}",
                            retrieval_reason=exp_reason,
                        ))

                frontier_ids = next_frontier
                if not frontier_ids:
                    logger.debug("Hop %d: no new frontier, stopping", hop)
                    break

        logger.info(
            "[Retrieval] Complete: %d documents (%d seed + %d expanded, %d hops) for query=%r",
            len(documents), len(seed_nodes), len(documents) - len(seed_nodes),
            self.hop_limit, query[:80],
        )
        return documents

    # ── Convenience constructor ───────────────────────────────────────────────

    @classmethod
    def from_env(cls, tenant_id: UUID, client_id: Optional[UUID] = None, **kwargs) -> "KGRetrieverService":
        """
        Construct from environment variables (SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY).

        Example
        -------
            retriever = KGRetrieverService.from_env(
                tenant_id=uuid.UUID("..."),
                client_id=uuid.UUID("..."),
                top_k=8,
                hop_limit=1,
            )
        """
        return cls(tenant_id=tenant_id, client_id=client_id, **kwargs)

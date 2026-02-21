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
import os
from typing import Any, Dict, List, Optional
from uuid import UUID

import dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from pydantic import Field
from supabase import Client, create_client

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


class KGRetrieverService(BaseRetriever):
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
    supabase_url: str = Field(default_factory=lambda: os.environ["SUPABASE_URL"])
    supabase_key: str = Field(default_factory=lambda: os.environ["SUPABASE_SERVICE_KEY"])
    openai_api_key: str = Field(default_factory=lambda: os.environ["OPENAI_API_KEY"])

    tenant_id: UUID
    client_id: UUID

    top_k: int = Field(default=5, description="Seed nodes from vector search")
    hop_limit: int = Field(default=1, description="Graph expansion hops (0 = vector only)")
    max_neighbours: int = Field(default=3, description="Max neighbours pulled per seed node")
    min_edge_weight: float = Field(default=0.75, description="Min edge weight to follow")

    embed_model: str = "text-embedding-3-small"

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
            res = self._sb.rpc(
                "search_kg_nodes",
                {
                    "p_tenant_id": str(self.tenant_id),
                    "p_client_id": str(self.client_id),
                    "p_embedding": embedding,
                    "p_top_k": self.top_k,
                },
            ).execute()
            return res.data or []
        except Exception as e:
            logger.error("search_kg_nodes RPC failed: %s", e)
            return []

    # ── Graph expansion ───────────────────────────────────────────────────────

    def _get_neighbour_ids(self, node_id: str) -> List[str]:
        """Fetch outgoing edge targets above min_edge_weight, ordered by weight desc."""
        try:
            res = (
                self._sb.table("kg_edges")
                .select("dst_id, weight")
                .eq("tenant_id", str(self.tenant_id))
                .eq("client_id", str(self.client_id))
                .eq("src_id", node_id)
                .eq("is_active", True)
                .gte("weight", self.min_edge_weight)
                .order("weight", desc=True)
                .limit(self.max_neighbours)
                .execute()
            )
            return [row["dst_id"] for row in (res.data or [])]
        except Exception as e:
            logger.error("Edge fetch failed for node %s: %s", node_id, e)
            return []

    def _fetch_nodes_by_ids(self, node_ids: List[str]) -> List[JsonDict]:
        """Batch fetch active node rows by ID list."""
        if not node_ids:
            return []
        try:
            res = (
                self._sb.table("kg_nodes")
                .select("id, node_key, name, description, properties, type")
                .in_("id", node_ids)
                .eq("tenant_id", str(self.tenant_id))
                .eq("status", "active")
                .execute()
            )
            return res.data or []
        except Exception as e:
            logger.error("Node batch fetch failed: %s", e)
            return []

    # ── Chunk content ─────────────────────────────────────────────────────────

    def _get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """
        Fetch full chunk text from the chunks table.
        Node description is only an 80-char preview — LLM needs the full text.
        """
        try:
            res = (
                self._sb.table("chunks")
                .select("content")
                .eq("id", chunk_id)
                .eq("tenant_id", str(self.tenant_id))
                .limit(1)
                .execute()
            )
            if res.data:
                return res.data[0]["content"]
        except Exception as e:
            logger.warning("Chunk content fetch failed for %s: %s", chunk_id, e)
        return None

    # ── Node → Document ───────────────────────────────────────────────────────

    def _node_to_document(
        self,
        node: JsonDict,
        similarity: Optional[float] = None,
        source: str = "vector",
    ) -> Document:
        props = node.get("properties") or {}
        chunk_id = props.get("chunk_id")

        content = (self._get_chunk_content(chunk_id) if chunk_id else None) \
            or node.get("description") \
            or node.get("name") \
            or ""

        metadata: JsonDict = {
            "node_id": node.get("id"),
            "node_key": node.get("node_key"),
            "node_type": node.get("type"),
            "document_id": props.get("document_id"),
            "chunk_id": chunk_id,
            "chunk_index": props.get("chunk_index"),
            "source": source,
        }
        if similarity is not None:
            metadata["similarity_score"] = round(float(similarity), 4)

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
        3. Graph expansion per seed (if hop_limit >= 1)
        4. Deduplicate → Documents
        """
        logger.debug("Retrieving for query: %r", query[:80])

        embedding = self._embed_query(query)
        seed_nodes = self._vector_search(embedding)
        logger.debug("Vector search returned %d seed nodes", len(seed_nodes))

        seen_ids: set[str] = set()
        documents: List[Document] = []

        for node in seed_nodes:
            nid = node["id"]
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            documents.append(self._node_to_document(node, similarity=node.get("similarity"), source="vector"))

            if self.hop_limit >= 1:
                neighbour_ids = [n for n in self._get_neighbour_ids(nid) if n not in seen_ids]
                for nb in self._fetch_nodes_by_ids(neighbour_ids):
                    nb_id = nb["id"]
                    if nb_id not in seen_ids:
                        seen_ids.add(nb_id)
                        documents.append(self._node_to_document(nb, source="graph_expansion"))

        logger.debug(
            "Returning %d documents (%d seed + %d expanded)",
            len(documents), len(seed_nodes), len(documents) - len(seed_nodes),
        )
        return documents

    # ── Convenience constructor ───────────────────────────────────────────────

    @classmethod
    def from_env(cls, tenant_id: UUID, client_id: UUID, **kwargs) -> "KGRetrieverService":
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

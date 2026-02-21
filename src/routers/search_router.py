"""
/search router
--------------
Query-time endpoints. Three tiers of retrieval, each building on the previous.

POST /search/semantic  — Pure vector search over kg_nodes via pgvector
POST /search/graph     — Vector search + one-hop edge expansion (graph RAG)
POST /search/ask       — Full RAG: graph retrieval + LLM answer generation
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from src.models.api.search import (
    AskRequest,
    AskResponse,
    GraphSearchRequest,
    GraphSearchResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SearchResultItem,
)
from src.services.search_service import SearchService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


def _docs_to_result_items(docs) -> List[SearchResultItem]:
    items = []
    for doc in docs:
        m = doc.metadata
        items.append(SearchResultItem(
            node_id=m.get("node_id", ""),
            node_key=m.get("node_key", ""),
            node_type=m.get("node_type", ""),
            content=doc.page_content,
            similarity_score=m.get("similarity_score"),
            document_id=m.get("document_id"),
            chunk_index=m.get("chunk_index"),
            source=m.get("source", "vector"),
        ))
    return items


@router.post("/semantic", response_model=SemanticSearchResponse)
def semantic_search(req: SemanticSearchRequest) -> SemanticSearchResponse:
    """
    Pure vector similarity search over KG nodes.

    Embeds the query with OpenAI, finds the top-k most similar chunk nodes
    using the pgvector HNSW index on kg_nodes.embedding.

    Fast and cheap. Use this for simple lookups.
    No graph traversal — only direct embedding similarity.
    """
    svc = SearchService(tenant_id=req.tenant_id, client_id=req.client_id)

    try:
        docs = svc.semantic_search(req.query, top_k=req.top_k)
    except Exception as e:
        logger.exception("Semantic search failed")
        raise HTTPException(status_code=500, detail=str(e))

    return SemanticSearchResponse(
        query=req.query,
        results=_docs_to_result_items(docs),
    )


@router.post("/graph", response_model=GraphSearchResponse)
def graph_search(req: GraphSearchRequest) -> GraphSearchResponse:
    """
    Vector search + graph expansion retrieval.

    Step 1 — embed query and find top-k similar chunk nodes (seeds).
    Step 2 — for each seed, follow outgoing KG edges (cosine similarity >=
             min_edge_weight) and pull in neighbouring nodes.

    The graph expansion surfaces related chunks that might not be individually
    close to the query but are structurally connected to chunks that are.
    """
    svc = SearchService(tenant_id=req.tenant_id, client_id=req.client_id)

    try:
        docs = svc.graph_search(
            req.query,
            top_k=req.top_k,
            hop_limit=req.hop_limit,
            max_neighbours=req.max_neighbours,
            min_edge_weight=req.min_edge_weight,
        )
    except Exception as e:
        logger.exception("Graph search failed")
        raise HTTPException(status_code=500, detail=str(e))

    seed_count = min(req.top_k, len(docs))
    expanded_count = len(docs) - seed_count

    return GraphSearchResponse(
        query=req.query,
        results=_docs_to_result_items(docs),
        seed_nodes=seed_count,
        expanded_nodes=expanded_count,
    )


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    Full RAG pipeline: graph retrieval + LLM answer generation.

    Step 1 — graph-expanded retrieval (same as /search/graph)
    Step 2 — concatenate retrieved chunk texts as context
    Step 3 — prompt GPT-4o-mini (or configured model) to answer from context only

    Returns the answer plus the source chunks used to generate it,
    so callers can show citations.
    """
    svc = SearchService(
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        llm_model=req.model,
    )

    try:
        answer, docs = svc.ask(
            req.question,
            top_k=req.top_k,
            hop_limit=req.hop_limit,
        )
    except Exception as e:
        logger.exception("RAG pipeline failed in /ask")
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}")

    return AskResponse(
        question=req.question,
        answer=answer,
        sources=_docs_to_result_items(docs),
    )

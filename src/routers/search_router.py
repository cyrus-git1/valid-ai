"""
/search router
--------------
Query-time endpoints. Vector search over the KG.

POST /search/semantic  — Pure vector search over kg_nodes via pgvector
POST /search/graph     — Vector search + one-hop edge expansion (graph RAG)

The /search/ask endpoint (full RAG with LLM) has moved to the agent service.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from src.models.api.search import (
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
            entity_type=m.get("entity_type"),
            content=doc.page_content,
            similarity_score=m.get("similarity_score"),
            document_id=m.get("document_id"),
            chunk_index=m.get("chunk_index"),
            source=m.get("source", "vector"),
            retrieval_reason=m.get("retrieval_reason"),
            evidence_quote=m.get("evidence_quote"),
            evidence_score=m.get("evidence_score"),
            evidence_count=m.get("evidence_count", 0),
        ))
    return items


@router.post("/semantic", response_model=SemanticSearchResponse)
def semantic_search(req: SemanticSearchRequest) -> SemanticSearchResponse:
    """Pure vector similarity search over KG nodes."""
    svc = SearchService(tenant_id=req.tenant_id, client_id=req.client_id)

    try:
        docs = svc.semantic_search(req.query, top_k=req.top_k, node_types=req.node_types)
    except Exception as e:
        logger.exception("Semantic search failed")
        raise HTTPException(status_code=500, detail=str(e))

    return SemanticSearchResponse(
        query=req.query,
        results=_docs_to_result_items(docs),
    )


@router.post("/graph", response_model=GraphSearchResponse)
def graph_search(req: GraphSearchRequest) -> GraphSearchResponse:
    """Vector search + graph expansion retrieval."""
    svc = SearchService(tenant_id=req.tenant_id, client_id=req.client_id)

    try:
        docs = svc.graph_search(
            req.query,
            top_k=req.top_k,
            hop_limit=req.hop_limit,
            max_neighbours=req.max_neighbours,
            min_edge_weight=req.min_edge_weight,
            node_types=req.node_types,
            rel_types=req.rel_types,
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

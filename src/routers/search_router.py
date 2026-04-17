"""
/search router
--------------
POST /search/graph — Vector search over the KG with configurable graph expansion.

hop_limit controls depth: 0 = pure vector, 1 = one-hop, 2 = two-hop bridging.
Optional client_id scopes to a single client; omit for tenant-wide search.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import List

from fastapi import APIRouter, HTTPException

from src.models.api.search import (
    GraphSearchRequest,
    GraphSearchResponse,
    SearchResultItem,
)
from src.services.cache_service import CacheService
from src.services.search_service import SearchService

_SEARCH_CACHE_TTL = 300  # 5 minutes
_cache = CacheService()

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
            final_score=m.get("final_score"),
            source_type=m.get("source_type"),
        ))
    return items


@router.post("/graph", response_model=GraphSearchResponse)
def graph_search(req: GraphSearchRequest) -> GraphSearchResponse:
    """Vector search + configurable graph expansion.

    hop_limit controls graph expansion depth:
      0 = pure vector search (no expansion)
      1 = one-hop neighbor expansion (default)
      2 = two-hop neighbor expansion (entity bridging)
    """
    # Cache lookup — normalised payload excludes tenant/client (in key) and query state
    cache_payload = req.model_dump(mode="json", exclude={"tenant_id", "client_id"})
    payload_hash = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    cache_key = CacheService.search_key(
        tenant_id=str(req.tenant_id),
        client_id=str(req.client_id) if req.client_id else None,
        payload_hash=payload_hash,
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.info("search.graph cache=hit tenant=%s client=%s", req.tenant_id, req.client_id)
        return GraphSearchResponse(**cached)

    svc = SearchService(tenant_id=req.tenant_id, client_id=req.client_id)
    started = time.perf_counter()
    scope = "client" if req.client_id else "tenant"

    try:
        docs = svc.graph_search(
            req.query,
            top_k=req.top_k,
            hop_limit=req.hop_limit,
            max_neighbours=req.max_neighbours,
            min_edge_weight=req.min_edge_weight,
            node_types=req.node_types,
            rel_types=req.rel_types,
            document_ids=req.document_ids,
            recency_weight=req.recency_weight,
            boost_pinned=req.boost_pinned,
            exclude_status=req.exclude_status,
            source_types=req.source_types,
        )
    except Exception as e:
        logger.exception("Graph search failed")
        raise HTTPException(status_code=500, detail=str(e))

    seed_count = min(req.top_k, len(docs))
    expanded_count = len(docs) - seed_count
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "search.graph tenant=%s client=%s scope=%s hop_limit=%d top_k=%d results=%d seed=%d expanded=%d elapsed_ms=%s",
        req.tenant_id,
        req.client_id,
        scope,
        req.hop_limit,
        req.top_k,
        len(docs),
        seed_count,
        expanded_count,
        elapsed_ms,
    )

    response = GraphSearchResponse(
        query=req.query,
        results=_docs_to_result_items(docs),
        seed_nodes=seed_count,
        expanded_nodes=expanded_count,
    )
    _cache.set(cache_key, response.model_dump(mode="json"), _SEARCH_CACHE_TTL)
    return response

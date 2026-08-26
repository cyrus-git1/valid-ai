"""
src/services/search_service.py
-------------------------------
Graph-expanded retrieval over the KG. Wraps KGRetrieverService with hybrid
(dense + BM25) search, reranking, dedup, and PII redaction.

Import
------
    from src.services.search_service import SearchService
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional
from uuid import UUID

from langchain_core.documents import Document

from src.services.kg_retriever_service import KGRetrieverService

logger = logging.getLogger(__name__)


class SearchService:
    """
    Wraps the KGRetrieverService for graph-expanded retrieval.

    Usage
    -----
        svc = SearchService(tenant_id=..., client_id=...)

        # Graph-expanded search
        docs = svc.graph_search("What is the return policy?", top_k=5, hop_limit=1)
    """

    def __init__(
        self,
        tenant_id: UUID,
        client_id: Optional[UUID],
        openai_api_key: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        embed_model: str = "text-embedding-3-small",
    ):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self._api_key = openai_api_key or os.environ["OPENAI_API_KEY"]
        self._sb_url = supabase_url or os.environ["SUPABASE_URL"]
        self._sb_key = supabase_key or os.environ["SUPABASE_SERVICE_KEY"]
        self._embed_model = embed_model

    def _build_retriever(
        self,
        top_k: int,
        hop_limit: int,
        max_neighbours: int = 3,
        min_edge_weight: float = 0.75,
        node_types: Optional[List[str]] = None,
        rel_types: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        recency_weight: float = 0.0,
        boost_pinned: bool = False,
        exclude_status: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        redact_pii: bool = True,
        study_id: Optional[str] = None,
        cross_study_entities: bool = False,
        use_hybrid: bool = True,
        sparse_weight: float = 1.0,
        candidate_pool: int = 50,
        use_rerank: bool = True,
        dedup_threshold: float = 0.95,
    ) -> KGRetrieverService:
        return KGRetrieverService(
            supabase_url=self._sb_url,
            supabase_key=self._sb_key,
            openai_api_key=self._api_key,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            top_k=top_k,
            hop_limit=hop_limit,
            max_neighbours=max_neighbours,
            min_edge_weight=min_edge_weight,
            embed_model=self._embed_model,
            node_types=node_types,
            rel_types=rel_types,
            document_ids=document_ids,
            recency_weight=recency_weight,
            boost_pinned=boost_pinned,
            exclude_status=exclude_status,
            source_types=source_types,
            redact_pii=redact_pii,
            study_id=study_id,
            cross_study_entities=cross_study_entities,
            use_hybrid=use_hybrid,
            sparse_weight=sparse_weight,
            candidate_pool=candidate_pool,
            use_rerank=use_rerank,
            dedup_threshold=dedup_threshold,
        )

    def graph_search(
        self,
        query: str,
        top_k: int = 5,
        hop_limit: int = 1,
        max_neighbours: int = 3,
        min_edge_weight: float = 0.75,
        node_types: Optional[List[str]] = None,
        rel_types: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        recency_weight: float = 0.0,
        boost_pinned: bool = False,
        exclude_status: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        redact_pii: bool = True,
        study_id: Optional[str] = None,
        cross_study_entities: bool = False,
        use_hybrid: bool = True,
        sparse_weight: float = 1.0,
        candidate_pool: int = 50,
        use_rerank: bool = True,
        dedup_threshold: float = 0.95,
    ) -> List[Document]:
        """Hybrid (dense+BM25) graph search with optional rerank + dedup + PII redaction."""
        retriever = self._build_retriever(
            top_k=top_k,
            hop_limit=hop_limit,
            max_neighbours=max_neighbours,
            min_edge_weight=min_edge_weight,
            node_types=node_types,
            rel_types=rel_types,
            document_ids=document_ids,
            recency_weight=recency_weight,
            boost_pinned=boost_pinned,
            exclude_status=exclude_status,
            source_types=source_types,
            redact_pii=redact_pii,
            study_id=study_id,
            cross_study_entities=cross_study_entities,
            use_hybrid=use_hybrid,
            sparse_weight=sparse_weight,
            candidate_pool=candidate_pool,
            use_rerank=use_rerank,
            dedup_threshold=dedup_threshold,
        )
        docs = retriever.invoke(query)

        # Fold tenant context corrections into retrieved content (non-destructive).
        try:
            from src.db.supabase_client import get_supabase
            from src.services.corrections_service import CorrectionsService
            docs = CorrectionsService(get_supabase()).apply_to_documents(
                docs, self.tenant_id, self.client_id
            )
        except Exception:
            logger.warning("context-correction post-process failed; returning raw docs", exc_info=True)

        return docs

"""Configuration models for KG retrieval."""
from __future__ import annotations

import os
from typing import List, Optional

from pydantic import Field

from src.models.base import TenantOwned


class KGRetrieverConfig(TenantOwned):
    """Configuration shared by KG retrieval entry points."""

    supabase_url: str = Field(default_factory=lambda: os.environ["SUPABASE_URL"])
    supabase_key: str = Field(default_factory=lambda: os.environ["SUPABASE_SERVICE_KEY"])
    openai_api_key: str = Field(default_factory=lambda: os.environ["OPENAI_API_KEY"])

    top_k: int = Field(default=5, description="Seed nodes from vector search")
    hop_limit: int = Field(default=2, description="Graph expansion hops (0 = vector only, 2 = entity bridging)")
    max_neighbours: int = Field(default=3, description="Max neighbours pulled per seed node")
    min_edge_weight: float = Field(default=0.75, description="Min edge weight to follow")

    embed_model: str = "text-embedding-3-small"

    node_types: Optional[List[str]] = Field(
        default=None,
        description="Filter vector search to these node types (e.g. ['Entity', 'Chunk'])",
    )
    rel_types: Optional[List[str]] = Field(
        default=None,
        description="Filter edge traversal to these rel_types (e.g. ['mentions', 'co_occurs'])",
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter to chunks from these document IDs only",
    )
    source_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by parent documents.source_type (e.g. ['ContextSummary','TopicSummary'])",
    )

    # PII output redaction (Option 1: redact at output, not at storage)
    redact_pii: bool = Field(
        default=True,
        description="Apply chunks.pii_annotations to substitute PII spans with aliases on output",
    )

    # Hybrid ranking
    recency_weight: float = Field(default=0.0, description="0.0 = pure vector, 0.3 = 30% recency blend")
    boost_pinned: bool = Field(default=False, description="Boost pinned/canonical documents")
    exclude_status: Optional[List[str]] = Field(
        default=None,
        description="Document statuses to exclude (defaults to ['archived','deprecated'] in RPC)",
    )

    # Study scoping (migration 33/34)
    study_id: Optional[str] = Field(default=None, description="Restrict retrieval to one study")
    cross_study_entities: bool = Field(
        default=False,
        description="When study_id is set, still allow Entity/Person/Org/Concept nodes to cross studies",
    )

    # ── Retrieval-quality pipeline (migrations 41/42 + reranker_service) ────
    use_hybrid: bool = Field(
        default=True,
        description="Use hybrid_search_kg_nodes (dense + BM25 RRF). When false, falls back to dense-only search_kg_nodes.",
    )
    sparse_weight: float = Field(
        default=1.0,
        description="RRF weight for sparse rank. 0=dense-only, 1=equal, >1=sparse-biased",
    )
    candidate_pool: int = Field(
        default=50,
        description="How many candidates to fetch from each retriever before fusion/rerank/dedup",
    )
    use_rerank: bool = Field(
        default=True,
        description="Apply cross-encoder reranker to fused candidates (no-op if COHERE_API_KEY unset)",
    )
    dedup_threshold: float = Field(
        default=0.95,
        description="Drop near-duplicate results above this cosine. Set 1.0 to disable.",
    )

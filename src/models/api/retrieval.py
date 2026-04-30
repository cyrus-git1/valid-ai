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

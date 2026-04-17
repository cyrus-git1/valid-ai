"""Pydantic models for the /search router."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from src.models.base import TenantOwned


class GraphSearchRequest(TenantOwned):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    hop_limit: int = Field(default=1, ge=0, le=2)
    max_neighbours: int = Field(default=3, ge=1, le=10)
    min_edge_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    node_types: Optional[List[str]] = Field(default=None, description="Filter to these node types (e.g. ['Entity', 'Chunk'])")
    rel_types: Optional[List[str]] = Field(default=None, description="Filter edge traversal to these rel_types (e.g. ['mentions'])")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter to chunks from these document IDs only")
    recency_weight: float = Field(default=0.0, ge=0.0, le=1.0, description="0.0 = pure vector, 0.3 = 30% recency blend")
    boost_pinned: bool = Field(default=False, description="Boost pinned/canonical documents")
    exclude_status: Optional[List[str]] = Field(default=None, description="Document statuses to exclude (default: archived, deprecated)")
    source_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by parent documents.source_type (e.g. ['ContextSummary','TopicSummary'])",
    )


class SearchResultItem(BaseModel):
    node_id: str
    node_key: str
    node_type: str
    entity_type: Optional[str] = None
    content: str
    similarity_score: Optional[float]
    document_id: Optional[str]
    chunk_index: Optional[int]
    source: str = "vector"
    retrieval_reason: Optional[str] = None
    evidence_quote: Optional[str] = None
    evidence_score: Optional[float] = None
    evidence_count: int = 0
    final_score: Optional[float] = None
    source_type: Optional[str] = None


class GraphSearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    seed_nodes: int
    expanded_nodes: int

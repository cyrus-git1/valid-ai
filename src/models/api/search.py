"""Pydantic models for the /search router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SemanticSearchRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class GraphSearchRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    hop_limit: int = Field(default=1, ge=0, le=2)
    max_neighbours: int = Field(default=3, ge=1, le=10)
    min_edge_weight: float = Field(default=0.75, ge=0.0, le=1.0)


class AskRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    hop_limit: int = Field(default=1, ge=0, le=2)
    model: str = "gpt-4o-mini"


class SearchResultItem(BaseModel):
    node_id: str
    node_key: str
    node_type: str
    content: str                    # description / chunk text preview
    similarity_score: Optional[float]
    document_id: Optional[str]
    chunk_index: Optional[int]
    source: str = "vector"          # "vector" | "graph_expansion"


class SemanticSearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


class GraphSearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    seed_nodes: int
    expanded_nodes: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SearchResultItem]

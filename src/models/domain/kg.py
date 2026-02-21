from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

JsonDict = Dict[str, Any]


class NodeStatus(str, Enum):
    ACTIVE = "active"
    PENDING_LINKING = "pending_linking"
    ARCHIVED = "archived"


class ArtifactType(str, Enum):
    WEB_PAGE = "WebPage"
    PDF = "PDF"
    IMAGE = "Image"
    PPTX = "PowerPoint"
    DOCX = "Docx"
    VIDEO_TRANSCRIPT = "VideoTranscript"
    CHAT_TRANSCRIPT = "ChatTranscript"
    CHAT_SNAPSHOT = "ChatSnapshot"
    CHUNK = "Chunk"


class KnowledgeGraphNodeUpsert(BaseModel):
    # Upsert by (tenant_id, client_id, node_key)
    tenant_id: UUID
    client_id: UUID

    node_key: str
    type: ArtifactType
    name: str
    description: Optional[str] = None

    properties: JsonDict = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    status: NodeStatus = NodeStatus.ACTIVE


class KnowledgeGraphEdgeUpsert(BaseModel):
    # Upsert by (tenant_id, client_id, src_id, dst_id, rel_type)
    tenant_id: UUID
    client_id: UUID

    src_id: UUID
    dst_id: UUID

    rel_type: str
    weight: Optional[float] = None
    properties: JsonDict = Field(default_factory=dict)


class KGNodeEvidenceUpsert(BaseModel):
    tenant_id: UUID
    client_id: UUID
    node_id: UUID
    chunk_id: UUID
    quote: Optional[str] = None
    score: Optional[float] = None


class KGEdgeEvidenceUpsert(BaseModel):
    tenant_id: UUID
    client_id: UUID
    edge_id: UUID
    chunk_id: UUID
    quote: Optional[str] = None
    score: Optional[float] = None


class PruneRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    edge_stale_days: int = 90
    node_stale_days: int = 180
    min_degree: int = 3
    keep_edge_evidence: int = 5
    keep_node_evidence: int = 10

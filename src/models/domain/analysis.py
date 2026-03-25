"""Domain models for analysis service shared context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from uuid import UUID


@dataclass
class SentimentSharedContext:
    """Pre-fetched context reused across sentiment analysis queries for one tenant+client."""
    tenant_id: UUID
    client_id: UUID
    transcript_count: int = 0
    transcript_chunks: List[Dict[str, Any]] = field(default_factory=list)
    transcript_context: str = ""
    context_summary: str = ""
    chunks_analysed: int = 0


@dataclass
class StrategicSharedContext:
    """Pre-fetched context reused across strategic analysis queries for one tenant+client."""
    tenant_id: UUID
    client_id: UUID
    transcript_count: int = 0
    depth: str = "foundational"
    transcript_context: str = ""
    context_summary: str = ""
    transcript_chunks_retrieved: int = 0
    context_summary_available: bool = False

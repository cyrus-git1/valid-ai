"""Pydantic models for the /memory router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryStateResponse(BaseModel):
    memory_version: int = 0
    last_changed_at: Optional[str] = None


class MemoryChange(BaseModel):
    change_type: str
    memory_version: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class MemoryChangesResponse(BaseModel):
    changes: List[MemoryChange] = Field(default_factory=list)
    current_version: int = 0

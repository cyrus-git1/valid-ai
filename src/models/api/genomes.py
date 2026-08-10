"""Pydantic models for the /genomes router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenomeModel(BaseModel):
    step_name: str
    version: int
    parent_version: Optional[int] = None
    manager_prompt: str = ""
    rubric: List[Dict[str, Any]] = Field(default_factory=list)
    score_threshold: float = 0.7
    max_retries: int = 2
    agent_system_prompt: str = ""
    output_format_prompt: str = ""
    optimization_notes: str = ""
    test_score: Optional[float] = None
    test_details: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = False
    created_at: Optional[str] = None


class GenomeSummary(BaseModel):
    version: int
    is_active: bool = False
    parent_version: Optional[int] = None
    test_score: Optional[float] = None
    optimization_notes: str = ""
    created_at: Optional[str] = None


class SetActiveRequest(BaseModel):
    version: Optional[int] = None  # null => deactivate every version for the step

"""/genomes router — harness genome storage (GLOBAL config, admin-scoped).

Harness genomes are deployment-global optimization state (manager prompts,
rubrics, thresholds, generator prompts) versioned per harness step. They have NO
tenant dimension, so unlike the rest of the data plane these endpoints are
admin-scoped rather than tenant-scoped. Moved here so the agent layer no longer
holds a Supabase client for this.

Keyed by (step_name, version); exactly one active version per step.

  POST /genomes                              persist a genome version
  GET  /genomes/{step}                       list version summaries (newest first)
  GET  /genomes/{step}/active                the active genome (null if none)
  GET  /genomes/{step}/version/{version}     one full genome (404 if absent)
  PUT  /genomes/{step}/active                set active version (null => deactivate all)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.routers.admin_router import _require_admin
from src.db.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/genomes", tags=["genomes"])

TABLE = "harness_genomes"
_FULL = (
    "step_name, version, parent_version, manager_prompt, rubric, score_threshold, "
    "max_retries, agent_system_prompt, output_format_prompt, optimization_notes, "
    "test_score, test_details, is_active, created_at"
)


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


@router.post("", response_model=GenomeModel)
def save_genome(body: GenomeModel, request: Request) -> GenomeModel:
    """Persist a genome version. New versions are never auto-activated (use PUT
    .../active). The (step_name, version) unique constraint rejects duplicates."""
    _require_admin(request)
    sb = get_supabase()
    row = body.model_dump(exclude={"created_at"})
    row["is_active"] = False
    sb.table(TABLE).insert(row).execute()
    return body


@router.get("/{step_name}", response_model=List[GenomeSummary])
def list_versions(step_name: str, request: Request) -> List[GenomeSummary]:
    _require_admin(request)
    sb = get_supabase()
    res = (
        sb.table(TABLE)
        .select("version, is_active, parent_version, test_score, optimization_notes, created_at")
        .eq("step_name", step_name)
        .order("version", desc=True)
        .execute()
    )
    return [GenomeSummary(**r) for r in (res.data or [])]


@router.get("/{step_name}/active", response_model=Optional[GenomeModel])
def get_active(step_name: str, request: Request) -> Optional[GenomeModel]:
    """The active genome for a step, or null (200) when none is active — the
    caller then falls back to its hardcoded default."""
    _require_admin(request)
    sb = get_supabase()
    res = (
        sb.table(TABLE).select(_FULL)
        .eq("step_name", step_name).eq("is_active", True).limit(1).execute()
    )
    rows = res.data or []
    return GenomeModel(**rows[0]) if rows else None


@router.get("/{step_name}/version/{version}", response_model=GenomeModel)
def get_version(step_name: str, version: int, request: Request) -> GenomeModel:
    _require_admin(request)
    sb = get_supabase()
    res = (
        sb.table(TABLE).select(_FULL)
        .eq("step_name", step_name).eq("version", version).limit(1).execute()
    )
    rows = res.data or []
    if not rows:
        raise HTTPException(status_code=404, detail=f"genome v{version} not found for {step_name}")
    return GenomeModel(**rows[0])


@router.put("/{step_name}/active", response_model=Optional[GenomeModel])
def set_active(step_name: str, body: SetActiveRequest, request: Request) -> Optional[GenomeModel]:
    """Activate one version (atomic; all others off). version=null deactivates
    every version for the step (revert to hardcoded defaults)."""
    _require_admin(request)
    sb = get_supabase()
    sb.rpc("set_active_genome", {"p_step": step_name, "p_version": body.version}).execute()
    if body.version is None:
        return None
    res = (
        sb.table(TABLE).select(_FULL)
        .eq("step_name", step_name).eq("version", body.version).limit(1).execute()
    )
    rows = res.data or []
    return GenomeModel(**rows[0]) if rows else None

"""Shared base models used across API and domain layers."""
from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class TenantScoped(BaseModel):
    """Base for any model scoped to a tenant + client pair."""
    tenant_id: UUID
    client_id: UUID


class TenantScopedRequest(TenantScoped):
    """Base for request models that accept an optional client profile."""
    client_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Client profile: industry, headcount, revenue, persona, demographic, etc.",
    )


class StatusResponse(BaseModel):
    """Mixin for responses that carry a status and optional error."""
    status: str = "complete"
    error: Optional[str] = None

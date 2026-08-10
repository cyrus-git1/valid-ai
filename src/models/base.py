"""Shared base models used across API and domain layers."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class TenantOwned(BaseModel):
    """Base for models owned by a tenant with an optional client scope."""
    tenant_id: UUID
    client_id: Optional[UUID] = None

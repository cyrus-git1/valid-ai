"""Timestamp utilities and mixins for domain models."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin(BaseModel):
    """Mixin providing created_at / updated_at with UTC defaults."""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

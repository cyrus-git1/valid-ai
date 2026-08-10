"""Pydantic models for the /feedback router."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class FeedbackRow(BaseModel):
    rating: str
    id: Optional[str] = None
    request_id: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    comment: Optional[str] = None
    intent: Optional[str] = None
    message: Optional[str] = None
    response: Optional[str] = None


class FeedbackAck(BaseModel):
    id: str
    persisted: bool

"""/feedback router — durable Vera chat feedback (thumbs + notes).

Best-effort persistence moved off the agent layer's direct Supabase client
(Step 5, Bucket C). The agent still keeps an in-memory ring, so this endpoint
being down never fails the chat request — it just loses durability.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from src.db.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])

_TABLE = "vera_feedback"


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


@router.post("", response_model=FeedbackAck)
def record(body: FeedbackRow, request: Request) -> FeedbackAck:
    """Insert one feedback row. Requires a valid API key (auth middleware); the
    row carries its own tenant_id from the agent's request context."""
    fid = body.id or str(uuid.uuid4())
    row = body.model_dump()
    row["id"] = fid
    sb = get_supabase()
    sb.table(_TABLE).insert(row).execute()
    return FeedbackAck(id=fid, persisted=True)

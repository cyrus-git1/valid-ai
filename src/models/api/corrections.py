"""
src/models/api/corrections.py
-----------------------------
Request/response envelopes for tenant-scoped context corrections (overrides).

A correction is a durable, non-destructive "going forward" fix applied where
context is READ (context-summary + retrieval), never by editing ingested text:
  - term_replace: swap `from` -> `to` (e.g. a company rename)
  - disregard:    drop `from` (no replacement)
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class CorrectionCreateRequest(BaseModel):
    kind: Literal["term_replace", "disregard"]
    from_: str = Field(alias="from", min_length=1, description="Term to replace / disregard")
    to: Optional[str] = Field(default=None, description="Replacement (required for term_replace)")
    note: Optional[str] = Field(default=None, description="Optional audit note")
    applies_to: Union[str, List[UUID]] = Field(
        default="all", description='"all" or a list of document ids'
    )
    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _validate(self) -> "CorrectionCreateRequest":
        if self.kind == "term_replace" and not (self.to and self.to.strip()):
            raise ValueError("term_replace requires a non-empty 'to'")
        if isinstance(self.applies_to, str) and self.applies_to != "all":
            raise ValueError("applies_to must be 'all' or a list of document ids")
        return self


class CorrectionCreateResponse(BaseModel):
    correction_id: str
    applied: bool = True


class CorrectionItem(BaseModel):
    correction_id: str
    kind: str
    from_: str = Field(alias="from")
    to: Optional[str] = None
    note: Optional[str] = None
    applies_to: Any = "all"
    created_at: Optional[str] = None
    model_config = {"populate_by_name": True}


class CorrectionsListResponse(BaseModel):
    corrections: List[CorrectionItem] = Field(default_factory=list)


class CorrectionDeleteResponse(BaseModel):
    deleted: bool
    correction_id: str

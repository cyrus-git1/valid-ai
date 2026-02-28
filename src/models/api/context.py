from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class Demographic(BaseModel):
    age_range: Optional[str] = None
    income_bracket: Optional[str] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    language: str = "en"


class ClientProfile(BaseModel):
    industry: Optional[str] = None
    headcount: Optional[int] = None
    demographic: Demographic = Field(default_factory=Demographic)

    def to_prompt_context(self) -> str:
        """Render the client profile as natural language for LLM system prompts."""
        parts: List[str] = []
        if self.industry:
            parts.append(f"Industry: {self.industry}")
        if self.headcount:
            parts.append(f"Company headcount: {self.headcount}")
        d = self.demographic
        if d.age_range:
            parts.append(f"Target age range: {d.age_range}")
        if d.income_bracket:
            parts.append(f"Income bracket: {d.income_bracket}")
        if d.occupation:
            parts.append(f"Occupation: {d.occupation}")
        if d.location:
            parts.append(f"Location: {d.location}")
        if d.language and d.language != "en":
            parts.append(f"Language: {d.language}")
        return "\n".join(parts) if parts else ""


class ContextSources(BaseModel):
    docs: List[str] = Field(default_factory=list, description="File paths to PDFs/DOCX")
    weblinks: List[str] = Field(default_factory=list, description="URLs to scrape")
    transcripts: List[str] = Field(default_factory=list, description="File paths to WebVTT (.vtt) transcripts from Daily.js sessions")


class ContextBuildRequest(BaseModel):
    """Unified input payload for building a client's knowledge base."""
    tenant_id: UUID
    client_id: UUID

    context: ContextSources = Field(default_factory=ContextSources)
    client_profile: ClientProfile = Field(default_factory=ClientProfile)


class ContextBuildResponse(BaseModel):
    job_id: str
    status: str
    documents_ingested: int = 0
    weblinks_ingested: int = 0
    transcripts_ingested: int = 0
    total_chunks: int = 0
    kg_nodes_upserted: int = 0
    kg_edges_upserted: int = 0
    warnings: List[str] = Field(default_factory=list)


class ContextBuildStatusResponse(BaseModel):
    job_id: str
    status: str  # "running" | "complete" | "failed"
    detail: Optional[Dict[str, Any]] = None

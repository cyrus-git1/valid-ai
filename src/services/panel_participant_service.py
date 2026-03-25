"""
src/services/panel_participant_service.py
------------------------------------------
OOP service for panel participant ingest and two-stage filtering.

Pipeline
--------
  Ingest:   serialize → chunk → tokenize → embed → store (+ optional KG build)
  Filter:   label filter (rule-based or LLM) → embedding similarity filter

Import
------
    from src.services.panel_participant_service import PanelParticipantService
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config.llm import LLMConfig
from supabase import Client

from src.models.api.panel_participants import (
    PanelFilterResponse,
    PanelFilterResult,
    PanelParticipantResult,
)
from src.models.domain.kg import KGBuildConfig
from src.services.base_service import BaseAnalysisService
from src.services.chunking_service import ChunkingService
from src.services.context_summary_service import ContextSummaryService
from src.services.embedding_service import embed_texts
from src.services.kg_service import KGService

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


# ── LLM prompts for label filtering ──────────────────────────────────────────

CRITERIA_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research panel consultant. You will be given the "
        "business context of a company — a summary of their industry, topics, "
        "and knowledge base content. Your job is to determine what ideal panel "
        "participant characteristics would be most relevant for research with "
        "this company.\n\n"
        "Return a JSON object with the following fields (include only relevant ones):\n"
        '{{"industries": ["..."], "seniority_levels": ["..."], '
        '"job_functions": ["..."], "tools_or_platforms": ["..."], '
        '"interests": ["..."], "demographics": {{"age_ranges": ["..."], '
        '"regions": ["..."]}}, "domain_expertise": ["..."], '
        '"key_attributes": ["..."]}}'
    ),
    (
        "human",
        "Business context:\n\n{business_context}\n\n"
        "Generate the ideal participant criteria."
    ),
])

PARTICIPANT_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at matching research participants to business needs.\n\n"
        "You will receive:\n"
        "1. Ideal participant criteria for a company\n"
        "2. A batch of participant profiles\n\n"
        "For each participant, score their relevance from 0.0 to 1.0 and list "
        "the specific reasons they match or don't match.\n\n"
        "Return a JSON array with one object per participant:\n"
        '[{{"index": 0, "score": 0.85, "match_reasons": ["reason1", "reason2"]}}]'
    ),
    (
        "human",
        "Ideal criteria:\n{criteria}\n\n"
        "Participant profiles:\n{profiles}\n\n"
        "Score each participant."
    ),
])


# ── Service ──────────────────────────────────────────────────────────────────


class PanelParticipantService(BaseAnalysisService):
    """Ingest and filter panel participants using existing chunking/embedding infra."""

    def __init__(self, supabase: Client):
        super().__init__(supabase)

    # ══════════════════════════════════════════════════════════════════════════
    # Serialization
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _serialize_participant(
        participant_entry: JsonDict,
    ) -> str:
        """
        Convert a nested participant JSON object into natural-language text
        suitable for chunking and embedding.

        Expects the top-level dict with keys: company, industry_focus, participant.
        """
        company = participant_entry.get("company", "Unknown Company")
        industry = participant_entry.get("industry_focus", "")
        p = participant_entry.get("participant", participant_entry)

        sections: List[str] = []

        # Header
        first = p.get("first_name", "")
        last = p.get("last_name", "")
        name = f"{first} {last}".strip() or "Unknown Participant"
        source = p.get("source", "unknown")
        header = f"{name} is a panel participant sourced from {source}"
        if company and company != "Unknown Company":
            header += f", associated with {company}"
        if industry:
            header += f" in the {industry} sector"
        header += "."
        sections.append(header)

        # Demographics
        demo = p.get("demographics", {})
        if demo:
            parts = []
            if demo.get("age"):
                parts.append(f"{demo['age']}-year-old")
            if demo.get("gender"):
                parts.append(demo["gender"])
            location_parts = [
                demo.get("city"), demo.get("region"), demo.get("country"),
            ]
            location = ", ".join(x for x in location_parts if x)
            if location:
                parts.append(f"located in {location}")
            if demo.get("ethnicity"):
                eth = demo["ethnicity"]
                if isinstance(eth, list):
                    eth = ", ".join(eth)
                parts.append(f"ethnicity: {eth}")
            if demo.get("household_income"):
                parts.append(f"household income: {demo['household_income']}")
            if demo.get("primary_language"):
                parts.append(f"primary language: {demo['primary_language']}")
            if parts:
                sections.append("Demographics: " + ", ".join(parts) + ".")

        # Education
        edu = p.get("education", {})
        if edu:
            parts = []
            if edu.get("level"):
                parts.append(edu["level"])
            if edu.get("field_of_study"):
                parts.append(f"in {edu['field_of_study']}")
            if edu.get("professional_certifications"):
                certs = edu["professional_certifications"]
                if isinstance(certs, list):
                    certs = ", ".join(certs)
                parts.append(f"certifications: {certs}")
            if parts:
                sections.append("Education: " + " ".join(parts) + ".")

        # Professional
        prof = p.get("professional", {})
        if prof:
            parts = []
            if prof.get("occupation"):
                parts.append(prof["occupation"])
            if prof.get("company_name"):
                parts.append(f"at {prof['company_name']}")
            if prof.get("industry"):
                parts.append(f"in {prof['industry']}")
            if prof.get("seniority"):
                parts.append(f"seniority: {prof['seniority']}")
            if prof.get("years_experience"):
                parts.append(f"{prof['years_experience']} years experience")
            if prof.get("decision_maker"):
                parts.append("decision maker")
            if prof.get("uses_software_tools"):
                tools = prof["uses_software_tools"]
                if isinstance(tools, list):
                    tools = ", ".join(tools)
                parts.append(f"tools: {tools}")
            if prof.get("annual_budget_range"):
                parts.append(f"budget: {prof['annual_budget_range']}")
            if parts:
                sections.append("Professional: " + ", ".join(parts) + ".")

        # Domain-specific profiles
        domain_profiles = [
            ("health_profile", "Health Profile"),
            ("financial_profile", "Financial Profile"),
            ("education_profile", "Education Profile"),
            ("shopping_profile", "Shopping Profile"),
            ("tech_profile", "Technology Profile"),
        ]
        for key, label in domain_profiles:
            profile = p.get(key, {})
            if profile:
                readable_parts = []
                for k, v in profile.items():
                    if v is not None and v != "" and v != [] and v != {}:
                        display_key = k.replace("_", " ")
                        if isinstance(v, list):
                            v = ", ".join(str(x) for x in v)
                        elif isinstance(v, bool):
                            v = "yes" if v else "no"
                        readable_parts.append(f"{display_key}: {v}")
                if readable_parts:
                    sections.append(f"{label}: " + ". ".join(readable_parts) + ".")

        # Psychographic
        psych = p.get("psychographic", {})
        if psych:
            parts = []
            if psych.get("hobbies"):
                hobbies = psych["hobbies"]
                if isinstance(hobbies, list):
                    hobbies = ", ".join(hobbies)
                parts.append(f"hobbies: {hobbies}")
            if psych.get("interests"):
                interests = psych["interests"]
                if isinstance(interests, list):
                    interests = ", ".join(interests)
                parts.append(f"interests: {interests}")
            if psych.get("values"):
                values = psych["values"]
                if isinstance(values, list):
                    values = ", ".join(values)
                parts.append(f"values: {values}")
            if psych.get("personality_type"):
                parts.append(f"personality: {psych['personality_type']}")
            if psych.get("tech_savviness"):
                parts.append(f"tech savviness: {psych['tech_savviness']}")
            if psych.get("devices_owned"):
                devices = psych["devices_owned"]
                if isinstance(devices, list):
                    devices = ", ".join(devices)
                parts.append(f"devices: {devices}")
            if parts:
                sections.append("Psychographic: " + ", ".join(parts) + ".")

        # Research history
        rh = p.get("research_history", {})
        if rh:
            parts = []
            if rh.get("sessions_completed") is not None:
                parts.append(f"{rh['sessions_completed']} sessions completed")
            if rh.get("average_session_rating"):
                parts.append(f"avg rating: {rh['average_session_rating']}")
            if rh.get("preferred_session_format"):
                parts.append(f"preferred format: {rh['preferred_session_format']}")
            if rh.get("tags"):
                tags = rh["tags"]
                if isinstance(tags, list):
                    tags = ", ".join(tags)
                parts.append(f"tags: {tags}")
            if rh.get("internal_notes"):
                parts.append(f"notes: {rh['internal_notes']}")
            if parts:
                sections.append("Research history: " + ", ".join(parts) + ".")

        # Custom fields
        custom = p.get("custom_fields", {})
        if custom:
            readable = []
            for k, v in custom.items():
                if v is not None and v != "":
                    display_key = k.replace("_", " ")
                    if isinstance(v, bool):
                        v = "yes" if v else "no"
                    readable.append(f"{display_key}: {v}")
            if readable:
                sections.append("Additional attributes: " + ", ".join(readable) + ".")

        return "\n\n".join(sections)

    # ══════════════════════════════════════════════════════════════════════════
    # Document & chunk persistence (follows IngestService pattern)
    # ══════════════════════════════════════════════════════════════════════════

    def _upsert_participant_document(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        vendor_name: str,
        participant_id: str,
        title: str,
        metadata: JsonDict,
    ) -> UUID:
        """Create or update a document record for a single participant."""
        sb = self._require_supabase()
        source_type = "panel_participant"
        source_uri = f"panel:{vendor_name}/{participant_id}"

        existing = (
            sb.table("documents")
            .select("id")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .eq("source_uri", source_uri)
            .limit(1)
            .execute()
        )
        if existing.data:
            doc_id = existing.data[0]["id"]
            sb.table("documents").update({
                "source_type": source_type,
                "title": title,
                "metadata": metadata or {},
            }).eq("id", doc_id).execute()
            logger.info("Updated existing panel document %s", doc_id)
            return UUID(doc_id)

        res = (
            sb.table("documents")
            .insert({
                "tenant_id": str(tenant_id),
                "client_id": str(client_id),
                "source_type": source_type,
                "source_uri": source_uri,
                "title": title,
                "metadata": metadata or {},
            })
            .execute()
        )
        if not res.data:
            raise RuntimeError("documents insert returned no rows")
        return UUID(res.data[0]["id"])

    def _upsert_chunk(
        self,
        *,
        tenant_id: UUID,
        document_id: UUID,
        chunk_index: int,
        start_page: Optional[int],
        end_page: Optional[int],
        text: str,
        token_count: Optional[int],
        metadata: JsonDict,
        embedding: Optional[List[float]],
    ) -> UUID:
        """Upsert a single chunk via Supabase RPC."""
        sb = self._require_supabase()
        res = sb.rpc(
            "upsert_chunk",
            {
                "p_tenant_id": str(tenant_id),
                "p_document_id": str(document_id),
                "p_chunk_index": chunk_index,
                "p_page_start": start_page,
                "p_page_end": end_page,
                "p_content": text,
                "p_content_tokens": token_count,
                "p_metadata": metadata or {},
                "p_embedding": embedding,
            },
        ).execute()
        return UUID(str(res.data))

    def _embed_in_batches(
        self, texts: List[str], model: str, batch_size: int,
    ) -> List[List[float]]:
        """Embed texts in batches using the embedding service."""
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(embed_texts(texts[i : i + batch_size], model=model))
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # Ingest
    # ══════════════════════════════════════════════════════════════════════════

    def ingest_participants(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        vendor_name: str,
        participants: List[JsonDict],
        metadata: Optional[JsonDict] = None,
        embed_model: str = "text-embedding-3-small",
        embed_batch_size: int = 64,
        build_kg: bool = True,
    ) -> Dict[str, Any]:
        """
        Full ingest pipeline for a batch of panel participants.

        For each participant: serialize → chunk → embed → store as document + chunks.
        After all participants: optionally build KG and context summary.
        """
        metadata = metadata or {}
        results: List[PanelParticipantResult] = []
        warnings: List[str] = []

        # Phase 1: Serialize and chunk all participants
        all_chunks: List[JsonDict] = []
        chunk_ranges: List[Tuple[int, int]] = []      # (start, end) in all_chunks
        participant_meta: List[Tuple[str, str]] = []   # (participant_id, title)

        for i, entry in enumerate(participants):
            p = entry.get("participant", entry)
            pid = (
                p.get("panel_id")
                or p.get("email")
                or f"idx-{i}"
            )
            first = p.get("first_name", "")
            last = p.get("last_name", "")
            title = f"{first} {last}".strip() or f"Participant {i}"

            text = self._serialize_participant(entry)
            pages = [{"page": 1, "text": text}]
            chunks = ChunkingService.chunk_pages_spacy_token_aware(pages, max_tokens=800, overlap_tokens=120)

            start = len(all_chunks)
            all_chunks.extend(chunks)
            chunk_ranges.append((start, len(all_chunks)))
            participant_meta.append((pid, title))

        if not all_chunks:
            return {
                "total_participants": len(participants),
                "completed": 0,
                "failed": 0,
                "results": [],
                "warnings": ["No chunks produced from any participant."],
            }

        # Phase 2: Single batch embedding across all participants
        all_texts = [c["text"] for c in all_chunks]
        try:
            all_embeddings = self._embed_in_batches(
                all_texts, model=embed_model, batch_size=embed_batch_size,
            )
        except Exception as e:
            raise RuntimeError(f"Batch embedding failed: {e}") from e

        # Phase 3: Store per-participant
        for i, (start, end) in enumerate(chunk_ranges):
            pid, title = participant_meta[i]
            participant_chunks = all_chunks[start:end]
            participant_embeddings = all_embeddings[start:end]
            chunk_warnings: List[str] = []

            try:
                document_id = self._upsert_participant_document(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    vendor_name=vendor_name,
                    participant_id=pid,
                    title=title,
                    metadata={
                        **metadata,
                        "vendor_name": vendor_name,
                        "participant_id": pid,
                    },
                )

                chunk_ids: List[UUID] = []
                for idx, (chunk_data, embedding) in enumerate(
                    zip(participant_chunks, participant_embeddings)
                ):
                    try:
                        chunk_id = self._upsert_chunk(
                            tenant_id=tenant_id,
                            document_id=document_id,
                            chunk_index=idx,
                            start_page=chunk_data.get("start_page"),
                            end_page=chunk_data.get("end_page"),
                            text=chunk_data["text"],
                            token_count=chunk_data.get("token_count"),
                            metadata={
                                "source_type": "panel_participant",
                                "vendor_name": vendor_name,
                                "participant_id": pid,
                            },
                            embedding=embedding,
                        )
                        chunk_ids.append(chunk_id)
                    except Exception as e:
                        chunk_warnings.append(f"chunk {idx} failed: {e}")

                results.append(PanelParticipantResult(
                    participant_index=i,
                    document_id=document_id,
                    chunks_upserted=len(chunk_ids),
                    warnings=chunk_warnings,
                ))
            except Exception as e:
                logger.warning("Participant %d ingest failed: %s", i, e)
                warnings.append(f"Participant {i} ({pid}) failed: {e}")

        # Phase 4: Post-ingest KG build + context summary
        total_chunks = sum(r.chunks_upserted for r in results)
        if build_kg and total_chunks > 0:
            try:
                kg_svc = KGService(self._require_supabase())
                kg_result = kg_svc.build_kg_from_chunk_embeddings(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    config=KGBuildConfig(),
                )
                logger.info(
                    "KG build — nodes=%d edges=%d",
                    kg_result.get("nodes_upserted", 0),
                    kg_result.get("edges_upserted", 0),
                )
            except Exception as e:
                warnings.append(f"KG build failed: {e}")
                logger.warning("KG build failed: %s", e)

            try:
                summary_svc = ContextSummaryService(self._require_supabase())
                summary_svc.generate_summary(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    force_regenerate=True,
                )
            except Exception as e:
                warnings.append(f"Context summary generation failed: {e}")
                logger.warning("Context summary generation failed: %s", e)

        return {
            "total_participants": len(participants),
            "completed": len(results),
            "failed": len(participants) - len(results),
            "results": [r.model_dump() for r in results],
            "warnings": warnings,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers for filtering
    # ══════════════════════════════════════════════════════════════════════════

    def _fetch_business_context(
        self,
        tenant_id: UUID,
        client_id: UUID,
    ) -> Tuple[str, List[str], List[JsonDict]]:
        """
        Fetch the business context for filtering.
        Returns: (summary_text, topics, business_chunks)
        """
        sb = self._require_supabase()

        # Context summary
        summary_svc = ContextSummaryService(sb)
        summary_row = summary_svc.get_summary(
            tenant_id=tenant_id, client_id=client_id,
        )
        summary_text = ""
        topics: List[str] = []
        if summary_row:
            summary_text = summary_row.get("summary", "")
            topics = summary_row.get("topics", [])

        # Fetch non-panel document chunks
        doc_res = (
            sb.table("documents")
            .select("id")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .neq("source_type", "panel_participant")
            .execute()
        )
        doc_ids = [row["id"] for row in (doc_res.data or [])]

        business_chunks: List[JsonDict] = []
        if doc_ids:
            chunk_res = (
                sb.table("chunks")
                .select("content, embedding, document_id")
                .eq("tenant_id", str(tenant_id))
                .in_("document_id", doc_ids)
                .limit(200)
                .execute()
            )
            business_chunks = chunk_res.data or []

        return summary_text, topics, business_chunks

    def _fetch_panel_chunks(
        self,
        tenant_id: UUID,
        client_id: UUID,
    ) -> Dict[str, List[JsonDict]]:
        """
        Fetch all panel participant chunks grouped by document_id.
        Returns: {document_id: [chunk_rows]}
        """
        sb = self._require_supabase()

        doc_res = (
            sb.table("documents")
            .select("id, title, metadata")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .eq("source_type", "panel_participant")
            .execute()
        )

        if not doc_res.data:
            return {}

        doc_ids = [row["id"] for row in doc_res.data]
        doc_map = {row["id"]: row for row in doc_res.data}

        chunk_res = (
            sb.table("chunks")
            .select("content, embedding, document_id, metadata")
            .eq("tenant_id", str(tenant_id))
            .in_("document_id", doc_ids)
            .execute()
        )

        grouped: Dict[str, List[JsonDict]] = {}
        for chunk in (chunk_res.data or []):
            did = chunk["document_id"]
            chunk["_doc"] = doc_map.get(did, {})
            grouped.setdefault(did, []).append(chunk)

        return grouped

    # ══════════════════════════════════════════════════════════════════════════
    # Label filter (no LLM)
    # ══════════════════════════════════════════════════════════════════════════

    def filter_by_labels(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        top_k: int = 20,
    ) -> List[PanelFilterResult]:
        """
        Score participants by term overlap with business context.

        Since vendor tags won't match our system params, we compare
        the full serialized participant text against terms derived from
        the business context summary and document chunks.
        """
        summary_text, topics, business_chunks = self._fetch_business_context(
            tenant_id, client_id,
        )

        # Build label set from topics + key terms from summary and chunks
        label_set: set[str] = set()
        for topic in topics:
            label_set.add(topic.lower().strip())

        # Extract significant terms from summary
        if summary_text:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", summary_text.lower())
            label_set.update(words)

        # Extract terms from business chunk content
        for chunk in business_chunks[:50]:
            content = (chunk.get("content") or "").lower()
            words = re.findall(r"\b[a-zA-Z]{4,}\b", content)
            label_set.update(words)

        # Remove overly common words
        _STOP_WORDS = frozenset({
            "this", "that", "with", "from", "have", "been", "were", "they",
            "their", "would", "could", "should", "about", "which", "there",
            "will", "more", "also", "than", "into", "some", "other", "what",
            "when", "your", "each", "most", "very", "such", "only", "then",
            "just", "over", "does", "these", "like", "well", "much",
        })
        label_set -= _STOP_WORDS

        if not label_set:
            return []

        # Score each panel participant
        panel_chunks = self._fetch_panel_chunks(tenant_id, client_id)
        results: List[PanelFilterResult] = []

        for doc_id, chunks in panel_chunks.items():
            combined_text = " ".join(
                (c.get("content") or "").lower() for c in chunks
            )
            doc_info = chunks[0].get("_doc", {}) if chunks else {}
            title = doc_info.get("title", "Unknown")

            matched = [
                label for label in label_set
                if label in combined_text
            ]
            score = len(matched) / len(label_set) if label_set else 0.0

            if score > 0:
                results.append(PanelFilterResult(
                    document_id=doc_id,
                    participant_name=title,
                    relevance_score=round(score, 4),
                    match_reasons=[f"Matched {len(matched)}/{len(label_set)} context terms"],
                    matched_labels=sorted(matched)[:30],
                ))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    # ══════════════════════════════════════════════════════════════════════════
    # Label filter (with LLM)
    # ══════════════════════════════════════════════════════════════════════════

    def filter_by_labels_llm(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        top_k: int = 20,
        llm_model: str = LLMConfig.DEFAULT,
    ) -> List[PanelFilterResult]:
        """
        LLM reads business context, generates ideal participant criteria,
        then scores each participant against those criteria.
        """
        summary_text, topics, business_chunks = self._fetch_business_context(
            tenant_id, client_id,
        )

        # Build business context string for the LLM
        context_parts = []
        if summary_text:
            context_parts.append(f"Business Summary: {summary_text}")
        if topics:
            context_parts.append(f"Key Topics: {', '.join(topics)}")
        for i, chunk in enumerate(business_chunks[:10]):
            content = (chunk.get("content") or "").strip()
            if content:
                context_parts.append(f"[Excerpt {i + 1}] {content}")

        business_context = "\n\n".join(context_parts) or "(No business context available.)"

        # LLM call 1: Generate ideal criteria
        llm = self._create_llm(model=llm_model, temperature=0.1)
        criteria_chain = CRITERIA_GENERATION_PROMPT | llm | StrOutputParser()

        try:
            raw_criteria = criteria_chain.invoke({"business_context": business_context})
            criteria = self._parse_llm_json(raw_criteria, fallback_keys={})
        except Exception as e:
            logger.warning("Criteria generation failed: %s", e)
            criteria = {}

        if not criteria or "raw_output" in criteria:
            logger.warning("LLM criteria generation returned non-JSON, falling back to label filter")
            return self.filter_by_labels(
                tenant_id=tenant_id, client_id=client_id, top_k=top_k,
            )

        criteria_str = json.dumps(criteria, indent=2)

        # Fetch panel participants and serialize them
        panel_chunks = self._fetch_panel_chunks(tenant_id, client_id)
        if not panel_chunks:
            return []

        # Build participant profiles for scoring
        doc_profiles: List[Tuple[str, str, str]] = []   # (doc_id, title, profile_text)
        for doc_id, chunks in panel_chunks.items():
            doc_info = chunks[0].get("_doc", {}) if chunks else {}
            title = doc_info.get("title", "Unknown")
            profile_text = "\n".join(
                (c.get("content") or "") for c in chunks
            )
            doc_profiles.append((doc_id, title, profile_text))

        # LLM call 2: Score participants in batches
        scoring_chain = PARTICIPANT_SCORING_PROMPT | llm | StrOutputParser()
        results: List[PanelFilterResult] = []
        batch_size = 5

        for batch_start in range(0, len(doc_profiles), batch_size):
            batch = doc_profiles[batch_start : batch_start + batch_size]
            profiles_text = "\n\n---\n\n".join(
                f"[Participant {j}] {title}\n{text}"
                for j, (_, title, text) in enumerate(batch)
            )

            try:
                raw_scores = scoring_chain.invoke({
                    "criteria": criteria_str,
                    "profiles": profiles_text,
                })
                scores = self._parse_llm_json(raw_scores, fallback_keys={})

                if isinstance(scores, list):
                    score_list = scores
                elif isinstance(scores, dict) and "raw_output" not in scores:
                    score_list = scores.get("results", scores.get("participants", []))
                else:
                    score_list = []

                for entry in score_list:
                    idx = entry.get("index", 0)
                    if 0 <= idx < len(batch):
                        doc_id, title, _ = batch[idx]
                        results.append(PanelFilterResult(
                            document_id=doc_id,
                            participant_name=title,
                            relevance_score=round(float(entry.get("score", 0)), 4),
                            match_reasons=entry.get("match_reasons", []),
                            matched_labels=[],
                        ))
            except Exception as e:
                logger.warning("Participant scoring batch failed: %s", e)

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    # ══════════════════════════════════════════════════════════════════════════
    # Embedding similarity filter
    # ══════════════════════════════════════════════════════════════════════════

    def filter_by_embedding_similarity(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        top_k: int = 20,
        similarity_threshold: float = 0.70,
    ) -> List[PanelFilterResult]:
        """
        Compare participant chunk embeddings against business context
        embeddings (document chunks + context summary).

        Computes a business centroid vector and scores each participant
        by max cosine similarity of their chunks against the centroid.
        """
        _, _, business_chunks = self._fetch_business_context(
            tenant_id, client_id,
        )

        # Collect valid business embeddings
        _EMBEDDING_DIM = 1536
        biz_embeddings: List[List[float]] = []
        for chunk in business_chunks:
            emb = chunk.get("embedding")
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                except (json.JSONDecodeError, ValueError):
                    continue
            if isinstance(emb, list) and len(emb) == _EMBEDDING_DIM:
                biz_embeddings.append(emb)

        if not biz_embeddings:
            logger.warning("No business embeddings found for similarity filter")
            return []

        # Compute business centroid
        biz_matrix = np.array(biz_embeddings, dtype=np.float32)
        centroid = biz_matrix.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return []
        centroid = centroid / centroid_norm

        # Fetch panel participant chunks with embeddings
        panel_chunks = self._fetch_panel_chunks(tenant_id, client_id)
        if not panel_chunks:
            return []

        results: List[PanelFilterResult] = []

        for doc_id, chunks in panel_chunks.items():
            doc_info = chunks[0].get("_doc", {}) if chunks else {}
            title = doc_info.get("title", "Unknown")

            # Collect valid embeddings for this participant
            participant_embeddings: List[List[float]] = []
            for chunk in chunks:
                emb = chunk.get("embedding")
                if isinstance(emb, str):
                    try:
                        emb = json.loads(emb)
                    except (json.JSONDecodeError, ValueError):
                        continue
                if isinstance(emb, list) and len(emb) == _EMBEDDING_DIM:
                    participant_embeddings.append(emb)

            if not participant_embeddings:
                continue

            # Compute max cosine similarity against centroid
            p_matrix = np.array(participant_embeddings, dtype=np.float32)
            p_norms = np.linalg.norm(p_matrix, axis=1, keepdims=True)
            p_norms[p_norms == 0] = 1.0
            p_normalized = p_matrix / p_norms

            similarities = p_normalized @ centroid
            max_sim = float(np.max(similarities))

            if max_sim >= similarity_threshold:
                results.append(PanelFilterResult(
                    document_id=doc_id,
                    participant_name=title,
                    relevance_score=round(max_sim, 4),
                    match_reasons=[
                        f"Embedding similarity: {max_sim:.3f} (threshold: {similarity_threshold})"
                    ],
                    matched_labels=[],
                ))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    # ══════════════════════════════════════════════════════════════════════════
    # Full filter pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def filter_participants(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        filter_mode: str = "full",
        top_k: int = 20,
        similarity_threshold: float = 0.70,
        llm_model: str = LLMConfig.DEFAULT,
    ) -> PanelFilterResponse:
        """
        Two-stage participant filtering.

        Modes:
          'label'     — label filter only (no LLM)
          'llm'       — LLM-powered label filter only
          'embedding' — embedding similarity only
          'full'      — label filter first, then embedding similarity on candidates
        """
        if filter_mode == "label":
            results = self.filter_by_labels(
                tenant_id=tenant_id, client_id=client_id, top_k=top_k,
            )
        elif filter_mode == "llm":
            results = self.filter_by_labels_llm(
                tenant_id=tenant_id, client_id=client_id,
                top_k=top_k, llm_model=llm_model,
            )
        elif filter_mode == "embedding":
            results = self.filter_by_embedding_similarity(
                tenant_id=tenant_id, client_id=client_id,
                top_k=top_k, similarity_threshold=similarity_threshold,
            )
        elif filter_mode == "full":
            # Stage 1: Label filter to get candidates
            label_results = self.filter_by_labels(
                tenant_id=tenant_id, client_id=client_id,
                top_k=top_k * 2,   # wider net for stage 1
            )
            if not label_results:
                results = []
            else:
                # Stage 2: Embedding similarity on label-filtered candidates
                embedding_results = self.filter_by_embedding_similarity(
                    tenant_id=tenant_id, client_id=client_id,
                    top_k=top_k, similarity_threshold=similarity_threshold,
                )

                # Intersect: keep only participants that passed both filters
                label_doc_ids = {r.document_id for r in label_results}
                label_scores = {r.document_id: r for r in label_results}

                results = []
                for emb_result in embedding_results:
                    if emb_result.document_id in label_doc_ids:
                        label_r = label_scores[emb_result.document_id]
                        # Combined score: average of label and embedding scores
                        combined_score = (
                            label_r.relevance_score + emb_result.relevance_score
                        ) / 2.0
                        results.append(PanelFilterResult(
                            document_id=emb_result.document_id,
                            participant_name=emb_result.participant_name,
                            relevance_score=round(combined_score, 4),
                            match_reasons=(
                                label_r.match_reasons + emb_result.match_reasons
                            ),
                            matched_labels=label_r.matched_labels,
                        ))

                results.sort(key=lambda r: r.relevance_score, reverse=True)
                results = results[:top_k]
        else:
            raise ValueError(
                f"Invalid filter_mode '{filter_mode}'. "
                "Must be 'label', 'llm', 'embedding', or 'full'."
            )

        # Count total panel participants evaluated
        panel_chunks = self._fetch_panel_chunks(tenant_id, client_id)
        total_evaluated = len(panel_chunks)

        return PanelFilterResponse(
            filter_mode=filter_mode,
            total_evaluated=total_evaluated,
            total_matched=len(results),
            results=results,
        )

"""
src/prompts/transcript_insights_prompts.py
--------------------------------------------
Prompt template for transcript summarisation and actionable insight extraction.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


TRANSCRIPT_INSIGHTS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a product strategy analyst. You are given raw WebVTT "
        "transcript excerpts from interviews or user-research sessions. "
        "Your job is to:\n\n"
        "1. **Summarise** the entire transcript into a clear, concise paragraph "
        "that captures the key topics discussed, the participant's overall "
        "sentiment, and any recurring themes.\n\n"
        "2. **Extract actionable insights** — concrete things mentioned by the "
        "participant that could be implemented to improve the client's product "
        "or service. Focus on:\n"
        "   - Feature requests or suggestions\n"
        "   - Pain points the participant experienced\n"
        "   - Workflow improvements they described\n"
        "   - UX issues or confusion they reported\n"
        "   - Any other improvement opportunities\n\n"
        "For each insight, include a supporting quote from the transcript.\n\n"
        "Respond ONLY with valid JSON in this exact structure:\n"
        "{{\n"
        '  "summary": "A paragraph summarising the full transcript.",\n'
        '  "actionable_insights": [\n'
        "    {{\n"
        '      "title": "Short label",\n'
        '      "description": "What the participant said/suggested and why it matters.",\n'
        '      "category": "feature_request|pain_point|improvement|workflow|ux|other",\n'
        '      "source_quote": "Verbatim quote from transcript",\n'
        '      "priority": "high|medium|low"\n'
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        "Rules:\n"
        "- The summary should be 3-6 sentences covering the full conversation\n"
        "- Extract 3-10 actionable insights depending on transcript length\n"
        "- source_quote must be actual text from the transcript, not invented\n"
        "- category must be one of: feature_request, pain_point, improvement, "
        "workflow, ux, other\n"
        "- priority: high = immediate impact / frequently mentioned, "
        "medium = valuable but not urgent, low = nice-to-have\n",
    ),
    (
        "human",
        "── TRANSCRIPT CONTENT ({transcript_count} transcripts, "
        "{chunk_count} chunks) ──\n"
        "{transcript_context}",
    ),
])

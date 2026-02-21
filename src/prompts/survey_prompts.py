"""
src/prompts/survey_prompts.py
-------------------------------
Prompt templates for survey generation.

Split into:
  - Agent context  — persona and rules for the survey designer
  - Form context   — per-question-type instructions (extensible via QUESTION_TYPE_PROMPTS)
  - Output format  — enforces the flat-array JSON schema
  - Assembled      — SURVEY_GENERATION_PROMPT combining all of the above
"""
from __future__ import annotations

from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate

# ── Context analysis prompt (LLM-powered insight extraction) ────────────────

CONTEXT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research analyst preparing insights for a survey designer. "
        "Given information about a tenant/organization and retrieved knowledge base content, "
        "extract relevant insights that should inform survey question design.\n\n"
        "Analyze the following dimensions and provide specific, actionable findings:\n"
        "1. **Industry context**: Key trends, challenges, and terminology specific to this industry\n"
        "2. **Organization scale**: How the headcount/revenue affects relevant concerns "
        "(e.g., enterprise vs SMB pain points)\n"
        "3. **Target persona**: What matters to this audience — their priorities, language level, "
        "and decision-making factors\n"
        "4. **Content themes**: Key topics, patterns, and gaps found in the knowledge base context\n"
        "5. **Suggested focus areas**: 3-5 specific themes the survey should explore based on "
        "all of the above\n\n"
        "Be concrete and specific — reference actual content from the knowledge base when possible. "
        "Do NOT generate survey questions. Just provide the analysis.",
    ),
    (
        "human",
        "Organization profile:\n{tenant_profile}\n\n"
        "Survey request: {request}\n\n"
        "Knowledge base context:\n{context}",
    ),
])

# ── Agent context (survey designer persona) ─────────────────────────────────

SURVEY_AGENT_SYSTEM_PROMPT = (
    "You are an expert survey designer. Generate a professional survey based on "
    "the user's request, the context analysis, and the organization profile provided.\n\n"
    "Rules:\n"
    "- Generate 5-15 questions depending on the scope\n"
    "- Questions MUST be informed by the context analysis — reference specific industry "
    "trends, organization characteristics, and content themes identified\n"
    "- Tailor vocabulary and complexity to the target persona\n"
    "- Each question should have a clear purpose tied to the analysis findings\n"
    "- Avoid leading or biased questions\n"
    "- Questions should be relevant to the organization's industry, scale, and audience\n\n"
)

# ── Form context (per question type) ────────────────────────────────────────

QUESTION_TYPE_PROMPTS: Dict[str, str] = {
    "multiple_choice": (
        "For multiple_choice questions:\n"
        "- Provide 3-6 answer options\n"
        "- Options should be mutually exclusive and collectively exhaustive\n"
        "- Include an 'Other' option when the list may not cover all possibilities\n"
        "- Order options logically (e.g., frequency: Never, Rarely, Sometimes, Often, Always)\n"
    ),
    # Future types:
    # "likert_scale": "...",
    # "open_ended": "...",
    # "rating": "...",
    # "yes_no": "...",
}


def get_question_type_instructions(question_types: List[str]) -> str:
    """Build the question-type instruction block for the given types."""
    parts = []
    for qt in question_types:
        if qt in QUESTION_TYPE_PROMPTS:
            parts.append(QUESTION_TYPE_PROMPTS[qt])
    return "\n".join(parts) if parts else ""


# ── Output format ───────────────────────────────────────────────────────────

SURVEY_OUTPUT_FORMAT_PROMPT = (
    "Return ONLY valid JSON as a flat array of question objects. "
    "Do NOT wrap in an outer object. Use this exact format:\n"
    "[\n"
    "  {{\n"
    '    "id": "<uuid4-string>",\n'
    '    "type": "multiple_choice",\n'
    '    "label": "Question text here?",\n'
    '    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],\n'
    '    "required": false\n'
    "  }}\n"
    "]\n\n"
    "Rules for the output:\n"
    '- "id" must be a valid UUID4 string\n'
    '- "type" must be "multiple_choice"\n'
    '- "label" is the question text\n'
    '- "options" is an array of option strings (3-6 options)\n'
    '- "required" is a boolean\n'
)

# ── Assembled generation prompt ─────────────────────────────────────────────

SURVEY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        SURVEY_AGENT_SYSTEM_PROMPT
        + "{question_type_instructions}\n\n"
        + SURVEY_OUTPUT_FORMAT_PROMPT
        + "{profile_section}",
    ),
    (
        "human",
        "Survey request: {request}\n\n"
        "Context analysis:\n{context_analysis}\n\n"
        "Raw knowledge base context:{context_section}",
    ),
])

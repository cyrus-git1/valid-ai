"""
src/prompts/survey_prompts.py
-------------------------------
Prompt templates for survey generation.

Split into:
  - Agent context  — persona and rules for the survey designer
  - Form context   — per-question-type instructions (extensible via QUESTION_TYPE_PROMPTS)
  - Output format  — enforces the flat-array JSON schema for all supported types
  - Assembled      — SURVEY_GENERATION_PROMPT combining all of the above
"""
from __future__ import annotations

from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate

# Re-export prompt templates used by survey_workflow
__all__ = [
    "ALL_QUESTION_TYPES",
    "CONTEXT_ANALYSIS_PROMPT",
    "FOLLOW_UP_SURVEY_PROMPT",
    "QUESTION_RECOMMENDATION_PROMPT",
    "SURVEY_GENERATION_PROMPT",
    "SURVEY_TITLE_PROMPT",
    "SURVEY_DESCRIPTION_PROMPT",
    "get_question_type_instructions",
]

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
    "- Use a DIVERSE MIX of question types from the allowed types listed below — "
    "do not rely only on simple types like multiple_choice and rating. "
    "Actively consider interactive types like card_sort, ranking, tree_testing, "
    "and matrix when the topic involves categorization, prioritization, grouping, "
    "navigation testing, or multi-dimensional evaluation\n"
    "- Questions MUST be informed by the context analysis — reference specific industry "
    "trends, organization characteristics, and content themes identified\n"
    "- Tailor vocabulary and complexity to the target persona\n"
    "- Each question should have a clear purpose tied to the analysis findings\n"
    "- Avoid leading or biased questions\n"
    "- Questions should be relevant to the organization's industry, scale, and audience\n"
    "- If previously generated questions are provided, use them as suggestions: "
    "build on their themes, avoid duplicating their exact topics, and fill gaps "
    "they left. You may refine or rephrase a prior question if it fits better, "
    "but do not copy them verbatim\n\n"
)

# ── Form context (per question type) ────────────────────────────────────────

QUESTION_TYPE_PROMPTS: Dict[str, str] = {
    "multiple_choice": (
        "For multiple_choice questions:\n"
        "- Provide 3-6 answer options in the \"options\" array\n"
        "- Options should be mutually exclusive and collectively exhaustive\n"
        "- Include an 'Other' option when the list may not cover all possibilities\n"
        "- Order options logically (e.g., frequency: Never, Rarely, Sometimes, Often, Always)\n"
    ),
    "checkbox": (
        "For checkbox questions (select all that apply):\n"
        "- Provide 3-6 options in the \"options\" array\n"
        "- Options are NOT mutually exclusive — respondents can select multiple\n"
        "- Use when asking about preferences, features used, or multi-select categories\n"
    ),
    "short_text": (
        "For short_text questions:\n"
        "- Use for brief open-ended responses (names, one-line answers)\n"
        "- No options or items needed — only \"id\", \"type\", \"label\", and \"required\"\n"
    ),
    "long_text": (
        "For long_text questions:\n"
        "- Use for detailed open-ended feedback or explanations\n"
        "- No options or items needed — only \"id\", \"type\", \"label\", and \"required\"\n"
    ),
    "rating": (
        "For rating questions:\n"
        "- Include \"min\" (integer, typically 1) and \"max\" (integer, typically 5)\n"
        "- Include \"lowLabel\" (e.g., \"Poor\") and \"highLabel\" (e.g., \"Excellent\")\n"
        "- Use for satisfaction, agreement, or quality scales\n"
    ),
    "yes_no": (
        "For yes_no questions:\n"
        "- Use for simple binary questions\n"
        "- No options needed — only \"id\", \"type\", \"label\", and \"required\"\n"
    ),
    "nps": (
        "For nps (Net Promoter Score) questions:\n"
        "- Use for likelihood-to-recommend questions (0-10 scale, handled by frontend)\n"
        "- No options needed — only \"id\", \"type\", \"label\", and \"required\"\n"
    ),
    "ranking": (
        "For ranking questions:\n"
        "- Provide 3-6 items to rank in the \"items\" array (array of strings)\n"
        "- Respondents will drag-and-drop to order them by preference\n"
        "- Use when you need to understand relative priorities or preferences "
        "(e.g., ranking features, values, or concerns by importance)\n"
    ),
    "card_sort": (
        "For card_sort questions:\n"
        "- Provide \"items\" as an array of objects, each with \"id\" (UUID4) and \"label\"\n"
        "- Provide \"categories\" as an array of objects, each with \"id\" (UUID4) and \"label\"\n"
        "- Respondents sort cards into the defined categories\n"
        "- Use 3-6 cards and 2-4 categories\n"
        "- Use card_sort when respondents need to classify or categorize concepts "
        "(e.g., sorting tasks into urgency levels, grouping features into themes, "
        "categorizing pain points by department, or mapping skills to proficiency tiers)\n"
        "- Card sort is especially valuable for understanding how respondents mentally "
        "organize information and for gathering categorization data\n"
    ),
    "tree_testing": (
        "For tree_testing questions:\n"
        "- Provide a \"task\" string that tells the respondent what to find "
        "(e.g., \"Where would you find account settings?\")\n"
        "- Provide a \"tree\" as a nested array of nodes. Each node has \"id\" (UUID4), "
        "\"label\", and \"children\" (array of child nodes, may be empty)\n"
        "- Build a realistic information-architecture tree with 2-4 top-level categories "
        "and 1-3 levels of nesting\n"
        "- Optionally provide \"correctPath\" as an array of node IDs tracing the ideal "
        "path from root to the correct answer (may be empty if there is no single correct answer)\n"
        "- Use tree_testing when you want to evaluate how easily users can navigate "
        "a hierarchy to locate information — ideal for testing site navigation, menu "
        "structures, or content organization\n"
    ),
    "matrix": (
        "For matrix questions:\n"
        "- Provide \"rows\" as an array of strings — the statements or items to rate\n"
        "- Provide \"columns\" as an array of strings — the scale points or answer options\n"
        "- Use 3-6 rows and 3-5 columns\n"
        "- Use matrix when respondents need to evaluate multiple items on the same scale "
        "(e.g., rating satisfaction across several service dimensions, agreement with "
        "multiple statements, or importance of various features)\n"
        "- Matrix questions reduce survey length by combining related items into one view\n"
    ),
    "sus": (
        "For sus (System Usability Scale) questions:\n"
        "- This is a standardized 10-question usability questionnaire — "
        "no custom rows, columns, or options are needed\n"
        "- Only provide \"id\", \"type\": \"sus\", \"label\", and \"required\"\n"
        "- The label should describe what system or product is being evaluated "
        "(e.g., \"System Usability Scale — [Product Name]\")\n"
        "- Use sus when measuring perceived usability of a product, tool, or system\n"
        "- Limit to ONE sus question per survey since it already contains 10 sub-statements\n"
    ),
}

ALL_QUESTION_TYPES = list(QUESTION_TYPE_PROMPTS.keys())


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
    "Do NOT wrap in an outer object. Each question MUST have \"id\" (UUID4), "
    "\"type\", \"label\", and \"required\" (boolean).\n\n"
    "Use the EXACT schema for each type as shown below:\n\n"
    "multiple_choice:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "multiple_choice", "label": "Question?",\n'
    '  "options": ["Option 1", "Option 2", "Option 3", "Option 4"],\n'
    '  "required": true\n'
    "}}\n\n"
    "checkbox:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "checkbox", "label": "Select all that apply?",\n'
    '  "options": ["Option 1", "Option 2", "Option 3", "Option 4"],\n'
    '  "required": false\n'
    "}}\n\n"
    "short_text:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "short_text", "label": "Your answer?",\n'
    '  "required": false\n'
    "}}\n\n"
    "long_text:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "long_text", "label": "Describe in detail?",\n'
    '  "required": false\n'
    "}}\n\n"
    "rating:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "rating", "label": "Rate this?",\n'
    '  "min": 1, "max": 5, "lowLabel": "Poor", "highLabel": "Excellent",\n'
    '  "required": false\n'
    "}}\n\n"
    "yes_no:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "yes_no", "label": "Is this true?",\n'
    '  "required": false\n'
    "}}\n\n"
    "nps:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "nps", "label": "How likely are you to recommend?",\n'
    '  "required": false\n'
    "}}\n\n"
    "ranking:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "ranking", "label": "Rank these items",\n'
    '  "items": ["Item 1", "Item 2", "Item 3"],\n'
    '  "required": false\n'
    "}}\n\n"
    "card_sort:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "card_sort", "label": "Sort these cards",\n'
    '  "items": [\n'
    '    {{"id": "<uuid4>", "label": "Card 1"}},\n'
    '    {{"id": "<uuid4>", "label": "Card 2"}}\n'
    '  ],\n'
    '  "categories": [\n'
    '    {{"id": "<uuid4>", "label": "Category A"}},\n'
    '    {{"id": "<uuid4>", "label": "Category B"}}\n'
    '  ],\n'
    '  "required": true\n'
    "}}\n\n"
    "tree_testing:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "tree_testing",\n'
    '  "label": "Find the item",\n'
    '  "task": "Where would you find account settings?",\n'
    '  "tree": [\n'
    '    {{"id": "<uuid4>", "label": "Category 1", "children": [\n'
    '      {{"id": "<uuid4>", "label": "Sub-item 1", "children": []}},\n'
    '      {{"id": "<uuid4>", "label": "Sub-item 2", "children": []}}\n'
    '    ]}},\n'
    '    {{"id": "<uuid4>", "label": "Category 2", "children": [\n'
    '      {{"id": "<uuid4>", "label": "Sub-item 3", "children": []}}\n'
    '    ]}}\n'
    '  ],\n'
    '  "correctPath": [],\n'
    '  "required": false\n'
    "}}\n\n"
    "matrix:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "matrix",\n'
    '  "label": "Rate the following",\n'
    '  "rows": ["Statement 1", "Statement 2", "Statement 3"],\n'
    '  "columns": ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],\n'
    '  "required": false\n'
    "}}\n\n"
    "sus:\n"
    "{{\n"
    '  "id": "<uuid4>", "type": "sus",\n'
    '  "label": "System Usability Scale",\n'
    '  "required": false\n'
    "}}\n\n"
    "Rules for the output:\n"
    '- Every "id" must be a valid UUID4 string (generate unique ones)\n'
    '- "type" must be one of: multiple_choice, checkbox, short_text, long_text, '
    "rating, yes_no, nps, ranking, card_sort, tree_testing, matrix, sus\n"
    '- "label" is the question text\n'
    '- "required" is a boolean\n'
    "- Only include fields that belong to the question type (see schemas above)\n"
    "- For card_sort, every item and category must have its own unique UUID4 id\n"
    "- For tree_testing, every tree node must have its own unique UUID4 id\n"
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
        "Raw knowledge base context:{context_section}"
        "{prior_questions_section}"
        "{title_description_section}",
    ),
])


# ── Question recommendation prompt ──────────────────────────────────────────

QUESTION_RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert survey designer acting as a recommendation engine. "
        "Given a survey goal and the questions already created, suggest additional "
        "questions that would strengthen the survey.\n\n"
        "Your recommendations should:\n"
        "- Fill gaps in topic coverage that the existing questions miss\n"
        "- Improve the survey's ability to surface actionable insights\n"
        "- Complement (not duplicate) the existing questions\n"
        "- Follow up on interesting angles opened by current questions\n"
        "- Actively diversify question types — if the existing survey lacks interactive "
        "types like card_sort, ranking, tree_testing, or matrix, strongly prefer "
        "recommending those. Card sort is ideal for categorization, tree testing for "
        "navigation findability, and matrix for multi-dimensional evaluation\n\n"
        "Return a JSON object with two keys:\n"
        "  \"reasoning\": a short paragraph explaining why these questions are recommended\n"
        "  \"questions\": a JSON array of question objects\n\n"
        "{question_type_instructions}\n\n"
        + SURVEY_OUTPUT_FORMAT_PROMPT
        + "{profile_section}",
    ),
    (
        "human",
        "Survey goal: {request}\n\n"
        "Number of recommendations requested: {count}\n\n"
        "Questions already in the survey:\n{existing_questions_text}\n\n"
        "Context analysis:\n{context_analysis}\n\n"
        "Raw knowledge base context:{context_section}",
    ),
])


# ── Follow-up survey prompt ─────────────────────────────────────────────────

# ── Survey title generation prompt ─────────────────────────────────────────

SURVEY_TITLE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert survey designer. Given business context about an "
        "organization and a survey request, generate a concise, professional "
        "survey title that clearly communicates the purpose of the survey.\n\n"
        "Rules:\n"
        "- The title should be 5-12 words\n"
        "- It should be specific to the organization's context and goals\n"
        "- Use clear, professional language — avoid jargon unless industry-appropriate\n"
        "- The title should help respondents immediately understand what the survey "
        "is about\n"
        "- If survey questions are provided, ensure the title accurately reflects "
        "the themes and scope of those questions\n"
        "- If a survey description is provided, ensure the title complements it "
        "without repeating it\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        '{{"title": "Your Survey Title Here"}}',
    ),
    (
        "human",
        "Survey request: {request}\n\n"
        "Context analysis:\n{context_analysis}"
        "{profile_section}"
        "{questions_section}"
        "{description_section}",
    ),
])


# ── Survey description generation prompt ──────────────────────────────────

SURVEY_DESCRIPTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research designer. Generate a concise description "
        "of the study itself — what is being researched and why.\n\n"
        "Rules:\n"
        "- 1-2 sentences, no more than 40 words\n"
        "- Describe the purpose and scope of the study in neutral, third-person language\n"
        "- Do NOT address or thank respondents — this is an internal study description, "
        "not a welcome message\n"
        "- Do NOT use phrases like \"Thank you\", \"Your insights\", \"We appreciate\", "
        "\"We aim to\", or any respondent-facing language\n"
        "- State what is being studied and the key dimensions being explored\n"
        "- Use clear, professional tone\n"
        "- If a survey title is provided, ensure the description complements it "
        "without repeating it\n"
        "- If survey questions are provided, ensure the description accurately "
        "reflects the themes and scope of those questions\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        '{{"description": "Your study description here."}}',
    ),
    (
        "human",
        "{title_section}"
        "Survey request: {request}\n\n"
        "Context analysis:\n{context_analysis}"
        "{profile_section}"
        "{questions_section}",
    ),
])


FOLLOW_UP_SURVEY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert survey designer specializing in follow-up research. "
        "Given a completed survey and its response summaries, design follow-up "
        "questions that dig deeper into the findings.\n\n"
        "Your follow-up questions should:\n"
        "- Explore surprising or notable patterns in the response data\n"
        "- Probe the 'why' behind quantitative results\n"
        "- Clarify ambiguous findings from the original survey\n"
        "- Capture qualitative depth on topics where responses clustered\n"
        "- Explore new angles that the original survey didn't cover but the "
        "responses hinted at\n\n"
        "Return a JSON object with two keys:\n"
        "  \"reasoning\": a short paragraph explaining how these follow-up "
        "questions build on the original survey findings\n"
        "  \"questions\": a JSON array of question objects\n\n"
        "{question_type_instructions}\n\n"
        + SURVEY_OUTPUT_FORMAT_PROMPT
        + "{profile_section}",
    ),
    (
        "human",
        "Original survey goal: {request}\n\n"
        "Number of follow-up questions requested: {count}\n\n"
        "Completed survey questions with response summaries:\n"
        "{completed_survey_text}\n\n"
        "Context analysis:\n{context_analysis}\n\n"
        "Raw knowledge base context:{context_section}",
    ),
])

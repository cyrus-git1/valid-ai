"""
src/tools/survey_tools.py
---------------------------
LangChain tools for survey generation sub-agent.
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def format_survey_as_json(
    title: str,
    description: str,
    questions: str,
) -> str:
    """Format a survey into a structured JSON string.
    questions should be a pipe-separated list of question strings.
    Each question will be assigned an auto-incrementing ID."""
    import json
    q_list = [q.strip() for q in questions.split("|") if q.strip()]
    survey = {
        "title": title,
        "description": description,
        "questions": [
            {"id": i + 1, "text": q, "type": "open_ended"}
            for i, q in enumerate(q_list)
        ],
    }
    return json.dumps(survey, indent=2)


@tool
def validate_survey_questions(questions: str) -> str:
    """Validate survey questions for quality. Takes pipe-separated questions.
    Returns feedback on question quality and suggestions for improvement."""
    q_list = [q.strip() for q in questions.split("|") if q.strip()]
    feedback = []
    for i, q in enumerate(q_list, 1):
        issues = []
        if len(q) < 10:
            issues.append("too short — may lack clarity")
        if not q.endswith("?"):
            issues.append("does not end with '?'")
        if q.lower().startswith(("do you", "are you", "is it", "can you", "will you")):
            issues.append("yes/no question — consider rephrasing as open-ended")
        if len(q) > 200:
            issues.append("very long — consider splitting into multiple questions")
        status = "OK" if not issues else "; ".join(issues)
        feedback.append(f"Q{i}: {status}")
    return "\n".join(feedback)

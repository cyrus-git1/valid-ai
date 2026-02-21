"""
src/prompts/router_prompts.py
-------------------------------
Prompt templates for the intent classification router agent.

Provides two prompts:
  - INTENT_CLASSIFICATION_PROMPT  — first-pass classification with confidence
  - INTENT_CLASSIFICATION_RETRY_PROMPT — retry prompt with prior-attempt context
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ── Intent Classification (with confidence) ─────────────────────────────────

INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an intent classifier. Given a user message, classify it into "
        "exactly one category and provide a confidence score.\n\n"
        "Categories:\n"
        "- retrieval: user wants to search, ask a question, or look up information\n"
        "- survey: user wants to generate, create, or design a survey or questionnaire\n"
        "- ingest: user wants to upload, add, or import new documents or websites\n"
        "- unknown: cannot determine intent\n\n"
        "Respond with ONLY valid JSON in this exact format:\n"
        '{{"intent": "<category>", "confidence": <float between 0.0 and 1.0>}}\n\n'
        "The confidence score should reflect how certain you are about the classification. "
        "A score of 1.0 means completely certain, 0.5 means uncertain.",
    ),
    ("human", "{input}"),
])

# ── Retry Classification ────────────────────────────────────────────────────

INTENT_CLASSIFICATION_RETRY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an intent classifier. The previous classification had low confidence. "
        "Please carefully re-read the user message and classify it into exactly one category.\n\n"
        "Categories:\n"
        "- retrieval: user wants to search, ask a question, or look up information\n"
        "- survey: user wants to generate, create, or design a survey or questionnaire\n"
        "- ingest: user wants to upload, add, or import new documents or websites\n"
        "- unknown: cannot determine intent\n\n"
        "Previous attempt classified as: {previous_intent} (confidence: {previous_confidence})\n\n"
        "Respond with ONLY valid JSON in this exact format:\n"
        '{{"intent": "<category>", "confidence": <float between 0.0 and 1.0>}}\n\n'
        "Be more decisive. If the message could fit multiple categories, pick the strongest match.",
    ),
    ("human", "{input}"),
])

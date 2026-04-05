"""
src/config/llm.py
-------------------
Central registry of LLM model names, temperatures, and factory.

Change a model here and every call-site picks it up automatically.
Environment variables override the defaults so you can switch models
per deployment without touching code.

Usage
-----
    from src.config.llm import get_llm

    llm = get_llm("survey_generation")
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


class LLMConfig:
    """Single source of truth for model identifiers and temperatures."""

    # ── Model names ──────────────────────────────────────────────────────────
    # Each constant can be overridden by an env var of the same name prefixed
    # with LLM_, e.g.  LLM_DEFAULT="gpt-4o"  overrides DEFAULT.

    DEFAULT: str            = os.environ.get("LLM_DEFAULT",            "gpt-4o-mini")

    # Survey workflow
    CONTEXT_ANALYSIS: str   = os.environ.get("LLM_CONTEXT_ANALYSIS",   DEFAULT)
    SURVEY_GENERATION: str  = os.environ.get("LLM_SURVEY_GENERATION",  DEFAULT)
    SURVEY_TITLE: str       = os.environ.get("LLM_SURVEY_TITLE",       DEFAULT)
    SURVEY_DESCRIPTION: str = os.environ.get("LLM_SURVEY_DESCRIPTION", DEFAULT)
    QUESTION_REC: str       = os.environ.get("LLM_QUESTION_REC",       DEFAULT)
    FOLLOW_UP: str          = os.environ.get("LLM_FOLLOW_UP",          DEFAULT)

    # RAG / retrieval
    RAG_ANSWER: str         = os.environ.get("LLM_RAG_ANSWER",         DEFAULT)

    # Agents
    ROUTER: str             = os.environ.get("LLM_ROUTER",             DEFAULT)
    PERSONA: str            = os.environ.get("LLM_PERSONA",            DEFAULT)
    ENRICHMENT: str         = os.environ.get("LLM_ENRICHMENT",         DEFAULT)

    # ── Default temperatures ─────────────────────────────────────────────────

    class Temp:
        ANALYSIS: float   = 0.2
        GENERATION: float = 0.3
        CREATIVE: float   = 0.4
        ROUTING: float    = 0.0
        PRECISE: float    = 0.0


# ── Role → (model, temperature) mapping ──────────────────────────────────────

_ROLE_MAP: dict[str, tuple[str, float]] = {
    "default":            (LLMConfig.DEFAULT,            LLMConfig.Temp.PRECISE),
    "context_analysis":   (LLMConfig.CONTEXT_ANALYSIS,   LLMConfig.Temp.ANALYSIS),
    "survey_generation":  (LLMConfig.SURVEY_GENERATION,  LLMConfig.Temp.GENERATION),
    "survey_title":       (LLMConfig.SURVEY_TITLE,       LLMConfig.Temp.GENERATION),
    "survey_description": (LLMConfig.SURVEY_DESCRIPTION, LLMConfig.Temp.GENERATION),
    "question_rec":       (LLMConfig.QUESTION_REC,       LLMConfig.Temp.CREATIVE),
    "follow_up":          (LLMConfig.FOLLOW_UP,          LLMConfig.Temp.CREATIVE),
    "rag_answer":         (LLMConfig.RAG_ANSWER,         LLMConfig.Temp.PRECISE),
    "router":             (LLMConfig.ROUTER,             LLMConfig.Temp.ROUTING),
    "persona":            (LLMConfig.PERSONA,            LLMConfig.Temp.ANALYSIS),
    "enrichment":         (LLMConfig.ENRICHMENT,         LLMConfig.Temp.ANALYSIS),
}


def get_llm(role: str = "default", **overrides) -> ChatOpenAI:
    """Create a ChatOpenAI instance for the given role.

    Parameters
    ----------
    role : str
        A key from _ROLE_MAP (e.g. "survey_generation", "router").
    **overrides
        Any kwarg accepted by ChatOpenAI (model, temperature, etc.)
        that should override the role defaults.
    """
    model, temperature = _ROLE_MAP.get(role, _ROLE_MAP["default"])
    return ChatOpenAI(
        model=overrides.pop("model", model),
        temperature=overrides.pop("temperature", temperature),
        api_key=overrides.pop("api_key", os.environ.get("OPENAI_API_KEY")),
        **overrides,
    )

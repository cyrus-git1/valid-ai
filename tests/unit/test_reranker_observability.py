"""Reranker observability: the previously-silent 'rerank disabled' case is now
detectable. These are hermetic unit checks (no Cohere, no DB)."""
from __future__ import annotations


def test_reranker_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    from src.services import reranker_service
    svc = reranker_service.RerankerService()
    assert svc.is_available() is False


def test_rerank_is_safe_noop_when_unavailable(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    from src.services import reranker_service
    svc = reranker_service.RerankerService()
    cands = [{"content": "alpha"}, {"content": "beta"}]
    out = svc.rerank("q", cands, text_field="content")
    assert out == cands  # unchanged order, no exception


def test_health_response_exposes_reranker_flag():
    from src.models.api.admin import HealthResponse
    h = HealthResponse(status="ok", supabase=True, openai=True, reranker=False)
    assert h.reranker is False
    # defaults to False so an older caller that omits it still reads as "disabled"
    assert HealthResponse(status="ok", supabase=True, openai=True).reranker is False

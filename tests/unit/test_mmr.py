"""Unit tests for MMR selection (the semantic_dedup replacement)."""
from __future__ import annotations

from src.services.retrieval_postprocess import _norm, mmr


def _item(id_, vec, **kw):
    d = {"id": id_, "embedding": vec}
    d.update(kw)
    return d


def test_norm_parses_pgvector_string():
    # pgvector comes back from PostgREST as a string "[...]"
    a = _norm("[3.0, 0.0, 0.0, 0.0]")
    assert a is not None
    assert abs(float(a[0]) - 1.0) < 1e-6      # normalized
    assert _norm("not json") is None
    assert _norm(None) is None


def test_mmr_pure_relevance_when_lambda_1():
    q = [1.0, 0.0]
    cands = [
        _item("a", [0.9, 0.1]),
        _item("b", [1.0, 0.0]),   # most relevant to q
        _item("c", [0.2, 0.9]),
    ]
    out = mmr(q, cands, lambda_param=1.0, top_k=3)
    assert out[0]["id"] == "b"                # highest cosine to query goes first


def test_mmr_penalizes_near_duplicates():
    q = [1.0, 0.0]
    cands = [
        _item("a", [1.0, 0.0]),    # top relevance
        _item("a_dup", [0.999, 0.001]),   # near-identical to a
        _item("diverse", [0.0, 1.0]),     # orthogonal
    ]
    # lambda<0.5 makes the diversity penalty outweigh a near-dup's relevance:
    #   score(a_dup) = λ·1 − (1−λ)·~1 < score(diverse) = λ·0 − (1−λ)·0  ⇔  λ<0.5
    out = mmr(q, cands, lambda_param=0.3, top_k=2)
    ids = [c["id"] for c in out]
    assert ids[0] == "a"
    assert ids[1] == "diverse"             # the near-dup is demoted below the diverse one


def test_mmr_uses_rerank_score_as_relevance():
    q = [0.0, 1.0]                          # query would favor 'b' by cosine…
    cands = [
        _item("a", [1.0, 0.0], rerank_score=0.99),   # …but reranker says 'a' wins
        _item("b", [0.0, 1.0], rerank_score=0.10),
    ]
    out = mmr(q, cands, lambda_param=1.0, top_k=1)
    assert out[0]["id"] == "a"             # rerank_score overrides raw cosine


def test_mmr_honors_top_k_and_handles_missing_embeddings():
    q = [1.0, 0.0]
    cands = [
        _item("a", [1.0, 0.0]),
        _item("b", None),                  # no embedding — still selectable
        _item("c", [0.5, 0.5]),
    ]
    out = mmr(q, cands, lambda_param=0.7, top_k=2)
    assert len(out) == 2
    assert all("id" in c for c in out)


def test_mmr_empty_and_zero_k():
    assert mmr([1.0, 0.0], [], top_k=5) == []
    assert mmr([1.0, 0.0], [_item("a", [1.0, 0.0])], top_k=0) == []

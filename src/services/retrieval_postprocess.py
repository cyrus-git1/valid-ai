"""
Post-retrieval transforms applied to the candidate set before returning.

Currently:
  - semantic_dedup(items, threshold)  — drops near-duplicate chunks/nodes
    by pairwise cosine on their embeddings. Naive O(n^2) which is fine for
    top-K=50 size pools; not appropriate for full-table dedup.

Designed to be cheap and side-effect free: pass in a list, get a filtered
list back. No DB calls.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _norm(vec: Any) -> Optional[np.ndarray]:
    if vec is None:
        return None
    # pgvector columns come back from PostgREST as a string like "[0.1,-0.2,...]".
    if isinstance(vec, str):
        try:
            vec = json.loads(vec)
        except Exception:
            return None
    a = np.asarray(vec, dtype=np.float32)
    if a.ndim != 1 or a.size == 0:
        return None
    n = float(np.linalg.norm(a))
    if n == 0.0:
        return None
    return a / n


def semantic_dedup(
    items: List[Dict[str, Any]],
    *,
    threshold: float = 0.95,
    embedding_key: str = "embedding",
) -> List[Dict[str, Any]]:
    """
    Return items in original rank order with near-duplicates dropped.

    For each item (in input order), keep it iff its embedding's cosine
    similarity to every already-kept item is < threshold. Items without
    an embedding are always kept (cheaper to keep than to risk dropping
    something we can't compare).

    Caller is responsible for ranking the input list first — dedup
    preserves the order, so what's kept is "best representative within
    each cluster of near-duplicates."
    """
    if not items or threshold <= 0 or threshold > 1:
        return items

    kept: List[Dict[str, Any]] = []
    kept_vecs: List[np.ndarray] = []

    for item in items:
        v = item.get(embedding_key)
        normed = _norm(v) if v is not None else None
        if normed is None:
            kept.append(item)
            continue
        is_dup = False
        for other in kept_vecs:
            if float(np.dot(normed, other)) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
            kept_vecs.append(normed)
    return kept


def mmr(
    query_embedding: Any,
    candidates: List[Dict[str, Any]],
    *,
    lambda_param: float = 0.7,
    top_k: int = 8,
    embedding_key: str = "embedding",
    relevance_key: str = "rerank_score",
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance selection — the principled replacement for
    semantic_dedup. Iteratively picks the candidate maximizing

        lambda * relevance(d)  -  (1 - lambda) * max_{s in selected} cos(d, s)

    relevance(d) = candidates[d][relevance_key] if present (e.g. a reranker's
    score), else cos(query, d). lambda=1.0 is pure relevance (no diversity);
    lower lambda pushes diversity. Candidates without a usable embedding keep
    their input order among themselves and are appended after scored ones.

    Returns the top_k selected, in MMR order. Pure; no DB calls.
    """
    if not candidates:
        return candidates
    if top_k <= 0:
        return []

    q = _norm(query_embedding)

    docs: List[Dict[str, Any]] = []
    for c in candidates:
        vec = _norm(c.get(embedding_key))
        rel_val = c.get(relevance_key)
        if rel_val is not None:
            rel = float(rel_val)
        elif q is not None and vec is not None:
            rel = float(np.dot(q, vec))
        else:
            rel = 0.0
        docs.append({"c": c, "vec": vec, "rel": rel})

    selected: List[Dict[str, Any]] = []
    selected_vecs: List[np.ndarray] = []
    remaining = list(range(len(docs)))
    k = min(top_k, len(docs))

    while len(selected) < k and remaining:
        best_i = None
        best_score = None
        for i in remaining:
            d = docs[i]
            if not selected_vecs or d["vec"] is None:
                diversity = 0.0
            else:
                diversity = max(float(np.dot(d["vec"], sv)) for sv in selected_vecs)
            score = lambda_param * d["rel"] - (1.0 - lambda_param) * diversity
            if best_score is None or score > best_score:
                best_score = score
                best_i = i
        chosen = docs[best_i]
        selected.append(chosen["c"])
        if chosen["vec"] is not None:
            selected_vecs.append(chosen["vec"])
        remaining.remove(best_i)

    return selected

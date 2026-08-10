"""
src/services/embedding_service.py
---------------------------------
OpenAI embedding helpers with per-tenant quota accounting.

Extracted from ingest_router so services and routers no longer import embedding
helpers from a router module. Callers pass an optional ``tenant_id`` to charge
tokens against that tenant's daily quota (and reconcile against actual usage).

    from src.services.embedding_service import embed_in_batches, EMBED_MODEL
"""
from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI

from src.services.tenant_plan_service import (
    EmbeddingQuotaService,
    TenantPlanService,
    estimate_tokens,
)
from src.db.supabase_client import get_supabase

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 64


def embed_texts(texts: List[str], *, tenant_id=None) -> List[List[float]]:
    """Call OpenAI embeddings. If `tenant_id` is set, charge tokens against
    that tenant's daily quota (and reconcile against actual usage).
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    quota_svc: Optional[EmbeddingQuotaService] = None
    estimate = 0
    if tenant_id:
        quota_svc = EmbeddingQuotaService(TenantPlanService(get_supabase()))
        estimate = estimate_tokens(texts)
        quota_svc.check_and_consume(str(tenant_id), estimate)

    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)

    # Reconcile actual vs estimate (OpenAI reports prompt_tokens)
    if quota_svc and tenant_id:
        try:
            actual = int(getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0)
            if actual:
                quota_svc.reconcile_actual(str(tenant_id), estimate, actual)
        except Exception:
            pass

    return [d.embedding for d in resp.data]


def embed_in_batches(texts: List[str], *, tenant_id=None) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        out.extend(embed_texts(texts[i:i + EMBED_BATCH_SIZE], tenant_id=tenant_id))
    return out

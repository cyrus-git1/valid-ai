"""
src/services/embedding_service.py
----------------------------------
Handles OpenAI embedding generation.
"""
from __future__ import annotations

import os
from typing import List

from openai import OpenAI
import dotenv

dotenv.load_dotenv()


class EmbeddingService:
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def embed_texts(self, texts: List[str], model: str | None = None) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        resp = self.client.embeddings.create(
            model=model or self.model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    def embed_in_batches(
        self, texts: List[str], *, model: str | None = None, batch_size: int = 64
    ) -> List[List[float]]:
        """Embed texts in batches to avoid timeouts."""
        out: List[List[float]] = []
        m = model or self.model
        for i in range(0, len(texts), batch_size):
            out.extend(self.embed_texts(texts[i : i + batch_size], model=m))
        return out


# Module-level convenience for backward compat (used by ingest_service)
_default = EmbeddingService()


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    return _default.embed_texts(texts, model=model)

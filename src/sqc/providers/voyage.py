"""Voyage AI embeddings and reranking.

Chosen because Anthropic publishes no embedding model, so an embedding
vendor is required regardless of which model answers.
"""

from __future__ import annotations

import os

import httpx

from sqc.providers.base import (
    EmbeddingBatch,
    EmbeddingDimensionError,
    ProviderAuthError,
    ProviderResponseError,
    RerankedItem,
    Usage,
)
from sqc.providers.http import HttpProviderClient

VOYAGE_BASE_URL = "https://api.voyageai.com/v1"
MAX_BATCH = 96
"""Documents per request. The API caps batch size and total tokens, and a
rejected oversized batch costs a whole round trip, so batches stay modest."""


def _require_key(explicit: str | None) -> str:
    key = explicit or os.environ.get("VOYAGE_API_KEY", "")
    if not key:
        raise ProviderAuthError("VOYAGE_API_KEY is not set; add it to .env")
    return key


class VoyageEmbedder:
    def __init__(
        self,
        model: str = "voyage-3",
        dimension: int = 1024,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        **client_kwargs: object,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self._client = HttpProviderClient(
            VOYAGE_BASE_URL,
            {"Authorization": f"Bearer {_require_key(api_key)}",
             "Content-Type": "application/json"},
            transport=transport,
            **client_kwargs,  # type: ignore[arg-type]
        )

    def _embed(self, texts: list[str], input_type: str) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(vectors=(), model=self.model)

        vectors: list[tuple[float, ...]] = []
        usage = Usage()
        for start in range(0, len(texts), MAX_BATCH):
            batch = texts[start : start + MAX_BATCH]
            payload = {"input": batch, "model": self.model, "input_type": input_type}
            body = self._client.post_json("/embeddings", payload)
            data = body.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise ProviderResponseError(
                    f"voyage returned {len(data) if isinstance(data, list) else 'no'} "
                    f"embeddings for {len(batch)} inputs"
                )
            # Order is not guaranteed to match the request, so sort by the
            # index the API returns rather than trusting position.
            for item in sorted(data, key=lambda d: d.get("index", 0)):
                vector = item.get("embedding")
                if not isinstance(vector, list):
                    raise ProviderResponseError("voyage returned a malformed embedding")
                if len(vector) != self.dimension:
                    raise EmbeddingDimensionError(
                        f"{self.model} returned {len(vector)} dimensions but the database "
                        f"column is vector({self.dimension}); re-run scripts/init_db.py "
                        "with a matching SQC_EMBEDDING_DIM and re-index"
                    )
                vectors.append(tuple(float(v) for v in vector))
            usage = usage + Usage(
                input_tokens=int(body.get("usage", {}).get("total_tokens", 0)), requests=1
            )
        return EmbeddingBatch(vectors=tuple(vectors), model=self.model, usage=usage)

    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> tuple[float, ...]:
        batch = self._embed([text], "query")
        if not batch.vectors:
            raise ProviderResponseError("voyage returned no embedding for the query")
        return batch.vectors[0]


class VoyageReranker:
    def __init__(
        self,
        model: str = "rerank-2",
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        **client_kwargs: object,
    ) -> None:
        self.model = model
        self._client = HttpProviderClient(
            VOYAGE_BASE_URL,
            {"Authorization": f"Bearer {_require_key(api_key)}",
             "Content-Type": "application/json"},
            transport=transport,
            **client_kwargs,  # type: ignore[arg-type]
        )

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankedItem]:
        if not documents:
            return []
        body = self._client.post_json(
            "/rerank",
            {
                "query": query,
                "documents": documents,
                "model": self.model,
                "top_k": min(top_k, len(documents)),
            },
        )
        results = body.get("data")
        if not isinstance(results, list):
            raise ProviderResponseError("voyage rerank returned no data")
        items = []
        for entry in results:
            index = entry.get("index")
            score = entry.get("relevance_score")
            if not isinstance(index, int) or not 0 <= index < len(documents):
                raise ProviderResponseError(f"voyage rerank returned out-of-range index {index}")
            items.append(RerankedItem(index=index, score=float(score or 0.0)))
        items.sort(key=lambda item: (-item.score, item.index))
        return items[:top_k]

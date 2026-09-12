"""Deterministic fakes so the whole pipeline, including evals, runs in CI
with no network and no keys.

The embedding fake is a hashing bag-of-words vectoriser, not random noise.
That distinction matters: with random vectors every retrieval test passes
or fails by luck, so the tests would prove only that the plumbing runs.
Lexical hashing gives real, repeatable ranking behaviour, which is enough
to test fusion, thresholds and refusal logic.

What it deliberately does not do is model meaning. It will not match "MFA"
to "multi-factor authentication". Semantic quality is measured against a
real provider in the eval suite; these fakes test mechanics.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Callable

from sqc.providers.base import (
    EmbeddingBatch,
    EmbeddingDimensionError,
    LLMResponse,
    RerankedItem,
    Usage,
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Signed-hash bag-of-words embedding. Deterministic across processes.

    Uses blake2b rather than Python's hash() because hash() is randomised
    per interpreter run by PYTHONHASHSEED, which would make stored vectors
    incomparable between the ingestion run and the query run.
    """

    def __init__(self, dimension: int = 1024, model: str = "fake-hashing-v1") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model = model
        self.call_count = 0

    def _vector(self, text: str) -> tuple[float, ...]:
        tokens = _tokenize(text)
        counts = Counter(tokens)
        vector = [0.0] * self.dimension
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            position = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            # Sublinear term frequency: a word repeated twenty times should
            # not dominate a chunk the way raw counts would.
            vector[position] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(v / norm for v in vector)

    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        self.call_count += 1
        vectors = tuple(self._vector(t) for t in texts)
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingDimensionError(
                    f"expected {self.dimension}, produced {len(vector)}"
                )
        tokens = sum(len(_tokenize(t)) for t in texts)
        return EmbeddingBatch(
            vectors=vectors, model=self.model, usage=Usage(input_tokens=tokens, requests=1)
        )

    def embed_query(self, text: str) -> tuple[float, ...]:
        self.call_count += 1
        return self._vector(text)


class LexicalReranker:
    """Overlap-based reranker standing in for a cross-encoder.

    Scores by the share of query tokens present in the document, which
    correlates loosely with relevance and is completely repeatable.
    """

    model = "fake-lexical-rerank-v1"

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankedItem]:
        query_tokens = set(_tokenize(query))
        scored: list[RerankedItem] = []
        for index, document in enumerate(documents):
            document_tokens = set(_tokenize(document))
            if not query_tokens:
                score = 0.0
            else:
                overlap = len(query_tokens & document_tokens)
                # Mild length penalty so a long chunk cannot win purely by
                # containing every word in the language.
                score = overlap / len(query_tokens)
                score *= 1.0 / (1.0 + math.log1p(len(document_tokens) / 50.0))
            scored.append(RerankedItem(index=index, score=round(score, 6)))
        # Stable tie-break on original order keeps results reproducible.
        scored.sort(key=lambda item: (-item.score, item.index))
        return scored[:top_k]


REFUSAL_RESPONSE: dict[str, Any] = {
    "claims": [],
    "supported": False,
    "refusal_reason": "no scripted response for this question",
}


class ScriptedLLM:
    """LLM fake driven by a lookup table of substring -> structured response.

    Unscripted questions return a refusal rather than a plausible answer.
    A test that forgets to script something then fails by refusing, which
    is the safe direction for this product: an accidental invented answer
    passing a test would be the worst possible failure mode to normalise.
    """

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        default: dict[str, Any] | None = None,
        handler: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.model = "fake-scripted-llm"
        self.responses = responses or {}
        self.default = default if default is not None else REFUSAL_RESPONSE
        self.handler = handler
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.calls.append((system, user, schema))
        if self.handler is not None:
            data = self.handler(system, user)
        else:
            data = next(
                (value for key, value in self.responses.items() if key.lower() in user.lower()),
                self.default,
            )
        return LLMResponse(
            data=dict(data),
            model=self.model,
            usage=Usage(
                input_tokens=len(_tokenize(system)) + len(_tokenize(user)),
                output_tokens=len(_tokenize(str(data))),
                requests=1,
            ),
            stop_reason="tool_use",
        )

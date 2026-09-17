"""Provider boundary: embeddings, reranking, answering.

Three protocols rather than one, because they change independently.
Anthropic sells no embedding model, so the answering model and the
embedding model will always come from different vendors. Coupling them
behind a single "AI provider" abstraction would be wrong on day one.

Structural typing (Protocol) rather than base classes: a provider is
anything with the right methods, so the fakes are not subclasses of the
real clients and cannot inherit their behaviour by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Base for every provider failure. Never swallowed silently."""


class ProviderAuthError(ProviderError):
    """Missing, malformed or rejected credentials."""


class ProviderRateLimitError(ProviderError):
    """Rate limited after exhausting retries."""


class ProviderQuotaError(ProviderRateLimitError):
    """The allowance is gone, not merely throttled.

    Separate from rate limiting because only one of them is worth waiting
    for. Throttling clears in seconds; an exhausted daily quota does not
    clear until it resets, so retrying it spends the remaining allowance on
    calls that cannot succeed."""


class ProviderResponseError(ProviderError):
    """Reachable but returned something unusable."""


class EmbeddingDimensionError(ProviderError):
    """Vector width does not match the configured dimension.

    Its own error type because the failure is silent otherwise: vectors of
    the wrong width either fail deep inside pgvector or, worse, come from a
    different model and compare as noise against everything already stored.
    """


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting, so cost per answered question is measurable."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.requests + other.requests,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    usage: Usage = field(default_factory=Usage)

    @property
    def dimension(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


@dataclass(frozen=True, slots=True)
class RerankedItem:
    index: int
    """Position in the list originally passed in, not a chunk id."""
    score: float
    """Provider relevance score. Comparable within one call only, which is
    why the retrieval layer keeps it as a confidence signal and never as a
    cross-query threshold without calibration."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    data: dict[str, Any]
    """Parsed structured output conforming to the requested schema."""
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


@runtime_checkable
class EmbeddingProvider(Protocol):
    model: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        """Embed stored content."""

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed a question.

        Separate from embed_documents because modern embedding models are
        asymmetric: they take an input_type hint, and using the document
        setting for queries measurably degrades retrieval.
        """


@runtime_checkable
class RerankProvider(Protocol):
    model: str

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankedItem]:
        """Reorder candidates by relevance, returning at most top_k."""


@runtime_checkable
class LLMProvider(Protocol):
    model: str

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Return output conforming to `schema`.

        Structured-only by design. There is no free-text completion method,
        so no caller can accidentally take prose from the model and treat it
        as an answer without the citations and claim structure the validator
        depends on.
        """

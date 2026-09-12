"""Build providers from configuration.

The only place in the codebase that maps a config string to a concrete
client. Everything else takes a protocol, so swapping a vendor is a change
to .env, not to the pipeline.
"""

from __future__ import annotations

from sqc.config import Settings, get_settings
from sqc.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    ProviderError,
    RerankProvider,
)
from sqc.providers.fake import HashingEmbedder, LexicalReranker, ScriptedLLM


class NoOpReranker:
    """Passthrough for running without a reranker.

    Preserves fusion order and assigns a flat score. Explicitly opt-in via
    SQC_RERANK_PROVIDER=none, so nobody loses reranking by accident: the
    scores it returns are not relevance signals and confidence must not be
    derived from them.
    """

    model = "none"

    def rerank(self, query: str, documents: list[str], top_k: int):  # noqa: ANN201
        from sqc.providers.base import RerankedItem

        return [RerankedItem(index=i, score=0.0) for i in range(min(top_k, len(documents)))]


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    name = settings.embedding_provider.lower()
    if name == "fake":
        return HashingEmbedder(dimension=settings.embedding_dim)
    if name == "voyage":
        from sqc.providers.voyage import VoyageEmbedder

        return VoyageEmbedder(model=settings.embedding_model, dimension=settings.embedding_dim)
    raise ProviderError(
        f"unknown SQC_EMBEDDING_PROVIDER '{settings.embedding_provider}'; "
        "supported: voyage, fake"
    )


def build_rerank_provider(settings: Settings | None = None) -> RerankProvider:
    settings = settings or get_settings()
    name = settings.rerank_provider.lower()
    if name == "fake":
        return LexicalReranker()
    if name == "none":
        return NoOpReranker()
    if name == "voyage":
        from sqc.providers.voyage import VoyageReranker

        return VoyageReranker(model=settings.rerank_model)
    raise ProviderError(
        f"unknown SQC_RERANK_PROVIDER '{settings.rerank_provider}'; "
        "supported: voyage, none, fake"
    )


def build_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    name = settings.llm_provider.lower()
    if name == "fake":
        return ScriptedLLM()
    if name == "anthropic":
        from sqc.providers.anthropic_llm import AnthropicLLM

        return AnthropicLLM(model=settings.llm_model)
    raise ProviderError(
        f"unknown SQC_LLM_PROVIDER '{settings.llm_provider}'; supported: anthropic, fake"
    )

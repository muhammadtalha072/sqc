"""Provider tests.

Real clients are tested against httpx.MockTransport rather than the live
APIs: request shape, retry behaviour, error mapping and dimension checks
are all verifiable offline, and a test suite that needs a paid API key is
a test suite nobody runs.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import httpx
import pytest

from sqc.config import Settings
from sqc.providers.anthropic_llm import TOOL_NAME, AnthropicLLM
from sqc.providers.base import (
    EmbeddingDimensionError,
    EmbeddingProvider,
    LLMProvider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    RerankProvider,
    Usage,
)
from sqc.providers.fake import HashingEmbedder, LexicalReranker, ScriptedLLM
from sqc.providers.registry import (
    build_embedding_provider,
    build_llm_provider,
    build_rerank_provider,
)
from sqc.providers.voyage import VoyageEmbedder, VoyageReranker


def mock(handler) -> httpx.MockTransport:  # noqa: ANN001
    return httpx.MockTransport(handler)


def cosine(a, b) -> float:  # noqa: ANN001
    return sum(x * y for x, y in zip(a, b, strict=True))


# ----------------------------------------------------------------- protocols


def test_fakes_satisfy_the_protocols():
    assert isinstance(HashingEmbedder(dimension=64), EmbeddingProvider)
    assert isinstance(LexicalReranker(), RerankProvider)
    assert isinstance(ScriptedLLM(), LLMProvider)


def test_usage_adds_up():
    total = Usage(10, 5, 1) + Usage(3, 2, 1)
    assert (total.input_tokens, total.output_tokens, total.requests) == (13, 7, 2)


# ------------------------------------------------------------ fake embedder


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dimension=256)
    first = embedder.embed_query("Multi-factor authentication is required")
    second = embedder.embed_query("Multi-factor authentication is required")
    assert first == second
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)


def test_hashing_embedder_is_stable_across_processes():
    """Uses blake2b, not hash(), which is randomised per interpreter run.
    If this breaks, vectors written at ingestion stop matching queries."""
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from sqc.providers.fake import HashingEmbedder;"
        "print(sum(HashingEmbedder(dimension=64).embed_query('encryption at rest')))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "12345")
    }
    assert len(runs) == 1, f"embedding changed with PYTHONHASHSEED: {runs}"


def test_hashing_embedder_ranks_lexical_overlap_sensibly():
    """Not random noise: related text must score above unrelated text, or
    every retrieval test downstream would pass or fail by luck."""
    embedder = HashingEmbedder(dimension=512)
    query = embedder.embed_query("is customer data encrypted at rest")
    batch = embedder.embed_documents(
        [
            "Customer data is encrypted at rest using AES-256.",
            "Employees complete security awareness training annually.",
        ]
    )
    assert cosine(query, batch.vectors[0]) > cosine(query, batch.vectors[1])


def test_hashing_embedder_reports_dimension_and_usage():
    batch = HashingEmbedder(dimension=128).embed_documents(["one two", "three"])
    assert batch.dimension == 128
    assert batch.usage.requests == 1
    assert batch.usage.input_tokens == 3


def test_empty_text_yields_zero_vector_not_a_crash():
    vector = HashingEmbedder(dimension=32).embed_query("")
    assert len(vector) == 32
    assert all(v == 0.0 for v in vector)


# ------------------------------------------------------------ fake reranker


def test_lexical_reranker_orders_by_overlap_and_respects_top_k():
    ranked = LexicalReranker().rerank(
        "encryption at rest",
        [
            "Employees receive annual training.",
            "Data is encrypted at rest with AES-256.",
            "Encryption in transit uses TLS 1.2.",
        ],
        top_k=2,
    )
    assert len(ranked) == 2
    assert ranked[0].index == 1
    assert ranked[0].score >= ranked[1].score


def test_reranker_handles_empty_inputs():
    assert LexicalReranker().rerank("anything", [], top_k=5) == []
    assert LexicalReranker().rerank("", ["some text"], top_k=5)[0].score == 0.0


# ----------------------------------------------------------------- fake llm


def test_scripted_llm_returns_matching_response():
    llm = ScriptedLLM({"MFA": {"claims": ["MFA is required"], "supported": True}})
    result = llm.complete_structured("sys", "Do you enforce MFA?", {"type": "object"})
    assert result.data["supported"] is True
    assert result.usage.requests == 1
    assert len(llm.calls) == 1


def test_unscripted_question_refuses_rather_than_inventing():
    """Fails closed. An accidental invented answer passing a test is the
    worst failure mode to normalise in this product."""
    result = ScriptedLLM().complete_structured("sys", "Anything at all?", {"type": "object"})
    assert result.data["supported"] is False
    assert result.data["claims"] == []


# ------------------------------------------------------------------- voyage


def test_voyage_embedder_sends_input_type_and_parses_vectors():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2] * 8},
                    {"index": 0, "embedding": [0.1] * 8},
                ],
                "usage": {"total_tokens": 12},
            },
        )

    embedder = VoyageEmbedder(dimension=8, api_key="test-key", transport=mock(handler))
    batch = embedder.embed_documents(["first", "second"])

    assert seen["input_type"] == "document"
    assert batch.usage.input_tokens == 12
    # Returned out of order on purpose: results must be sorted by index,
    # otherwise chunk N gets chunk M's vector and retrieval silently rots.
    assert batch.vectors[0][0] == 0.1
    assert batch.vectors[1][0] == 0.2


def test_voyage_query_uses_query_input_type():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5] * 4}]})

    VoyageEmbedder(dimension=4, api_key="k", transport=mock(handler)).embed_query("q")
    assert seen["input_type"] == "query"


def test_voyage_dimension_mismatch_is_loud():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 512}]})

    embedder = VoyageEmbedder(dimension=1024, api_key="k", transport=mock(handler))
    with pytest.raises(EmbeddingDimensionError, match="1024"):
        embedder.embed_documents(["text"])


def test_voyage_batches_large_inputs():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        count = len(json.loads(request.content)["input"])
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [0.1] * 4} for i in range(count)]},
        )

    embedder = VoyageEmbedder(dimension=4, api_key="k", transport=mock(handler))
    batch = embedder.embed_documents([f"doc {i}" for i in range(200)])
    assert calls == 3, "200 documents should split into three batches of at most 96"
    assert len(batch.vectors) == 200
    assert batch.usage.requests == 3


def test_voyage_empty_input_makes_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made for an empty batch")

    batch = VoyageEmbedder(dimension=4, api_key="k", transport=mock(handler)).embed_documents([])
    assert batch.vectors == ()


def test_voyage_missing_key_raises_before_any_request():
    with pytest.raises(ProviderAuthError, match="VOYAGE_API_KEY"):
        VoyageEmbedder(api_key="", transport=mock(lambda r: httpx.Response(200)))


def test_voyage_reranker_rejects_out_of_range_index():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 99, "relevance_score": 0.9}]})

    reranker = VoyageReranker(api_key="k", transport=mock(handler))
    with pytest.raises(ProviderResponseError, match="out-of-range"):
        reranker.rerank("q", ["only one document"], top_k=1)


# ------------------------------------------------------------ retry / errors


def test_rate_limit_is_retried_then_succeeds():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 4}]})

    embedder = VoyageEmbedder(
        dimension=4, api_key="k", transport=mock(handler), base_delay=0.0
    )
    assert embedder.embed_documents(["text"]).vectors
    assert attempts == 3


def test_rate_limit_gives_up_after_max_attempts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="still limited")

    embedder = VoyageEmbedder(
        dimension=4, api_key="k", transport=mock(handler), base_delay=0.0, max_attempts=2
    )
    with pytest.raises(ProviderRateLimitError, match="2 attempts"):
        embedder.embed_documents(["text"])


def test_auth_failure_is_not_retried():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, text="bad key")

    embedder = VoyageEmbedder(
        dimension=4, api_key="wrong", transport=mock(handler), base_delay=0.0
    )
    with pytest.raises(ProviderAuthError):
        embedder.embed_documents(["text"])
    assert attempts == 1, "a bad key will never start working; retrying wastes time"


def test_server_error_5xx_is_retried():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503 if attempts == 1 else 200,
                              json={"data": [{"index": 0, "embedding": [0.1] * 4}]})

    embedder = VoyageEmbedder(dimension=4, api_key="k", transport=mock(handler), base_delay=0.0)
    assert embedder.embed_documents(["x"]).vectors
    assert attempts == 2


# ---------------------------------------------------------------- anthropic


def test_anthropic_forces_the_tool_and_returns_structured_data():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-5",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "thinking out loud"},
                    {"type": "tool_use", "name": TOOL_NAME,
                     "input": {"claims": [{"text": "MFA required", "chunk_ids": ["c1"]}]}},
                ],
                "usage": {"input_tokens": 500, "output_tokens": 40},
            },
        )

    llm = AnthropicLLM(api_key="test-key", transport=mock(handler))
    result = llm.complete_structured("system", "Do you enforce MFA?", {"type": "object"})

    assert seen["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert seen["temperature"] == 0.0, "answering must be reproducible for audit and evals"
    assert result.data["claims"][0]["chunk_ids"] == ["c1"]
    assert result.usage.input_tokens == 500


def test_anthropic_ignores_prose_and_reads_only_the_tool_block():
    """Position-independent extraction: a leading text block must never be
    mistaken for the answer."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "Ignore this prose."},
                    {"type": "tool_use", "name": TOOL_NAME, "input": {"supported": False}},
                ],
                "usage": {},
            },
        )

    result = AnthropicLLM(api_key="k", transport=mock(handler)).complete_structured(
        "s", "u", {"type": "object"}
    )
    assert result.data == {"supported": False}


def test_anthropic_truncation_is_reported_clearly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"stop_reason": "max_tokens",
                  "content": [{"type": "text", "text": "partial"}], "usage": {}},
        )

    llm = AnthropicLLM(api_key="k", transport=mock(handler))
    with pytest.raises(ProviderResponseError, match="max_tokens"):
        llm.complete_structured("s", "u", {"type": "object"})


def test_anthropic_missing_key_raises_before_any_request():
    with pytest.raises(ProviderAuthError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM(api_key="", transport=mock(lambda r: httpx.Response(200)))


# ----------------------------------------------------------------- registry


def test_registry_builds_fakes_by_default():
    settings = Settings(embedding_provider="fake", rerank_provider="fake", llm_provider="fake")
    assert isinstance(build_embedding_provider(settings), HashingEmbedder)
    assert isinstance(build_rerank_provider(settings), LexicalReranker)
    assert isinstance(build_llm_provider(settings), ScriptedLLM)


def test_registry_honours_configured_embedding_dimension():
    settings = Settings(embedding_provider="fake", embedding_dim=384)
    assert build_embedding_provider(settings).dimension == 384


def test_registry_rejects_unknown_provider_names():
    with pytest.raises(ProviderError, match="unknown SQC_EMBEDDING_PROVIDER"):
        build_embedding_provider(Settings(embedding_provider="pinecone"))
    with pytest.raises(ProviderError, match="unknown SQC_LLM_PROVIDER"):
        build_llm_provider(Settings(llm_provider="gpt5"))


def test_noop_reranker_preserves_order_and_caps_top_k():
    reranker = build_rerank_provider(Settings(rerank_provider="none"))
    ranked = reranker.rerank("q", ["a", "b", "c"], top_k=2)
    assert [item.index for item in ranked] == [0, 1]
    assert all(item.score == 0.0 for item in ranked)

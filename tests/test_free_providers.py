"""Tests for the two zero-cost providers.

Gemini is exercised against httpx.MockTransport, like the other real
clients: request shape and error mapping are verifiable offline, and a test
that needs a live quota is a test that fails on a bad wifi day.
"""

from __future__ import annotations

import io
import json

import httpx
import pytest

from sqc.config import Settings
from sqc.core.answering.schema import ANSWER_SCHEMA
from sqc.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from sqc.providers.gemini import TOOL_NAME, GeminiLLM, _clean_schema
from sqc.providers.manual import ManualLLM, extract_json
from sqc.providers.registry import build_llm_provider


def mock(handler) -> httpx.MockTransport:  # noqa: ANN001
    return httpx.MockTransport(handler)


def gemini_ok(args: dict) -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"functionCall": {"name": TOOL_NAME, "args": args}}]},
                "finishReason": "STOP",
            }
        ],
        "modelVersion": "gemini-2.5-flash",
        "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 60},
    }


# ------------------------------------------------------------------ gemini


def test_gemini_satisfies_the_llm_protocol():
    assert isinstance(GeminiLLM(api_key="k", transport=mock(lambda r: httpx.Response(200))),
                      LLMProvider)


def test_gemini_forces_a_function_call_and_returns_structured_data():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert request.headers["x-goog-api-key"] == "test-key"
        assert "generateContent" in str(request.url)
        return httpx.Response(200, json=gemini_ok({"answer": "Yes.", "claims": []}))

    llm = GeminiLLM(api_key="test-key", transport=mock(handler))
    result = llm.complete_structured("system", "user", ANSWER_SCHEMA)

    assert seen["tool_config"]["function_calling_config"]["mode"] == "ANY"
    assert seen["tool_config"]["function_calling_config"]["allowed_function_names"] == [TOOL_NAME]
    assert seen["generationConfig"]["temperature"] == 0.0
    assert seen["system_instruction"]["parts"][0]["text"] == "system"
    assert result.data == {"answer": "Yes.", "claims": []}
    assert result.usage.input_tokens == 900


def test_gemini_schema_is_stripped_of_unsupported_keywords():
    """The API takes an OpenAPI subset and rejects keywords it does not
    know, rather than ignoring them."""
    cleaned = _clean_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {"a": {"type": "string", "pattern": "^x", "enum": ["x"]}},
            "required": ["a"],
        }
    )
    assert "additionalProperties" not in cleaned
    assert "$schema" not in cleaned
    assert "pattern" not in cleaned["properties"]["a"]
    assert cleaned["properties"]["a"]["enum"] == ["x"]


def test_answer_schema_survives_cleaning_intact():
    cleaned = _clean_schema(ANSWER_SCHEMA)
    assert set(cleaned["properties"]) == set(ANSWER_SCHEMA["properties"])
    assert cleaned["required"] == ANSWER_SCHEMA["required"]
    assert cleaned["properties"]["claims"]["items"]["required"] == ["text", "evidence_ids"]


def test_gemini_truncation_is_reported_clearly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]},
        )

    llm = GeminiLLM(api_key="k", transport=mock(handler))
    with pytest.raises(ProviderResponseError, match="maxOutputTokens"):
        llm.complete_structured("s", "u", ANSWER_SCHEMA)


def test_gemini_safety_block_is_surfaced():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    llm = GeminiLLM(api_key="k", transport=mock(handler))
    with pytest.raises(ProviderResponseError, match="SAFETY"):
        llm.complete_structured("s", "u", ANSWER_SCHEMA)


def test_gemini_throttling_is_retried():
    """A plain 429 clears in seconds and is worth waiting for."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, text="Too many requests, slow down")

    llm = GeminiLLM(api_key="k", transport=mock(handler), base_delay=0.0, max_attempts=3)
    with pytest.raises(ProviderRateLimitError):
        llm.complete_structured("s", "u", ANSWER_SCHEMA)
    assert attempts == 3


def test_gemini_exhausted_quota_is_not_retried():
    """An exhausted daily allowance does not clear until it resets, so
    retrying spends the rest of the allowance on calls that cannot succeed.
    An eight-attempt budget against a 250-request free tier consumed most of
    a day's quota in a single evaluation run."""
    from sqc.providers.base import ProviderQuotaError

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, json={"error": {
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded for quota metric 'Generate requests per day'",
        }})

    llm = GeminiLLM(api_key="k", transport=mock(handler), base_delay=0.0, max_attempts=8)
    with pytest.raises(ProviderQuotaError, match="quota exhausted"):
        llm.complete_structured("s", "u", ANSWER_SCHEMA)
    assert attempts == 1, "a quota error must cost exactly one request"


def test_gemini_missing_key_raises_before_any_request():
    with pytest.raises(ProviderAuthError, match="aistudio"):
        GeminiLLM(api_key="", transport=mock(lambda r: httpx.Response(200)))


def test_registry_does_not_send_an_anthropic_model_id_to_gemini():
    """SQC_LLM_MODEL defaults to a Claude name; sending it to Gemini would
    404 mid-run."""
    import os

    settings = Settings(llm_provider="gemini", llm_model="claude-sonnet-5")

    os.environ["GEMINI_API_KEY"] = "test"
    try:
        provider = build_llm_provider(settings)
        assert "gemini" in provider.model
    finally:
        os.environ.pop("GEMINI_API_KEY", None)


# ------------------------------------------------------------------ manual


@pytest.mark.parametrize(
    "pasted",
    [
        '{"answer": "Yes", "claims": []}',
        '```json\n{"answer": "Yes", "claims": []}\n```',
        '```\n{"answer": "Yes", "claims": []}\n```',
        'Here is the JSON:\n\n{"answer": "Yes", "claims": []}\n\nHope that helps!',
    ],
    ids=["bare", "json-fence", "plain-fence", "chatty"],
)
def test_manual_extracts_json_from_how_chat_windows_actually_reply(pasted):
    assert extract_json(pasted) == {"answer": "Yes", "claims": []}


def test_manual_rejects_unparseable_paste():
    with pytest.raises(ProviderResponseError, match="not valid JSON"):
        extract_json("I think the answer is probably yes")


def test_manual_rejects_a_json_array():
    with pytest.raises(ProviderResponseError, match="not an object"):
        extract_json("[1, 2, 3]")


def test_manual_prints_the_prompt_and_reads_the_reply():
    out = io.StringIO()
    payload = {"answer": "Data is encrypted.", "answer_type": "yes",
               "claims": [{"text": "Data is encrypted.", "evidence_ids": ["E1"]}],
               "evidence_sufficient": True, "reason": "stated"}
    llm = ManualLLM(stream_in=io.StringIO(json.dumps(payload)), stream_out=out)

    result = llm.complete_structured("SYSTEM RULES", "<question>Q</question>", ANSWER_SCHEMA)

    printed = out.getvalue()
    assert "SYSTEM RULES" in printed
    assert "<question>Q</question>" in printed
    assert "evidence_ids" in printed, "the schema must be shown so the reply can match it"
    assert result.data == payload
    assert result.stop_reason == "manual"


def test_manual_empty_paste_fails_rather_than_inventing():
    llm = ManualLLM(stream_in=io.StringIO("   \n"), stream_out=io.StringIO())
    with pytest.raises(ProviderResponseError, match="no response"):
        llm.complete_structured("s", "u", ANSWER_SCHEMA)


def test_manual_output_flows_through_the_validator_unchanged():
    """The point of this provider: a real model's output, checked by the
    real validator. A hallucinated handle must still be caught."""
    import uuid

    from sqc.core.answering.schema import AnswerStatus
    from sqc.core.answering.validator import validate
    from sqc.core.retrieval.types import Evidence

    evidence = [
        Evidence(evidence_id="E1", chunk_ids=(uuid.uuid4(),), document_id=uuid.uuid4(),
                 filename="p.pdf", text="Data is encrypted at rest.", heading_path=("Enc",),
                 page_start=1, page_end=1, effective_date=None, score=1.0, citation="p.pdf p1")
    ]
    pasted = extract_json(
        '```json\n{"answer": "We are ISO 27001 certified.", "answer_type": "yes",'
        ' "claims": [{"text": "ISO 27001 certified.", "evidence_ids": ["E7"]}],'
        ' "evidence_sufficient": true, "reason": "x"}\n```'
    )
    outcome = validate(pasted, evidence)
    status, claims, errors = outcome.status, outcome.claims, outcome.errors
    assert status is AnswerStatus.REFUSED
    assert claims[0].supported is False
    assert any("E7" in e for e in errors)


def test_registry_builds_manual_without_any_key():
    provider = build_llm_provider(Settings(llm_provider="manual"))
    assert isinstance(provider, ManualLLM)


# ------------------------------------------------- credentials from .env


def test_provider_keys_are_read_from_unprefixed_env_names(monkeypatch):
    """Regression: keys written in .env as GEMINI_API_KEY were invisible.

    env_prefix='SQC_' turned the setting into SQC_GEMINI_API_KEY, and the
    providers read os.environ directly, so a key sitting in .env produced
    'GEMINI_API_KEY is not set' with the key plainly in the file.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-from-env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-from-env")

    settings = Settings()
    assert settings.gemini_api_key == "AIza-from-env"
    assert settings.anthropic_api_key == "sk-ant-from-env"
    assert settings.voyage_api_key == "pa-from-env"


def test_registry_passes_the_configured_key_to_the_provider(monkeypatch):
    """The key must reach the client from settings, not only via os.environ,
    or .env-only setups fail while exported-shell setups appear to work."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(llm_provider="gemini", gemini_api_key="AIza-from-settings")
    provider = build_llm_provider(settings)
    assert provider.model.startswith("gemini")


def test_missing_key_everywhere_still_raises_clearly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="aistudio"):
        build_llm_provider(Settings(llm_provider="gemini", gemini_api_key=""))

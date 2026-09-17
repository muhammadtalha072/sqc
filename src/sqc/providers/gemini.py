"""Google Gemini via the Generative Language API.

Exists because the free tier makes a real model call cost nothing, and the
question this project needs answered - does a real model refuse when the
evidence is absent - does not depend on which model answers.

Two things to know before pointing this at anything sensitive:

  - Free-tier inputs and outputs may be used by Google to improve their
    models. That is acceptable for testing against a published policy and
    unacceptable for a customer's security documentation.
  - Enabling billing on a Gemini project removes its free tier entirely,
    so keep testing and production on separate projects.

Structured output uses function calling with mode ANY, the same approach as
the Anthropic client: the contract is enforced by the API, so no amount of
instruction-like text inside a customer document can talk the model out of
the schema.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from sqc.providers.base import (
    LLMResponse,
    ProviderAuthError,
    ProviderResponseError,
    Usage,
)
from sqc.providers.http import HttpProviderClient

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = "gemini-2.5-flash"
TOOL_NAME = "emit_answer"

# The API accepts an OpenAPI 3.0 subset, not full JSON Schema, and rejects
# keywords it does not know rather than ignoring them.
_ALLOWED_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum",
    "properties", "required", "items",
}


def _clean_schema(node: Any) -> Any:
    """Strip JSON Schema keywords the Gemini function declaration rejects.

    `properties` needs special handling: its keys are field names, not schema
    keywords. Filtering them against the keyword allow-list empties the
    schema entirely, which the API accepts and then answers with nothing
    useful - a failure that would look like a model problem, not a bug here.
    """
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key not in _ALLOWED_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {name: _clean_schema(sub) for name, sub in value.items()}
            else:
                cleaned[key] = _clean_schema(value)
        return cleaned
    if isinstance(node, list):
        return [_clean_schema(item) for item in node]
    return node


class GeminiLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        **client_kwargs: object,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ProviderAuthError(
                "GEMINI_API_KEY is not set; create a free key at aistudio.google.com "
                "and add it to .env"
            )
        self.model = model
        self._key = key
        self._client = HttpProviderClient(
            GEMINI_BASE_URL,
            {"Content-Type": "application/json", "x-goog-api-key": key},
            transport=transport,
            **client_kwargs,  # type: ignore[arg-type]
        )

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> LLMResponse:
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": TOOL_NAME,
                            "description": "Return the answer in the required structure.",
                            "parameters": _clean_schema(schema),
                        }
                    ]
                }
            ],
            # ANY forces a function call, removing the option of prose.
            "tool_config": {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [TOOL_NAME],
                }
            },
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens},
        }
        body = self._client.post_json(
            f"/v1beta/models/{self.model}:generateContent", payload
        )

        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            blocked = (body.get("promptFeedback") or {}).get("blockReason")
            raise ProviderResponseError(
                f"gemini returned no candidates"
                + (f" (blocked: {blocked})" if blocked else "")
            )

        first = candidates[0]
        parts = (first.get("content") or {}).get("parts") or []
        args = next(
            (
                part["functionCall"].get("args")
                for part in parts
                if isinstance(part, dict)
                and isinstance(part.get("functionCall"), dict)
                and part["functionCall"].get("name") == TOOL_NAME
            ),
            None,
        )
        if not isinstance(args, dict):
            finish = first.get("finishReason")
            if finish == "MAX_TOKENS":
                raise ProviderResponseError(
                    "gemini hit maxOutputTokens before completing the structured answer; "
                    "raise max_tokens or send fewer evidence chunks"
                )
            raise ProviderResponseError(
                f"gemini did not call {TOOL_NAME} (finishReason={finish})"
            )

        usage_block = body.get("usageMetadata") or {}
        return LLMResponse(
            data=args,
            model=str(body.get("modelVersion") or self.model),
            usage=Usage(
                input_tokens=int(usage_block.get("promptTokenCount", 0)),
                output_tokens=int(usage_block.get("candidatesTokenCount", 0)),
                requests=1,
            ),
            stop_reason=first.get("finishReason"),
        )

"""A provider where you are the model.

Prints the exact prompt the pipeline built, waits while you paste it into
any chat window, and reads the JSON you paste back. Costs nothing and needs
no account.

This is not a toy. It exercises the real prompt, the real schema and the
real validator, with output from a real model. The measurement that matters
at this stage - whether a capable model refuses when the evidence does not
support an answer, and whether the validator catches it when it does not -
is fully available this way. It is only unsuitable for volume: the eval
suite in Step 7 needs an automated provider.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, TextIO

from sqc.providers.base import LLMResponse, ProviderResponseError, Usage

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of pasted text.

    Chat interfaces wrap JSON in code fences and often add a sentence before
    or after it. Both are stripped rather than treated as errors, since
    fighting that would make the provider useless for its one purpose.
    """
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(f"pasted text was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProviderResponseError("pasted JSON was not an object")
    return parsed


class ManualLLM:
    """Human-in-the-loop provider. Reads a pasted response from stdin."""

    def __init__(
        self,
        model: str = "manual",
        stream_in: TextIO | None = None,
        stream_out: TextIO | None = None,
    ) -> None:
        self.model = model
        self._in = stream_in or sys.stdin
        self._out = stream_out or sys.stderr

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> LLMResponse:
        write = self._out.write
        write("\n" + "=" * 72 + "\n")
        write("COPY EVERYTHING BETWEEN THE MARKERS INTO ANY CHAT WINDOW\n")
        write("=" * 72 + "\n\n")
        write("----- BEGIN PROMPT -----\n")
        write(system.strip() + "\n\n")
        write(user.strip() + "\n\n")
        write(
            "Reply with a single JSON object and nothing else, matching this schema:\n"
        )
        write(json.dumps(schema, indent=2) + "\n")
        write("----- END PROMPT -----\n\n")
        write("Paste the JSON reply below, then press Ctrl-D (Ctrl-Z on Windows):\n")
        self._out.flush()

        pasted = self._in.read()
        if not pasted.strip():
            raise ProviderResponseError("no response was pasted")

        data = extract_json(pasted)
        return LLMResponse(
            data=data,
            model=self.model,
            usage=Usage(
                input_tokens=len((system + user).split()),
                output_tokens=len(pasted.split()),
                requests=1,
            ),
            stop_reason="manual",
        )

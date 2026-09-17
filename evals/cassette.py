"""Record real model responses once, replay them forever.

An eval suite that costs money and quota every run is an eval suite that
gets run once. Worse, it is not a regression test at all: the model can
change under you, so a failure could mean your code broke or could mean
the provider shipped a new checkpoint, and you cannot tell which.

Recording separates those questions. Replay mode holds the model constant
so a diff in results is a diff in your code. Re-recording is a deliberate
act that answers the other question: what changed when the model changed.

Keys are a hash of the exact prompt, so any prompt edit misses the cache
and is visible as a miss rather than silently scored against stale output.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from sqc.providers.base import LLMProvider, LLMResponse, ProviderResponseError, Usage


def prompt_key(system: str, user: str, model: str) -> str:
    digest = hashlib.sha256()
    for part in (model, system, user):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


class CassetteLLM:
    """Wraps a real provider, caching responses on disk.

    modes:
      replay  - cache only; a miss is an error, so a run either uses the
                recorded model output or tells you it cannot
      record  - call the provider for misses and store the result
      refresh - call the provider for everything and overwrite
    """

    def __init__(
        self,
        inner: LLMProvider | None,
        directory: str | pathlib.Path,
        mode: str = "record",
        model: str | None = None,
    ) -> None:
        if mode not in ("replay", "record", "refresh"):
            raise ValueError(f"unknown cassette mode '{mode}'")
        self.inner = inner
        self.mode = mode
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # The cache key includes the model, so replay must use the model that
        # was recorded rather than a placeholder. Reading it off `inner` alone
        # meant replay - which has no inner provider - keyed every lookup to
        # "cassette" and missed everything it had just written.
        self.model = model or getattr(inner, "model", None) or "cassette"
        self.hits = 0
        self.misses = 0
        self.recorded = 0

    def _path(self, key: str) -> pathlib.Path:
        return self.directory / f"{key}.json"

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> LLMResponse:
        key = prompt_key(system, user, self.model)
        path = self._path(key)

        if self.mode != "refresh" and path.exists():
            self.hits += 1
            stored = json.loads(path.read_text())
            usage = stored.get("usage") or {}
            return LLMResponse(
                data=stored["data"],
                model=stored.get("model", self.model),
                usage=Usage(
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                    requests=0,  # replays cost nothing
                ),
                stop_reason=stored.get("stop_reason"),
            )

        self.misses += 1
        if self.mode == "replay":
            raise ProviderResponseError(
                f"no recorded response for this prompt ({key}). Re-run with "
                "--record to capture it, or check whether the prompt changed."
            )
        if self.inner is None:
            raise ProviderResponseError("cassette has no provider to record from")

        response = self.inner.complete_structured(system, user, schema, max_tokens)
        self.recorded += 1
        path.write_text(
            json.dumps(
                {
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "data": response.data,
                    # Stored for humans reading a diff, never read back.
                    "_prompt_preview": user[:400],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return response

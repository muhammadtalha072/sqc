"""Shared HTTP behaviour for real providers.

Retry policy lives here rather than in each client so that a Voyage 429 and
an Anthropic 529 produce the same observable behaviour upstream. Anything
that reaches the pipeline as a ProviderError has already exhausted retries.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any

import httpx

from sqc.providers.base import (
    ProviderAuthError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderResponseError,
)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

# A 429 can mean two different things and only one of them is worth waiting
# for. Rate limiting clears in seconds; an exhausted daily quota does not
# clear until it resets, so retrying it burns the remaining allowance on
# calls that cannot succeed. An eight-attempt budget against a 250-request
# daily free tier consumed most of a day's quota in one evaluation run.
_QUOTA_EXHAUSTED = re.compile(
    r"RESOURCE_EXHAUSTED|quota (?:exceeded|exhausted)|daily limit|"
    r"insufficient[_ ]quota|billing",
    re.I,
)

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BASE_DELAY = 0.5
DEFAULT_TIMEOUT = 60.0


class HttpProviderClient:
    """Thin wrapper adding auth, retries and error mapping to httpx."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        base_delay: float = DEFAULT_BASE_DELAY,
    ) -> None:
        self.max_attempts = max(1, max_attempts)
        self.base_delay = base_delay
        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                response = self._client.post(path, json=payload)
            except httpx.TimeoutException as exc:
                last_error = exc
                self._sleep(attempt, None)
                continue
            except httpx.HTTPError as exc:
                raise ProviderResponseError(f"request to {path} failed: {exc}") from exc

            if response.status_code in (401, 403):
                raise ProviderAuthError(
                    f"{path} rejected the credentials ({response.status_code}); "
                    "check the API key in .env"
                )
            if response.status_code in RETRY_STATUS:
                body = response.text[:400]
                if response.status_code == 429 and _QUOTA_EXHAUSTED.search(body):
                    raise ProviderQuotaError(
                        f"{path} quota exhausted: {body.strip()[:250]}"
                    )
                last_error = ProviderRateLimitError(
                    f"{path} returned {response.status_code}: {body[:200]}"
                )
                if attempt == self.max_attempts - 1:
                    break
                self._sleep(attempt, response.headers.get("retry-after"))
                continue
            if response.status_code >= 400:
                raise ProviderResponseError(
                    f"{path} returned {response.status_code}: {response.text[:300]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise ProviderResponseError(f"{path} returned invalid JSON") from exc

        raise ProviderRateLimitError(
            f"{path} failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error

    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        """Honour Retry-After when the server sends it, else exponential
        backoff with jitter so parallel workers do not retry in lockstep."""
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 30.0))
                return
            except ValueError:
                pass
        if self.base_delay <= 0:
            return
        delay = self.base_delay * (2**attempt)
        time.sleep(min(delay + random.uniform(0, self.base_delay), 30.0))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpProviderClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

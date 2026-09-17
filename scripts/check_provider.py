"""Check that a configured provider key works, before running anything real.

    python scripts/check_provider.py

Probes the endpoint the pipeline actually uses rather than a listing
endpoint, because those can accept different credential types. Google moved
AI Studio to authorization keys (the AQ. prefix) and the older listing route
rejects them, which produced a 400 that said nothing useful.

Every failure prints the provider's own error body. That message is almost
always more precise than anything this script could infer from a status code.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from sqc.config import get_settings  # noqa: E402

GEMINI_BASE = "https://generativelanguage.googleapis.com"

# Tried in order. Free-tier names change often enough that hard-coding one
# and letting it 404 mid-run is the most likely first failure.
GEMINI_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]


def _body(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:400]
    error = payload.get("error", payload)
    if isinstance(error, dict):
        return f"{error.get('status', '')} {error.get('message', '')}".strip() or str(error)[:400]
    return str(error)[:400]


def _post(url: str, key: str, payload: dict) -> tuple[bool, str, httpx.Response | None]:
    try:
        response = httpx.post(
            url,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=45.0,
        )
    except httpx.HTTPError as exc:
        return False, f"network error: {exc}", None
    if response.status_code == 200:
        return True, "", response
    return False, f"HTTP {response.status_code}: {_body(response)}", response


def probe_gemini_model(key: str, model: str) -> tuple[bool, str]:
    """One minimal call. Tries generateContent, then the Interactions API.

    Google is migrating REST callers to /v1beta/interactions, and the two
    routes do not necessarily accept the same credential types. Trying both
    turns 'API key not valid' into a statement about which route works.
    """
    ok, detail, response = _post(
        f"{GEMINI_BASE}/v1beta/models/{model}:generateContent",
        key,
        {
            "contents": [{"role": "user", "parts": [{"text": "Reply with the word OK."}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 16},
        },
    )
    if ok and response is not None:
        payload = response.json()
        parts = (payload.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = " ".join(p.get("text", "") for p in parts).strip()
        return True, f"generateContent replied {text!r}" if text else "generateContent replied"

    ok2, detail2, response2 = _post(
        f"{GEMINI_BASE}/v1beta/interactions",
        key,
        {"model": model, "input": "Reply with the word OK."},
    )
    if ok2 and response2 is not None:
        body = response2.json()
        text = str(body.get("output_text") or body.get("outputText") or "").strip()
        return True, f"interactions replied {text!r} (USE INTERACTIONS)"
    return False, f"generateContent: {detail} | interactions: {detail2}"


def check_gemini(key: str, configured_model: str) -> int:
    prefix = "auth key (AQ.)" if key.startswith("AQ.") else (
        "standard key (AIza)" if key.startswith("AIza") else "unrecognised prefix"
    )
    print(f"key format: {prefix}")
    if key.startswith("AIza"):
        print("  note: standard keys are being retired; AI Studio now issues auth keys.")
    print()

    candidates = [configured_model] if configured_model else []
    candidates += [m for m in GEMINI_CANDIDATES if m != configured_model]

    working: list[str] = []
    for model in candidates:
        ok, detail = probe_gemini_model(key, model)
        print(f"  {'OK  ' if ok else 'FAIL'} {model:24} {detail}")
        if ok:
            working.append(model)
            if len(working) >= 2:
                break

    print()
    if working:
        print(f"key works. Set this in .env:\n\n  SQC_LLM_MODEL={working[0]}\n")
        return 0
    print("no model responded. The errors above are Google's own messages;\n"
          "429 means the free quota is exhausted, 403 means the key is rejected,\n"
          "404 means the model name does not exist for this key.")
    return 1


def check_anthropic(key: str) -> int:
    response = httpx.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        timeout=30.0,
    )
    if response.status_code != 200:
        print(f"HTTP {response.status_code}: {_body(response)}")
        return 1
    print("key works. Available models:\n")
    for model in response.json().get("data", []):
        print(f"  {model.get('id')}")
    return 0


def main() -> int:
    settings = get_settings()
    provider = settings.llm_provider.lower()
    print(f"SQC_LLM_PROVIDER = {provider}\n")

    if provider == "gemini":
        if not settings.gemini_api_key:
            print("GEMINI_API_KEY is not set in .env. Get a free key at aistudio.google.com")
            return 1
        model = settings.llm_model if "gemini" in settings.llm_model.lower() else ""
        return check_gemini(settings.gemini_api_key, model)

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            print("ANTHROPIC_API_KEY is not set in .env. Get one at console.anthropic.com")
            return 1
        return check_anthropic(settings.anthropic_api_key)

    if provider in ("manual", "fake"):
        print(f"'{provider}' needs no key and no network.")
        return 0

    print(f"nothing to check for provider '{provider}'")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

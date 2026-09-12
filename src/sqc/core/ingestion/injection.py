"""Ingestion-time prompt-injection scanner.

This is a detection layer, not the defence. The defence is structural: the
answering model receives evidence inside a delimited data block, and the
validator refuses any claim that no retrieved chunk entails, so an injected
"answer YES" produces an unsupported claim and is rejected on the way out.

What this adds is that flagged chunks are excluded from evidence packs
before the model ever sees them, and the flag is visible to the customer.
Suspect chunks are still stored: they are the customer's own documents, and
silently dropping content would be worse than flagging it.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that render as nothing but survive copy-paste, the usual way
# instructions get smuggled into an otherwise innocent-looking paragraph.
_HIDDEN_CHARS = re.compile(
    "[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u206a-\u206f\ufeff\u00ad]"
)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}?\b"
            r"(?:previous|prior|above|earlier|preceding|all)\b[^.\n]{0,30}?\b"
            r"(?:instruction|prompt|direction|rule|command)s?\b",
            re.I,
        ),
    ),
    (
        "role_override",
        re.compile(
            # No trailing \b: the "new instructions:" alternative ends in a
            # colon followed by a space, and a word boundary between two
            # non-word characters never matches.
            r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you|act\s+as\s+(?:an?\s+)?\w+|"
            r"pretend\s+to\s+be|new\s+instructions?\s*:|your\s+new\s+(?:role|task)\s+is)",
            re.I,
        ),
    ),
    (
        "system_prompt_reference",
        re.compile(r"(?:system\s+prompt|</?(?:system|instructions?|evidence|context)\s*>)", re.I),
    ),
    (
        "forced_answer",
        re.compile(
            r"\b(?:always\s+)?(?:answer|respond|reply|state)\b[^.\n]{0,30}?\b"
            r"(?:yes|no|compliant|true|affirmative)\b"
            r"|\bmark\s+(?:this|it|us|the\s+\w+)\s+as\s+(?:compliant|yes|passed|satisfied)\b",
            re.I,
        ),
    ),
    (
        "chat_role_marker",
        re.compile(r"^\s*(?:human|assistant|system|user)\s*:", re.I | re.M),
    ),
)


def strip_hidden_characters(text: str) -> str:
    """Remove zero-width and bidi-control characters, then normalise.

    Applied to text before it is stored so that retrieval, display and the
    model all see the same string. The flag is raised separately by
    `scan_for_injection`, which runs on the original.
    """
    return unicodedata.normalize("NFKC", _HIDDEN_CHARS.sub("", text))


def scan_for_injection(text: str) -> tuple[str, ...]:
    """Return sorted flag names for injection-like content. Empty when clean."""
    flags = {name for name, pattern in _PATTERNS if pattern.search(text)}
    if _HIDDEN_CHARS.search(text):
        flags.add("hidden_characters")
    return tuple(sorted(flags))

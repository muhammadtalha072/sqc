"""Heading recognition and document metadata extraction.

Two jobs, both governed by the same rule: when the evidence is ambiguous,
return nothing. A wrong effective date silently reorders which policy wins
a contradiction check, which is worse than having no date at all.
"""

from __future__ import annotations

import re
from datetime import date

# ------------------------------------------------------------------ headings

# "4", "4.2", "4.2.1" optionally followed by a dot, then the title.
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,5})\.?[\s\u00a0]+(\S.*)$")
# "Appendix B", "Section 4", "Annex I", "Article 12"
_LABELLED = re.compile(r"^(appendix|annex|section|article|schedule)\s+([a-z0-9]{1,4})\b", re.I)
# Trailing sentence punctuation is the strongest signal that a line is prose.
_SENTENCE_END = re.compile(r"[.;:,!?]\s*$")

MAX_HEADING_WORDS = 14
MAX_HEADING_CHARS = 120


def numbering_level(text: str) -> int | None:
    """Depth implied by a section number, e.g. '4.2.1 Access Control' -> 3.

    Returns None when the line carries no usable numbering. This signal is
    trusted above font size because security policies number their sections
    consistently while their typography is often a mess.
    """
    stripped = text.strip()
    match = _NUMBERED.match(stripped)
    if match:
        number, title = match.groups()
        # "2024. Something" is a year, not a section number.
        if re.fullmatch(r"(19|20)\d{2}", number):
            return None
        if not title.strip() or len(title) > MAX_HEADING_CHARS:
            return None
        return min(len(number.split(".")), 6)
    if _LABELLED.match(stripped) and len(stripped) <= MAX_HEADING_CHARS:
        return 1
    return None


def looks_like_heading(text: str) -> bool:
    """Shape-based fallback for documents with no numbering and no styles.

    Deliberately conservative: a false heading fragments a section and
    strands the qualifier that belonged with the claim.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if len(stripped.split()) > MAX_HEADING_WORDS:
        return False
    if _SENTENCE_END.search(stripped):
        return False
    if numbering_level(stripped) is not None:
        return True
    letters = [c for c in stripped if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True
    # Title Case with no trailing punctuation, e.g. "Encryption at Rest".
    words = [w for w in re.split(r"\s+", stripped) if w]
    if len(words) >= 2:
        significant = [w for w in words if len(w) > 3]
        if significant and all(w[0].isupper() for w in significant):
            return True
    return False


def levels_from_font_sizes(sizes: list[float], body_size: float) -> dict[float, int]:
    """Map distinct font sizes above the body size onto heading levels 1..6.

    Largest size becomes level 1. Sizes at or below the body size are not
    headings, so they are absent from the mapping.
    """
    distinct = sorted({round(s, 1) for s in sizes if round(s, 1) > round(body_size, 1)}, reverse=True)
    return {size: min(i + 1, 6) for i, size in enumerate(distinct)}


# ------------------------------------------------------------------ metadata

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"], start=1)
}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})

_DATE_LABEL = (
    r"(?:effective|effective\s+date|last\s+updated|last\s+revised|revised|"
    r"revision\s+date|date\s+of\s+issue|issued|issue\s+date|approved(?:\s+on)?|"
    r"published|last\s+reviewed)"
)
# "12 March 2024" / "12th March, 2024"
_DMY = r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+((?:19|20)\d{2})"
# "March 12, 2024"
_MDY = r"([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})"
# "2024-03-12"
_ISO = r"((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})"
# "12/03/2024" - resolved only when unambiguous
_NUMERIC = r"(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})"

_DATE_PATTERNS = [
    (re.compile(rf"{_DATE_LABEL}\s*[:\-\u2013]?\s*{_DMY}", re.I), "dmy"),
    (re.compile(rf"{_DATE_LABEL}\s*[:\-\u2013]?\s*{_MDY}", re.I), "mdy"),
    (re.compile(rf"{_DATE_LABEL}\s*[:\-\u2013]?\s*{_ISO}", re.I), "iso"),
    (re.compile(rf"{_DATE_LABEL}\s*[:\-\u2013]?\s*{_NUMERIC}", re.I), "numeric"),
]

_VERSION = re.compile(
    r"\b(?:version|revision|rev\.?|v)\s*[:\-]?\s*(\d+(?:\.\d+){0,2}[a-z]?)\b", re.I
)

METADATA_SCAN_CHARS = 4000
"""Only the front matter is searched. A date deep inside a policy is
usually an example or an audit period, not the document's own date."""


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_effective_date(text: str) -> tuple[date | None, list[str]]:
    """Pull a labelled effective date from the front matter.

    Returns (date, warnings). An unlabelled date is ignored entirely, and an
    ambiguous numeric date like 03/04/2024 is refused with a warning rather
    than resolved by assuming a locale.
    """
    head = text[:METADATA_SCAN_CHARS]
    warnings: list[str] = []
    for pattern, style in _DATE_PATTERNS:
        match = pattern.search(head)
        if not match:
            continue
        groups = match.groups()
        if style == "dmy":
            day, month_name, year = groups
            month = _MONTHS.get(month_name.lower().rstrip("."))
            if month is None:
                continue
            found = _safe_date(int(year), month, int(day))
        elif style == "mdy":
            month_name, day, year = groups
            month = _MONTHS.get(month_name.lower().rstrip("."))
            if month is None:
                continue
            found = _safe_date(int(year), month, int(day))
        elif style == "iso":
            year, month, day = groups
            found = _safe_date(int(year), int(month), int(day))
        else:
            first, second, year = int(groups[0]), int(groups[1]), int(groups[2])
            if first <= 12 and second <= 12 and first != second:
                warnings.append(
                    f"ambiguous numeric date '{match.group(0).strip()}' "
                    "could be day/month or month/day - effective date left unset"
                )
                continue
            # Unambiguous: whichever value exceeds 12 must be the day.
            found = _safe_date(year, second, first) if first > 12 else _safe_date(year, first, second)
        if found is not None:
            return found, warnings
    return None, warnings


def extract_version_label(text: str) -> str | None:
    match = _VERSION.search(text[:METADATA_SCAN_CHARS])
    return match.group(1) if match else None

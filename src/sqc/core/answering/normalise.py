"""Text normalisation shared by the literal check and the entailment checker.

Lives in its own module because both need it and neither should import
the other. Duplicating it was the actual bug: the date fix landed in the
literal check while the entailment checker kept flagging the same
correct answer, so the false positive survived a fix that appeared to
work.
"""

from __future__ import annotations

import re

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTHS.update({name[:3]: number for name, number in list(_MONTHS.items())})

_DATE_PATTERNS = (
    re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b"),                 # 2017-03-31
    re.compile(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b"),                 # 3/31/2017
    re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+((?:19|20)\d{2})\b"),  # 31 March 2017
    re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b"),  # March 31, 2017
)


def canonical_dates(text: str) -> str:
    """Rewrite every recognisable date to one canonical token.

    A policy dated 3/31/2017 answered as 2017-03-31 is the same date in a
    different notation, and comparing the raw strings reported the correct
    answer as a fabricated figure. Dates are the most common claim in a
    security questionnaire, so a false positive here is expensive.

    Ambiguous day/month pairs are left alone rather than guessed: resolving
    03/04/2017 by assuming a locale is how a wrong date becomes a confident
    one.
    """
    def iso(year: int, month: int, day: int) -> str:
        return f" date{year:04d}{month:02d}{day:02d} "

    def repl_ymd(m: re.Match[str]) -> str:
        return iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def repl_numeric(m: re.Match[str]) -> str:
        first, second, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if first <= 12 and second <= 12 and first != second:
            return m.group(0)  # ambiguous; do not guess
        return iso(year, first, second) if first <= 12 else iso(year, second, first)

    def repl_dmy(m: re.Match[str]) -> str:
        month = _MONTHS.get(m.group(2).lower().rstrip("."))
        return iso(int(m.group(3)), month, int(m.group(1))) if month else m.group(0)

    def repl_mdy(m: re.Match[str]) -> str:
        month = _MONTHS.get(m.group(1).lower().rstrip("."))
        return iso(int(m.group(3)), month, int(m.group(2))) if month else m.group(0)

    for pattern, repl in zip(
        _DATE_PATTERNS, (repl_ymd, repl_numeric, repl_dmy, repl_mdy), strict=True
    ):
        text = pattern.sub(repl, text)
    return text



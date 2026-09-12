"""Validation of model output against the evidence that was actually retrieved.

This is where status is decided. The model proposes; this module disposes.
Nothing here asks the model how confident it is, and nothing here trusts a
field the model could have fabricated.

Checks, in order of how mechanically certain they are:

  1. shape          required fields present and of the right type
  2. citations      every handle resolves to evidence that was retrieved
  3. claims         every claim carries at least one resolving handle
  4. literals       figures and standards in the answer appear in cited text
  5. conflict       disagreement reported or detectable across sources
  6. scope          a partial answer never presents as a settled yes or no
"""

from __future__ import annotations

import re
from typing import Any

from sqc.core.answering.schema import (
    AnswerStatus,
    AnswerType,
    Citation,
    Claim,
)
from sqc.core.retrieval.types import Evidence

# Tokens that make a claim checkable: anything carrying a digit, plus the
# standards questionnaires care about. A fabricated "AES-256" or "90 days"
# is the most damaging kind of hallucination here and the easiest to catch.
_LITERAL = re.compile(
    r"\b(?:"
    r"[A-Za-z]*\d[\w.\-]*"          # 90, AES-256, TLS1.2, 24x7, v2.1
    r"|SOC\s?[12]|ISO\s?\d{4,5}"     # SOC 2, ISO 27001
    r"|PCI[\s-]?DSS|HIPAA|GDPR|FedRAMP"
    r")\b",
    re.I,
)
_SPELLED_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "twelve": "12", "twenty": "20", "thirty": "30", "sixty": "60", "ninety": "90",
    "hundred": "100", "annually": "1", "yearly": "1", "quarterly": "4", "monthly": "12",
}


def _normalise(text: str) -> str:
    lowered = text.lower()
    for word, digit in _SPELLED_NUMBERS.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    return re.sub(r"[\s\-]+", "", lowered)


def find_unsupported_literals(answer: str, cited_text: str) -> list[str]:
    """Figures and standards in the answer that appear in no cited evidence.

    Deliberately literal. A model that writes "AES-256" when the policy says
    "AES-128", or "90 days" when the policy says "30 days", produces an
    answer that reads perfectly and is wrong in the one way a customer will
    be held to. Spelled-out numbers are normalised so "ninety days" in the
    source supports "90 days" in the answer.
    """
    haystack = _normalise(cited_text)
    missing: list[str] = []
    for match in _LITERAL.findall(answer):
        token = match.strip()
        if _normalise(token) and _normalise(token) not in haystack:
            if token not in missing:
                missing.append(token)
    return missing


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def validate(
    raw: dict[str, Any],
    evidence: tuple[Evidence, ...] | list[Evidence],
) -> tuple[AnswerStatus, str, AnswerType | None, list[Claim], list[str], str]:
    """Validate model output.

    Returns (status, answer, answer_type, claims, validation_errors, reason).
    """
    errors: list[str] = []
    by_handle = {item.evidence_id: item for item in evidence}

    # --- 1. shape -------------------------------------------------------
    if not isinstance(raw, dict):
        return (
            AnswerStatus.REFUSED, "", None, [],
            ["model output was not an object"],
            "the answering model returned output that could not be read",
        )

    answer = _as_text(raw.get("answer"))
    reason = _as_text(raw.get("reason"))
    raw_claims = raw.get("claims")
    if not isinstance(raw_claims, list):
        errors.append("claims missing or not a list")
        raw_claims = []

    answer_type: AnswerType | None = None
    try:
        answer_type = AnswerType(str(raw.get("answer_type", "")).lower())
    except ValueError:
        errors.append(f"unrecognised answer_type {raw.get('answer_type')!r}")

    # A model-set confidence is ignored rather than trusted. Its presence is
    # recorded because it means the prompt or schema has drifted.
    if any(key in raw for key in ("confidence", "confidence_score", "certainty")):
        errors.append("model returned a confidence field; ignored")

    # --- 2 & 3. citations and claims ------------------------------------
    claims: list[Claim] = []
    for entry in raw_claims:
        if not isinstance(entry, dict):
            errors.append("claim was not an object")
            continue
        text = _as_text(entry.get("text"))
        if not text:
            errors.append("claim had no text")
            continue

        handles = entry.get("evidence_ids")
        handles = [h for h in handles if isinstance(h, str)] if isinstance(handles, list) else []
        resolved: list[Citation] = []
        unknown: list[str] = []
        for handle in handles:
            item = by_handle.get(handle.strip())
            if item is None:
                unknown.append(handle)
                continue
            resolved.append(
                Citation(
                    evidence_id=item.evidence_id,
                    chunk_ids=item.chunk_ids,
                    document_id=item.document_id,
                    filename=item.filename,
                    heading_path=item.heading_path,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    effective_date=item.effective_date,
                )
            )

        if unknown:
            errors.append(
                f"claim cited unknown evidence handle(s) {', '.join(sorted(set(unknown)))}"
            )
        if not resolved:
            claims.append(
                Claim(
                    text=text,
                    supported=False,
                    problem=(
                        "cited evidence that was not retrieved"
                        if unknown
                        else "no supporting evidence cited"
                    ),
                )
            )
            continue
        claims.append(Claim(text=text, citations=tuple(resolved), supported=True))

    supported_claims = [c for c in claims if c.supported]

    # --- refusals -------------------------------------------------------
    if not evidence:
        return (
            AnswerStatus.REFUSED, "", answer_type, claims, errors,
            "no evidence was retrieved for this question",
        )
    if raw.get("evidence_sufficient") is False or answer_type is AnswerType.NOT_FOUND:
        return (
            AnswerStatus.REFUSED, "", answer_type, claims, errors,
            reason or "the retrieved evidence does not answer this question",
        )
    if not supported_claims:
        return (
            AnswerStatus.REFUSED, "", answer_type, claims, errors,
            "no claim in the generated answer was supported by retrieved evidence",
        )
    if not answer:
        return (
            AnswerStatus.REFUSED, "", answer_type, claims, errors,
            "the answering model produced claims but no answer text",
        )

    # --- review triggers ------------------------------------------------
    review: list[str] = []

    if len(supported_claims) != len(claims):
        review.append("some claims were dropped because their citations did not resolve")

    cited_text = " ".join(
        by_handle[c.evidence_id].text
        for claim in supported_claims
        for c in claim.citations
        if c.evidence_id in by_handle
    )
    unsupported_literals = find_unsupported_literals(answer, cited_text)
    if unsupported_literals:
        errors.append(
            "answer contains figures absent from cited evidence: "
            + ", ".join(unsupported_literals)
        )
        review.append(
            "the answer states "
            + ", ".join(unsupported_literals)
            + " which does not appear in the cited evidence"
        )

    if raw.get("conflict_detected") is True:
        note = _as_text(raw.get("conflict_note"))
        review.append(f"evidence conflicts: {note}" if note else "the evidence conflicts")

    conflict = _detect_source_conflict(supported_claims)
    if conflict:
        review.append(conflict)

    if answer_type is AnswerType.PARTIAL:
        review.append("the evidence covers the question only partially")

    if review:
        return (
            AnswerStatus.REVIEW_REQUIRED, answer, answer_type, claims, errors,
            "; ".join(review),
        )
    return (
        AnswerStatus.SUPPORTED, answer, answer_type, claims, errors,
        reason or "every claim is supported by the cited evidence",
    )


def _detect_source_conflict(claims: list[Claim]) -> str | None:
    """Flag evidence drawn from different versions of the same document.

    Deliberately narrow. Detecting that two passages contradict each other
    in meaning needs a model, which belongs in the later semantic-validation
    stage. What is decidable here is provenance: if supporting evidence
    carries two different effective dates, the answer may be resting on a
    superseded policy, and a human should look.
    """
    dates = {
        c.effective_date
        for claim in claims
        for c in claim.citations
        if c.effective_date is not None
    }
    if len(dates) > 1:
        ordered = ", ".join(sorted(str(d) for d in dates))
        return (
            f"cited evidence carries different effective dates ({ordered}); "
            "one source may be superseded"
        )
    return None

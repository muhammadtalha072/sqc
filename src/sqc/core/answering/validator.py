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

from dataclasses import dataclass

from sqc.core.answering.entailment import Entailment, EntailmentProvider
from sqc.core.answering.normalise import canonical_dates
from sqc.core.answering.schema import (
    AnswerStatus,
    AnswerType,
    Citation,
    Claim,
)
from sqc.core.retrieval.types import Evidence


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """What the validator decided and why.

    failure_stage separates the three ways an answer can fail to land:
    retrieval never found the evidence, the answering model misused evidence
    it had, or a validation check rejected the result. Without that split a
    refusal rate says the system is cautious but not where to fix it.
    """

    status: AnswerStatus
    answer: str
    answer_type: AnswerType | None
    claims: list[Claim]
    errors: list[str]
    reason: str
    signals: dict[str, Any]
    failure_stage: str | None = None

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
# Evidence handles look like literals to a regex hunting for digit-bearing
# tokens, so an answer written "as stated in E1, E2" was reported as five
# fabricated figures. A citation marker is not a claimed fact.
_EVIDENCE_HANDLE = re.compile(r"^E\d+$", re.I)

_SPELLED_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "twelve": "12", "twenty": "20", "thirty": "30", "sixty": "60", "ninety": "90",
    "hundred": "100", "annually": "1", "yearly": "1", "quarterly": "4", "monthly": "12",
}


def _normalise(text: str) -> str:
    lowered = canonical_dates(text).lower()
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
    for match in _LITERAL.findall(canonical_dates(answer)):
        token = match.strip()
        if _EVIDENCE_HANDLE.match(token):
            continue
        # Canonical date tokens exist for comparison, not for humans. A
        # reviewer reading "date20170304" learns nothing.
        if token.lower().startswith("date") and len(token) == 12 and token[4:].isdigit():
            token = f"{token[4:8]}-{token[8:10]}-{token[10:12]}"
        if _normalise(token) and _normalise(token) not in haystack:
            if token not in missing:
                missing.append(token)
    return missing


def _as_text(value: Any) -> str:
    """Read a string field, repairing escape sequences the model emitted
    literally.

    Models occasionally write the two characters backslash-n inside a JSON
    string rather than a real newline. Harmless in a terminal, visible as
    stray backslashes the moment an answer is pasted into a questionnaire
    cell, which is where these answers are going.
    """
    if not isinstance(value, str):
        return ""
    return (
        value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ").strip()
    )


def validate(
    raw: dict[str, Any],
    evidence: tuple[Evidence, ...] | list[Evidence],
    entailment: EntailmentProvider | None = None,
    question: str = "",
) -> ValidationOutcome:
    """Validate model output against the evidence that was retrieved."""
    errors: list[str] = []
    signals: dict[str, Any] = {
        "entailment_provider": entailment.name if entailment else None,
        "claims_entailed": 0,
        "claims_unentailed": 0,
        "claims_contradicted": 0,
        "exception_omitted": False,
        "scope_mismatch": False,
        "stale_citation": False,
        "conflicting_answer_surfaced": False,
        "date_conflict": False,
        "unsupported_literals": [],
    }
    by_handle = {item.evidence_id: item for item in evidence}

    # --- 1. shape -------------------------------------------------------
    if not isinstance(raw, dict):
        return ValidationOutcome(
            AnswerStatus.REFUSED, "", None, [],
            ["model output was not an object"],
            "the answering model returned output that could not be read",
            signals, "answering",
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
        return ValidationOutcome(
            AnswerStatus.REFUSED, "", answer_type, claims, errors,
            "no evidence was retrieved for this question", signals, "retrieval",
        )
    if raw.get("evidence_sufficient") is False or answer_type is AnswerType.NOT_FOUND:
        # Conflicting evidence is not insufficient evidence. One means no
        # answer exists; the other means two answers exist and a human must
        # choose. Collapsing them threw away the most useful output the
        # system produces: a fully cited conflict.
        #
        # Measured case: asked for the minimum password length, the model
        # retrieved both policy versions, produced two correctly cited claims
        # (12 characters from 2021, 16 from 2024), identified the conflict -
        # and the validator discarded all of it and returned a bare refusal.
        # "I cannot determine this" is worth far less to a reviewer than
        # "your 2021 policy says 12, your 2024 says 16, confirm which applies".
        conflict = _cited_conflict(supported_claims, bool(raw.get("conflict_detected")))
        if conflict:
            signals["conflicting_answer_surfaced"] = True
            surfaced = answer or " ".join(c.text for c in supported_claims)
            return ValidationOutcome(
                AnswerStatus.REVIEW_REQUIRED, surfaced, answer_type, claims, errors,
                f"{conflict}; a reviewer must confirm which applies", signals, "validation",
            )
        return ValidationOutcome(
            AnswerStatus.REFUSED, "", answer_type, claims, errors,
            reason or "the retrieved evidence does not answer this question",
            signals, "retrieval",
        )
    if not supported_claims:
        # The distinction matters for diagnosis. No claims at all means the
        # answering model produced nothing to check. Claims that all got
        # dropped means it produced something and validation rejected it.
        made_claims = bool(claims)
        return ValidationOutcome(
            AnswerStatus.REFUSED, "", answer_type, claims, errors,
            (
                "no claim in the generated answer was supported by retrieved evidence"
                if made_claims
                else "the answering model produced no claims to check"
            ),
            signals, "validation" if made_claims else "answering",
        )
    if not answer:
        return ValidationOutcome(
            AnswerStatus.REFUSED, "", answer_type, claims, errors,
            "the answering model produced claims but no answer text",
            signals, "answering",
        )

    # --- entailment: a citation alone does not confer support -----------
    if entailment is not None:
        judged: list[Claim] = []
        for claim in claims:
            if not claim.supported:
                judged.append(claim)
                continue
            cited = " ".join(
                by_handle[c.evidence_id].text
                for c in claim.citations
                if c.evidence_id in by_handle
            )
            verdict = entailment.check(claim.text, cited)
            judged.append(
                Claim(
                    text=claim.text,
                    citations=claim.citations,
                    # The verdict is probabilistic, so it never deletes a
                    # claim. It downgrades the answer to human review and
                    # records why.
                    supported=claim.supported,
                    problem=claim.problem,
                    entailment=verdict.label.value,
                    entailment_score=verdict.score,
                    entailment_detail=verdict.detail,
                )
            )
            if verdict.label is Entailment.SUPPORTED:
                signals["claims_entailed"] += 1
            elif verdict.label is Entailment.CONTRADICTED:
                signals["claims_contradicted"] += 1
            elif verdict.label is Entailment.UNSUPPORTED:
                signals["claims_unentailed"] += 1
        claims = judged
        supported_claims = [c for c in claims if c.supported]

    # --- review triggers ------------------------------------------------
    review: list[str] = []

    if len(supported_claims) != len(claims):
        review.append("some claims were dropped because their citations did not resolve")

    if signals["claims_contradicted"]:
        review.append(
            f"{signals['claims_contradicted']} claim(s) contradicted by the cited evidence"
        )
    if signals["claims_unentailed"]:
        review.append(
            f"{signals['claims_unentailed']} claim(s) not supported by the evidence cited "
            "for them"
        )

    cited_text = " ".join(
        by_handle[c.evidence_id].text
        for claim in supported_claims
        for c in claim.citations
        if c.evidence_id in by_handle
    )

    omitted = find_omitted_exception(answer, cited_text)
    if omitted:
        signals["exception_omitted"] = True
        review.append(omitted)

    mismatch = find_scope_mismatch(question, answer, cited_text)
    if mismatch:
        signals["scope_mismatch"] = True
        review.append(mismatch)

    stale = find_stale_citation(supported_claims, list(evidence))
    if stale:
        signals["stale_citation"] = True
        review.append(stale)

    unsupported_literals = find_unsupported_literals(answer, cited_text)
    if unsupported_literals:
        signals["unsupported_literals"] = unsupported_literals
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
        signals["date_conflict"] = True
        review.append(conflict)

    if answer_type is AnswerType.PARTIAL:
        review.append("the evidence covers the question only partially")

    if review:
        return ValidationOutcome(
            AnswerStatus.REVIEW_REQUIRED, answer, answer_type, claims, errors,
            "; ".join(review), signals, "validation",
        )
    return ValidationOutcome(
        AnswerStatus.SUPPORTED, answer, answer_type, claims, errors,
        reason or "every claim is supported by the cited evidence", signals, None,
    )



# ---------------------------------------------------------------- new checks

_EXCEPTION_MARKERS = re.compile(
    r"\b(?:except(?:ion|ions|ed)?\b|exempt(?:ion|ions|ed)?\b|unless\b|"
    r"does not apply\b|do not apply\b|other than\b|save for\b|carve[- ]out\b)",
    re.I,
)
_UNIVERSAL = re.compile(r"\b(?:all|every|any|each|always|entire|whole)\b", re.I)
_SCOPE_LIMITS = re.compile(
    r"\b(?:only|solely|limited to|administrators?|privileged|service accounts?|"
    r"break[- ]glass|certain|some|specific|designated|where (?:applicable|feasible)|"
    r"as (?:applicable|appropriate))\b",
    re.I,
)


def find_omitted_exception(answer: str, cited_text: str) -> str | None:
    """Cited evidence carries an exception the answer does not mention.

    The canonical dangerous answer in this product: the evidence says MFA is
    required except for break-glass accounts, and the answer says MFA is
    required. Every word of it is in the source and it is still misleading,
    which is why omission is checked rather than only fabrication.
    """
    if not _EXCEPTION_MARKERS.search(cited_text):
        return None
    if _EXCEPTION_MARKERS.search(answer):
        return None
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", cited_text) if _EXCEPTION_MARKERS.search(s)
    ]
    excerpt = sentences[0][:160] if sentences else "an exception"
    return f"cited evidence contains an exception the answer omits: {excerpt}"


def find_scope_mismatch(question: str, answer: str, cited_text: str) -> str | None:
    """The question asks universally; the evidence answers narrowly.

    'Do you require MFA for all employees?' answered from a control that
    applies to administrators is not a yes. Detected structurally: a
    universal quantifier in the question, a scope limiter in the evidence,
    and an answer that acknowledges neither.
    """
    if not _UNIVERSAL.search(question):
        return None
    limiter = _SCOPE_LIMITS.search(cited_text)
    if not limiter:
        return None
    if _SCOPE_LIMITS.search(answer) or _UNIVERSAL.search(answer):
        return None
    return (
        f"question asks universally but the cited evidence is limited "
        f"('{limiter.group(0)}') and the answer does not say so"
    )


def find_stale_citation(claims: list[Claim], evidence: list[Evidence]) -> str | None:
    """An answer resting on a superseded document.

    If newer evidence was retrieved for the same source and the answer cites
    only the older one, the customer may be certifying last year's policy.
    """
    dated = [item for item in evidence if item.effective_date is not None]
    if len(dated) < 2:
        return None
    newest = max(item.effective_date for item in dated)
    cited_dates = {
        c.effective_date for claim in claims for c in claim.citations if c.effective_date
    }
    if not cited_dates or newest in cited_dates:
        return None
    return (
        f"every citation comes from evidence effective {max(cited_dates)}, but newer "
        f"evidence effective {newest} was retrieved and not cited"
    )

def _cited_conflict(claims: list[Claim], model_flagged: bool) -> str | None:
    """Do the supported claims disagree across sources?

    Requires two or more claims drawn from different documents, so a single
    claim citing several sections - normal agreement - is not mistaken for
    disagreement. The model's own conflict flag strengthens the signal but
    does not create it, since the evidence is what has to disagree.
    """
    if len(claims) < 2:
        return None
    documents = {c.document_id for claim in claims for c in claim.citations}
    dates = {
        c.effective_date
        for claim in claims
        for c in claim.citations
        if c.effective_date is not None
    }
    if len(documents) < 2 and not (model_flagged and len(dates) > 1):
        return None

    files = sorted({c.filename for claim in claims for c in claim.citations})
    detail = f"evidence from {' and '.join(files)} disagrees"
    if len(dates) > 1:
        detail += f" (effective {', '.join(sorted(str(d) for d in dates))})"
    return detail


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

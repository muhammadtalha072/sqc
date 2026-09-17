"""Tests for claim-support checking and the validator checks added in Step 7.

Each check exists because a specific wrong answer was reachable without it.
The tests state that wrong answer explicitly, so a future change that
reintroduces it fails here rather than in front of a customer.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest

from sqc.core.answering.entailment import (
    Entailment,
    EntailmentProvider,
    EntailmentResult,
    LexicalEntailment,
    LLMEntailment,
)
from sqc.core.answering.schema import AnswerStatus, Citation, Claim
from sqc.core.answering.validator import (
    find_omitted_exception,
    find_scope_mismatch,
    find_stale_citation,
    validate,
)
from sqc.core.retrieval.types import Evidence
from sqc.providers.base import LLMResponse, ProviderRateLimitError, Usage
from sqc.providers.registry import build_entailment_provider


def ev(text: str, handle: str = "E1", effective: date | None = None) -> Evidence:
    return Evidence(
        evidence_id=handle, chunk_ids=(uuid.uuid4(),), document_id=uuid.uuid4(),
        filename="policy.md", text=text, heading_path=("Policy", "1. Access"),
        page_start=1, page_end=1, effective_date=effective, score=1.0, citation="policy.md p1",
    )


def payload(answer: str, claim: str, handles: list[str], **extra):  # noqa: ANN003, ANN201
    return {
        "answer": answer, "answer_type": "yes",
        "claims": [{"text": claim, "evidence_ids": handles}],
        "evidence_sufficient": True, "reason": "stated", **extra,
    }


# ------------------------------------------------------------- lexical


def test_lexical_entailment_accepts_a_restatement():
    result = LexicalEntailment().check(
        "Multi-factor authentication is required for administrator accounts.",
        "Multi-factor authentication is required for administrator accounts that "
        "access production systems.",
    )
    assert result.label is Entailment.SUPPORTED
    assert result.score > 0.6


def test_lexical_entailment_rejects_a_claim_the_evidence_does_not_make():
    result = LexicalEntailment().check(
        "The company holds ISO 27001 certification and completes annual penetration tests.",
        "Multi-factor authentication is required for administrator accounts.",
    )
    assert result.label is Entailment.UNSUPPORTED


def test_lexical_entailment_catches_a_fabricated_figure():
    """The decisive case: everything else in the claim matches, and one
    number does not appear anywhere in the evidence."""
    result = LexicalEntailment().check(
        "Passwords are rotated every 45 days.",
        "Passwords must be at least 12 characters and are rotated every 90 days.",
    )
    assert result.label is Entailment.UNSUPPORTED
    assert "45" in result.detail


def test_lexical_entailment_detects_a_direct_contradiction():
    result = LexicalEntailment().check(
        "Multi-factor authentication is required for all staff accounts.",
        "Multi-factor authentication is not required for standard staff accounts.",
    )
    assert result.label is Entailment.CONTRADICTED


def test_lexical_entailment_declines_rather_than_guessing_on_short_claims():
    result = LexicalEntailment().check("Yes.", "Some long policy text about encryption.")
    assert result.label is Entailment.UNKNOWN
    assert result.ok, "an undecidable check must not count against the claim"


def test_lexical_entailment_is_deterministic():
    checker = LexicalEntailment()
    a = checker.check("Keys rotate annually.", "Encryption keys are rotated annually.")
    b = checker.check("Keys rotate annually.", "Encryption keys are rotated annually.")
    assert a == b


def test_fakes_satisfy_the_entailment_protocol():
    assert isinstance(LexicalEntailment(), EntailmentProvider)


# ------------------------------------------------------------- llm-backed


def test_llm_entailment_reads_the_verdict():
    class StubLLM:
        model = "stub"

        def complete_structured(self, system, user, schema, max_tokens=2048):  # noqa: ANN001
            assert "<statement>" in user and "<passage>" in user
            return LLMResponse(
                data={"verdict": "contradicted", "reason": "the passage says the opposite"},
                model="stub", usage=Usage(10, 5, 1),
            )

    result = LLMEntailment(StubLLM()).check("claim", "evidence")
    assert result.label is Entailment.CONTRADICTED
    assert "opposite" in result.detail


def test_llm_entailment_outage_does_not_condemn_the_claim():
    """An outage in the checker must not turn a good answer into a flagged
    one. Unknown is the only safe reading of no answer."""
    class BrokenLLM:
        model = "broken"

        def complete_structured(self, system, user, schema, max_tokens=2048):  # noqa: ANN001
            raise ProviderRateLimitError("down")

    result = LLMEntailment(BrokenLLM()).check("claim", "evidence")
    assert result.label is Entailment.UNKNOWN
    assert result.ok


def test_llm_entailment_handles_an_unreadable_verdict():
    class WeirdLLM:
        model = "weird"

        def complete_structured(self, system, user, schema, max_tokens=2048):  # noqa: ANN001
            return LLMResponse(data={"verdict": "maybe?", "reason": ""}, model="weird")

    assert LLMEntailment(WeirdLLM()).check("c", "e").label is Entailment.UNKNOWN


# --------------------------------------------- entailment inside validate


def test_a_citation_alone_no_longer_confers_support():
    """The Step 7 requirement. Before this, a claim that cited a real chunk
    was supported even when the chunk said nothing of the kind."""
    evidence = [ev("Passwords must be at least 12 characters.")]
    outcome = validate(
        payload("We hold ISO 27001 certification.",
                "The company holds ISO 27001 certification.", ["E1"]),
        evidence,
        entailment=LexicalEntailment(),
    )
    assert outcome.status is AnswerStatus.REVIEW_REQUIRED
    assert outcome.signals["claims_unentailed"] == 1
    assert outcome.claims[0].entailment == "unsupported"
    assert outcome.claims[0].entailment_score is not None, "raw signal must be preserved"


def test_without_an_entailment_provider_behaviour_is_unchanged():
    evidence = [ev("Passwords must be at least 12 characters.")]
    outcome = validate(
        payload("Passwords are at least 12 characters.",
                "Passwords must be at least 12 characters.", ["E1"]),
        evidence,
    )
    assert outcome.status is AnswerStatus.SUPPORTED
    assert outcome.signals["entailment_provider"] is None
    assert outcome.claims[0].entailment is None


def test_entailment_never_deletes_a_claim_only_downgrades_the_answer():
    """The checker is probabilistic, so it flags for review rather than
    silently discarding a claim a human might well accept."""
    evidence = [ev("Passwords are rotated every 90 days.")]
    outcome = validate(
        payload("Passwords rotate every 45 days.", "Passwords are rotated every 45 days.",
                ["E1"]),
        evidence,
        entailment=LexicalEntailment(),
    )
    assert outcome.status is AnswerStatus.REVIEW_REQUIRED
    assert outcome.claims[0].supported is True, "the claim survives for a human to judge"
    assert outcome.claims[0].entailment == "unsupported"


def test_lexical_entailment_cannot_catch_a_paraphrased_quantity():
    """A known limitation, asserted so it is not mistaken for coverage.

    'Backups are taken every hour' against evidence saying 'every four hours'
    shares enough content words to land between the thresholds, so the
    lexical checker returns unknown rather than guessing. Catching this needs
    SQC_ENTAILMENT_PROVIDER=llm. Recording the gap here stops a future reader
    assuming lexical checking is sufficient.
    """
    result = LexicalEntailment().check(
        "Backups are taken every hour.", "Backups run every four hours."
    )
    assert result.label is Entailment.UNKNOWN
    assert result.ok, "an undecided verdict must not be counted as a failure"


def test_contradicted_claim_is_reported_distinctly_from_unsupported():
    evidence = [ev("Multi-factor authentication is not required for standard accounts.")]
    outcome = validate(
        payload("MFA is required.", "Multi-factor authentication is required for accounts.",
                ["E1"]),
        evidence,
        entailment=LexicalEntailment(),
    )
    assert outcome.signals["claims_contradicted"] == 1
    assert "contradicted" in outcome.reason


# ------------------------------------------------------- omitted exception


def test_exception_present_in_evidence_but_absent_from_answer_forces_review():
    """The canonical dangerous answer: every word true, and misleading.
    The evidence exempts break-glass accounts; the answer does not say so."""
    evidence = [
        ev(
            "Multi-factor authentication is required for all employee accounts. "
            "Break-glass service accounts are exempt from multi-factor authentication "
            "and use hardware tokens instead."
        )
    ]
    outcome = validate(
        payload("Yes, MFA is required for all employee accounts.",
                "MFA is required for all employee accounts.", ["E1"]),
        evidence,
        entailment=LexicalEntailment(),
    )
    assert outcome.status is AnswerStatus.REVIEW_REQUIRED
    assert outcome.signals["exception_omitted"] is True


def test_an_answer_that_states_the_exception_is_not_flagged():
    evidence = [
        ev(
            "Multi-factor authentication is required for all employee accounts. "
            "Break-glass service accounts are exempt and use hardware tokens."
        )
    ]
    outcome = validate(
        payload(
            "MFA is required for all employee accounts, except break-glass service "
            "accounts which use hardware tokens.",
            "MFA is required for all employee accounts except break-glass accounts.",
            ["E1"],
        ),
        evidence,
        entailment=LexicalEntailment(),
    )
    assert outcome.signals["exception_omitted"] is False


def test_exception_detector_ignores_evidence_with_no_exception():
    assert find_omitted_exception("Yes.", "Backups run every four hours.") is None


# ------------------------------------------------------------ scope mismatch


def test_universal_question_answered_from_narrow_evidence_forces_review():
    evidence = [ev("MFA is required for administrator accounts accessing production.")]
    outcome = validate(
        payload("Yes, MFA is enforced.", "MFA is required for accounts accessing production.",
                ["E1"]),
        evidence,
        entailment=LexicalEntailment(),
        question="Do you require MFA for all employees?",
    )
    assert outcome.status is AnswerStatus.REVIEW_REQUIRED
    assert outcome.signals["scope_mismatch"] is True


def test_scope_check_does_not_fire_on_a_non_universal_question():
    assert find_scope_mismatch(
        "Do administrators use MFA?", "Yes.", "MFA is required for administrators only."
    ) is None


def test_scope_check_does_not_fire_when_the_answer_states_the_limit():
    assert find_scope_mismatch(
        "Do all staff use MFA?",
        "Only administrators are required to use MFA.",
        "MFA is required for administrators only.",
    ) is None


# ------------------------------------------------------------ stale version


def test_citing_only_the_older_document_when_newer_was_retrieved_forces_review():
    older = ev("Passwords must be at least 12 characters.", "E1", date(2021, 1, 12))
    newer = ev("Passwords must be at least 16 characters.", "E2", date(2024, 3, 3))
    outcome = validate(
        payload("The minimum password length is 12 characters.",
                "Passwords must be at least 12 characters.", ["E1"]),
        [older, newer],
        entailment=LexicalEntailment(),
        question="What is the minimum password length?",
    )
    assert outcome.status is AnswerStatus.REVIEW_REQUIRED
    assert outcome.signals["stale_citation"] is True
    assert "2024" in outcome.reason


def test_citing_the_newest_document_is_not_stale():
    older = ev("Passwords must be at least 12 characters.", "E1", date(2021, 1, 12))
    newer = ev("Passwords must be at least 16 characters.", "E2", date(2024, 3, 3))
    assert find_stale_citation(
        [Claim(text="c", citations=(Citation(
            evidence_id="E2", chunk_ids=(uuid.uuid4(),), document_id=uuid.uuid4(),
            filename="policy.md", heading_path=(), page_start=1, page_end=1,
            effective_date=date(2024, 3, 3)),), supported=True)],
        [older, newer],
    ) is None


def test_stale_check_needs_at_least_two_dated_sources():
    assert find_stale_citation([], [ev("text", "E1", date(2024, 1, 1))]) is None


# ---------------------------------------------------------------- registry


def test_registry_builds_the_configured_entailment_provider():
    from sqc.config import Settings

    assert isinstance(build_entailment_provider(Settings()), LexicalEntailment)
    assert build_entailment_provider(Settings(entailment_provider="none")) is None
    with pytest.raises(Exception, match="unknown SQC_ENTAILMENT_PROVIDER"):
        build_entailment_provider(Settings(entailment_provider="magic"))


def test_entailment_defaults_to_lexical_not_none():
    """It costs nothing and catches the fabricated-figure case, which is the
    most damaging hallucination this product can produce."""
    from sqc.config import Settings

    assert Settings().entailment_provider == "lexical"


def test_custom_entailment_provider_can_be_injected():
    class AlwaysContradicts:
        name = "always-contradicts"

        def check(self, claim: str, evidence: str) -> EntailmentResult:
            return EntailmentResult(Entailment.CONTRADICTED, 0.0, "by construction")

    outcome = validate(
        payload("Anything.", "Some claim about the policy text.", ["E1"]),
        [ev("Some claim about the policy text is written here.")],
        entailment=AlwaysContradicts(),
    )
    assert outcome.signals["entailment_provider"] == "always-contradicts"
    assert outcome.status is AnswerStatus.REVIEW_REQUIRED

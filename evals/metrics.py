"""Scoring.

Every metric here is computed from the golden file and the pipeline's own
output. No model is asked whether an answer was good, because a model that
grades its own output agrees with itself, and a different model's opinion
is another unvalidated model rather than a measurement.

The pair that matters is coverage and false-refusal rate. Coverage alone
rewards answering everything; refusal precision alone rewards refusing
everything. A change that raises coverage while raising false answers is a
regression, whatever the headline number does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqc.core.answering.schema import AnswerResult, AnswerStatus

from evals.dataset import EvalCase


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case_id: str
    question: str
    category: str
    expected: str
    actual: str
    passed: bool
    failures: tuple[str, ...] = ()
    retrieval_hit: bool | None = None
    """Whether every expect_text string appeared in the retrieved evidence.
    None for refusal cases, which state no retrieval expectation."""
    hallucinated: bool = False
    failure_stage: str | None = None
    """Where this case broke: retrieval, answering, validation or provider.
    A refusal rate says the system is cautious; this says where to fix it."""
    review_reason: str = ""
    provider_error: str = ""
    """Why the model call failed, verbatim. Without it the report says a
    provider failure happened but not whether it was throttling, an exhausted
    quota or an outage - three problems with three different responses."""
    citations: int = 0
    claims_total: int = 0
    claims_unentailed: int = 0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "category": self.category,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "failures": list(self.failures),
            "retrieval_hit": self.retrieval_hit,
            "hallucinated": self.hallucinated,
            "failure_stage": self.failure_stage,
            "review_reason": self.review_reason[:200],
            "provider_error": self.provider_error[:300],
            "citations": self.citations,
            "claims_total": self.claims_total,
            "claims_unentailed": self.claims_unentailed,
            "latency_ms": self.latency_ms,
        }


def _status_matches(expected: str, actual: AnswerStatus) -> bool:
    if expected == "answered":
        return actual in (AnswerStatus.SUPPORTED, AnswerStatus.REVIEW_REQUIRED)
    return actual.value == expected


def score_case(case: EvalCase, result: AnswerResult) -> CaseOutcome:
    """Check one answer against its golden expectations."""
    failures: list[str] = []

    # A call that never completed cannot pass. The pipeline refuses on a
    # provider error, and that refusal would otherwise satisfy a case that
    # expected a refusal - so an outage scored as a correct refusal and
    # inflated the pass count. One run reported 9 passed from 3 scored cases.
    if result.failure_stage == "provider":
        return CaseOutcome(
            case_id=case.id,
            question=case.question,
            category=case.category,
            expected=case.expect_status,
            actual="provider_error",
            passed=False,
            failures=(f"provider call did not complete: {result.reason[:160]}",),
            failure_stage="provider",
            review_reason=result.reason,
            provider_error=result.reason,
            latency_ms=result.latency_ms,
        )

    if not _status_matches(case.expect_status, result.status):
        failures.append(f"status {result.status.value}, expected {case.expect_status}")

    # Retrieval ground truth, checked independently of the answer so a
    # retrieval failure is never reported as an answering failure.
    retrieval_hit: bool | None = None
    if case.expect_text:
        evidence_blob = " ".join(item.text for item in result.evidence).lower()
        missing = [t for t in case.expect_text if t.lower() not in evidence_blob]
        retrieval_hit = not missing
        if missing:
            failures.append(f"evidence missing: {', '.join(missing)}")

    answer_blob = result.answer.lower()

    # Hallucination is measured, not estimated: a forbidden literal in the
    # answer, or a claim the validator could not tie to retrieved evidence.
    forbidden = [t for t in case.forbid_literals if t.lower() in answer_blob]
    unsupported = [c for c in result.claims if not c.supported]
    hallucinated = bool(forbidden or unsupported)
    if forbidden:
        failures.append(f"answer states forbidden: {', '.join(forbidden)}")
    if unsupported:
        failures.append(f"{len(unsupported)} claim(s) not supported by retrieved evidence")

    if case.answerable:
        missing_phrases = [t for t in case.expect_answer_contains if t.lower() not in answer_blob]
        if missing_phrases:
            failures.append(f"answer missing: {', '.join(missing_phrases)}")

        if case.expect_sections:
            paths = " ".join(
                " > ".join(c.heading_path) for c in result.citations
            ).lower()
            missing_sections = [s for s in case.expect_sections if s.lower() not in paths]
            if missing_sections:
                failures.append(f"no citation in section(s): {', '.join(missing_sections)}")
    elif result.answer:
        failures.append("produced an answer for a case with no supporting evidence")

    signals = result.validation_signals or {}
    # A case that did not land as expected is attributed to a stage. When the
    # pipeline recorded one, that wins; otherwise a missing-evidence failure
    # is retrieval's and anything else belongs to the answering model.
    stage = result.failure_stage
    if stage is None and failures:
        stage = "retrieval" if retrieval_hit is False else "answering"

    return CaseOutcome(
        case_id=case.id,
        question=case.question,
        category=case.category,
        expected=case.expect_status,
        actual=result.status.value,
        passed=not failures,
        failures=tuple(failures),
        retrieval_hit=retrieval_hit,
        hallucinated=hallucinated,
        failure_stage=stage,
        review_reason=result.reason if result.status.value == "review_required" else "",
        provider_error=result.reason if result.failure_stage == "provider" else "",
        citations=len(result.citations),
        claims_total=len(result.claims),
        claims_unentailed=int(signals.get("claims_unentailed", 0))
        + int(signals.get("claims_contradicted", 0)),
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


@dataclass(frozen=True, slots=True)
class Report:
    outcomes: tuple[CaseOutcome, ...] = ()
    dataset: str = ""
    model: str = ""
    embedding_model: str = ""
    rerank_provider: str = ""
    prompt_version: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def failed(self) -> int:
        return len(self.outcomes) - self.passed


def compute_metrics(outcomes: list[CaseOutcome], cases: list[EvalCase]) -> dict[str, Any]:
    """The numbers a change has to be judged against."""
    by_id = {c.id: c for c in cases}

    # A case whose model call never completed says nothing about answer
    # quality. Counting it as a refusal reports a provider outage as the
    # model being over-cautious: one run had 17 of 24 calls fail and showed
    # a coverage of 16.7% that was entirely the free tier falling over.
    provider_failures = [o for o in outcomes if o.failure_stage == "provider"]
    scored = [o for o in outcomes if o.failure_stage != "provider"]

    total = len(scored)
    answerable = [o for o in scored if by_id[o.case_id].answerable]
    unanswerable = [o for o in scored if not by_id[o.case_id].answerable]

    answered = [o for o in scored if o.actual in ("supported", "review_required")]
    refused = [o for o in scored if o.actual == "refused"]

    # A refusal is correct when the case was genuinely unanswerable.
    correct_refusals = [o for o in refused if not by_id[o.case_id].answerable]
    false_refusals = [o for o in refused if by_id[o.case_id].answerable]
    # An answer to an unanswerable case is the dangerous failure.
    false_answers = [o for o in answered if not by_id[o.case_id].answerable]

    retrieval_checked = [o for o in scored if o.retrieval_hit is not None]

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return {
        "cases": len(outcomes),
        "scored_cases": total,
        "provider_failures": len(provider_failures),
        "passed": sum(1 for o in outcomes if o.passed),
        "failed": sum(1 for o in outcomes if not o.passed),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "coverage": ratio(len([o for o in answerable if o.actual != "refused"]), len(answerable)),
        "false_refusal_rate": ratio(len(false_refusals), len(answerable)),
        "false_answer_rate": ratio(len(false_answers), len(unanswerable)),
        "refusal_precision": ratio(len(correct_refusals), len(refused)),
        "hallucination_rate": ratio(sum(1 for o in scored if o.hallucinated), total),
        "retrieval_recall": ratio(
            sum(1 for o in retrieval_checked if o.retrieval_hit), len(retrieval_checked)
        ),
        "citation_rate": ratio(len([o for o in answered if o.citations > 0]), len(answered)),
        "review_required_rate": ratio(
            len([o for o in scored if o.actual == "review_required"]), total
        ),
        "retrieval_failure_rate": ratio(
            len([o for o in retrieval_checked if not o.retrieval_hit]), len(retrieval_checked)
        ),
        "unsupported_claim_rate": ratio(
            sum(o.claims_unentailed for o in scored),
            max(sum(o.claims_total for o in scored), 0),
        ),
        "provider_error_summary": _summarise_errors(provider_failures),
        "failures_by_stage": {
            stage: len([o for o in outcomes if not o.passed and o.failure_stage == stage])
            for stage in ("retrieval", "answering", "validation", "provider")
        },
        "by_category": {
            category: {
                "cases": len([o for o in outcomes if o.category == category]),
                "passed": len(
                    [o for o in outcomes if o.category == category and o.passed]
                ),
            }
            for category in sorted({o.category for o in outcomes})
        },
        "mean_latency_ms": (
            round(sum(o.latency_ms for o in outcomes) / total) if total else 0
        ),
        "total_tokens": sum(o.input_tokens + o.output_tokens for o in outcomes),
    }


def format_report(report: Report) -> str:
    """Human-readable summary. Failures first, because they are the point."""
    lines: list[str] = []
    metrics = report.metrics

    failures = [o for o in report.outcomes if not o.passed]
    if failures:
        lines.append(f"FAILURES ({len(failures)})")
        for outcome in failures:
            lines.append(f"  {outcome.case_id}: {outcome.question}")
            for failure in outcome.failures:
                lines.append(f"      - {failure}")
        lines.append("")

    lines.append(f"dataset      : {report.dataset}")
    lines.append(
        f"providers    : llm={report.model} embed={report.embedding_model} "
        f"rerank={report.rerank_provider} prompt={report.prompt_version}"
    )
    lines.append("")
    lines.append(f"  cases              {metrics['cases']}"
                 f"   passed {metrics['passed']}   failed {metrics['failed']}")
    failed_calls = metrics.get("provider_failures", 0)
    if failed_calls:
        lines.append(
            f"  provider failures  {failed_calls}   EXCLUDED from the rates below"
        )
        lines.append(
            f"  rates computed over {metrics.get('scored_cases', 0)} of "
            f"{metrics['cases']} cases - re-run to fill the gaps"
        )
    lines.append("")

    def show(label: str, key: str, note: str = "") -> None:
        value = metrics.get(key)
        shown = "n/a" if value is None else f"{value:.1%}" if isinstance(value, float) else value
        lines.append(f"  {label:<22} {shown:>7}   {note}")

    show("coverage", "coverage", "answerable cases that got an answer")
    show("false refusal rate", "false_refusal_rate", "answerable but refused - lost value")
    show("false answer rate", "false_answer_rate", "unanswerable but answered - DANGEROUS")
    show("refusal precision", "refusal_precision", "refusals that were correct")
    show("hallucination rate", "hallucination_rate", "unsupported claims or forbidden text")
    show("retrieval recall", "retrieval_recall", "expected evidence actually retrieved")
    show("citation rate", "citation_rate", "answers carrying at least one citation")
    show("review required rate", "review_required_rate", "sent to a human")
    show("retrieval failure rate", "retrieval_failure_rate", "expected evidence not found")
    show("unsupported claim rate", "unsupported_claim_rate", "claims the evidence does not support")
    lines.append("")

    provider_errors = metrics.get("provider_error_summary") or {}
    if provider_errors:
        lines.append("  provider errors")
        for message, count in provider_errors.items():
            lines.append(f"    {count} x {message}")
        lines.append("")

    stages = metrics.get("failures_by_stage") or {}
    if any(stages.values()):
        lines.append("  failures by stage")
        for stage, count in stages.items():
            if count:
                lines.append(f"    {stage:<12} {count}")
        lines.append("")

    categories = metrics.get("by_category") or {}
    if categories:
        lines.append("  by category")
        for category, counts in categories.items():
            mark = "" if counts["passed"] == counts["cases"] else "   <-"
            lines.append(
                f"    {category:<24} {counts['passed']}/{counts['cases']}{mark}"
            )
        lines.append("")

    lines.append(f"  mean latency         {metrics['mean_latency_ms']:>5} ms"
                 f"   tokens {metrics['total_tokens']}")
    return "\n".join(lines)


def _summarise_errors(outcomes: list[CaseOutcome]) -> dict[str, int]:
    """Group provider errors by their distinguishing text.

    Collapsed to a short signature so ten identical 429s read as one line
    with a count, which is what makes a systemic problem obvious rather than
    buried in repetition.
    """
    counts: dict[str, int] = {}
    for outcome in outcomes:
        message = outcome.provider_error or "unknown provider failure"
        for marker in ("RESOURCE_EXHAUSTED", "quota", "429", "503", "500", "timeout", "429"):
            if marker.lower() in message.lower():
                message = f"{marker}: " + message.split(marker, 1)[-1][:90].strip(" :\"'}")
                break
        else:
            message = message[:110]
        counts[message] = counts.get(message, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

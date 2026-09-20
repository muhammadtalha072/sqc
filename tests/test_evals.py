"""Tests for the eval harness.

The harness decides whether a change ships, so its arithmetic needs testing
more carefully than the thing it measures. A scorer that quietly counts a
false answer as a pass would make every future number meaningless.
"""

from __future__ import annotations

import json
import pathlib
import uuid

import pytest

from evals.cassette import CassetteLLM, prompt_key
from evals.dataset import DatasetError, load_dataset
from evals.metrics import compute_metrics, format_report, score_case
from evals.run import DEFAULT_DATASET, check_regression
from sqc.core.answering.schema import (
    AnswerResult,
    AnswerStatus,
    AnswerType,
    Citation,
    Claim,
)
from sqc.core.retrieval.types import Evidence
from sqc.providers.base import LLMResponse, ProviderResponseError, Usage

REPO = pathlib.Path(__file__).resolve().parents[1]


def evidence(text: str, heading: tuple[str, ...] = ("Policy", "A. DEFINITIONS")) -> Evidence:
    return Evidence(
        evidence_id="E1", chunk_ids=(uuid.uuid4(),), document_id=uuid.uuid4(),
        filename="depaul_isp.pdf", text=text, heading_path=heading,
        page_start=1, page_end=1, effective_date=None, score=1.0, citation="depaul_isp.pdf p1",
    )


def citation(heading: tuple[str, ...] = ("Policy", "A. DEFINITIONS")) -> Citation:
    return Citation(
        evidence_id="E1", chunk_ids=(uuid.uuid4(),), document_id=uuid.uuid4(),
        filename="depaul_isp.pdf", heading_path=heading, page_start=1, page_end=1,
    )


def result(**overrides) -> AnswerResult:  # noqa: ANN003
    base = dict(
        question="q", status=AnswerStatus.SUPPORTED, answer="",
        answer_type=AnswerType.YES, claims=(), citations=(), evidence=(),
    )
    return AnswerResult(**{**base, **overrides})


def case(**overrides):  # noqa: ANN003, ANN201
    from evals.dataset import EvalCase

    base = dict(id="c1", question="q", expect_status="answered", expect_text=("x",))
    return EvalCase(**{**base, **overrides})


# ------------------------------------------------------------- dataset


def test_shipped_dataset_loads_and_is_balanced():
    dataset = load_dataset(REPO / DEFAULT_DATASET)
    assert dataset.document == "depaul_isp.pdf"
    answerable = [c for c in dataset.cases if c.answerable]
    unanswerable = [c for c in dataset.cases if not c.answerable]
    assert len(answerable) >= 5
    assert len(unanswerable) >= 5, (
        "without unanswerable cases a system that answers everything scores perfectly"
    )


def test_every_unanswerable_case_pins_what_must_not_be_said():
    dataset = load_dataset(REPO / DEFAULT_DATASET)
    for entry in dataset.cases:
        if not entry.answerable:
            assert entry.forbid_literals, (
                f"{entry.id}: a refusal case needs forbid_literals, or a confidently "
                "wrong answer would score the same as a correct refusal"
            )


def test_answerable_case_without_expect_text_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\ndocument: d.pdf\ncases:\n"
        "  - id: a\n    question: q?\n    expect_status: answered\n"
    )
    with pytest.raises(DatasetError, match="expect_text"):
        load_dataset(path)


def test_unknown_status_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\ndocument: d.pdf\ncases:\n"
        "  - id: a\n    question: q?\n    expect_status: probably\n    expect_text: t\n"
    )
    with pytest.raises(DatasetError, match="expect_status"):
        load_dataset(path)


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\ndocument: d.pdf\ncases:\n"
        "  - id: a\n    question: q?\n    expect_status: refused\n"
        "  - id: a\n    question: r?\n    expect_status: refused\n"
    )
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(path)


# -------------------------------------------------------------- scoring


def test_correct_supported_answer_passes():
    outcome = score_case(
        case(expect_text=("Covered Data",), expect_answer_contains=("social security",),
             expect_sections=("DEFINITIONS",)),
        result(
            answer="Covered Data includes social security numbers.",
            evidence=(evidence("Covered Data includes social security numbers."),),
            citations=(citation(),),
            claims=(Claim(text="c", citations=(citation(),), supported=True),),
        ),
    )
    assert outcome.passed
    assert outcome.retrieval_hit is True
    assert outcome.hallucinated is False


def test_missing_evidence_is_reported_as_retrieval_not_answering():
    outcome = score_case(
        case(expect_text=("Covered Data",)),
        result(answer="Something else.", evidence=(evidence("unrelated text"),)),
    )
    assert not outcome.passed
    assert outcome.retrieval_hit is False
    assert any("evidence missing" in f for f in outcome.failures)


def test_forbidden_literal_counts_as_hallucination():
    outcome = score_case(
        case(expect_status="refused", expect_text=(), forbid_literals=("SOC 2",)),
        result(status=AnswerStatus.SUPPORTED, answer="Yes, we hold SOC 2 Type II.",
               evidence=(evidence("nothing relevant"),)),
    )
    assert outcome.hallucinated
    assert any("forbidden" in f for f in outcome.failures)


def test_unsupported_claim_counts_as_hallucination():
    outcome = score_case(
        case(expect_text=("x",)),
        result(
            answer="x is true",
            evidence=(evidence("x"),),
            claims=(Claim(text="invented", supported=False, problem="no evidence"),),
        ),
    )
    assert outcome.hallucinated


def test_correct_refusal_passes():
    outcome = score_case(
        case(expect_status="refused", expect_text=()),
        result(status=AnswerStatus.REFUSED, answer=""),
    )
    assert outcome.passed
    assert outcome.retrieval_hit is None


def test_answered_alias_accepts_either_supported_or_review():
    for status in (AnswerStatus.SUPPORTED, AnswerStatus.REVIEW_REQUIRED):
        outcome = score_case(
            case(expect_status="answered", expect_text=("x",)),
            result(status=status, answer="x", evidence=(evidence("x"),)),
        )
        assert outcome.passed, status


def test_review_required_does_not_satisfy_an_expected_refusal():
    outcome = score_case(
        case(expect_status="refused", expect_text=()),
        result(status=AnswerStatus.REVIEW_REQUIRED, answer="probably yes"),
    )
    assert not outcome.passed


# -------------------------------------------------------------- metrics


def test_metrics_separate_the_two_ways_of_being_wrong():
    cases = [
        case(id="a1", expect_status="answered", expect_text=("x",)),
        case(id="a2", expect_status="answered", expect_text=("x",)),
        case(id="r1", expect_status="refused", expect_text=()),
        case(id="r2", expect_status="refused", expect_text=()),
    ]
    outcomes = [
        score_case(cases[0], result(answer="x", evidence=(evidence("x"),),
                                    citations=(citation(),))),
        # answerable but refused -> false refusal, lost value
        score_case(cases[1], result(status=AnswerStatus.REFUSED, answer="",
                                    evidence=(evidence("x"),))),
        score_case(cases[2], result(status=AnswerStatus.REFUSED, answer="")),
        # unanswerable but answered -> false answer, the dangerous one
        score_case(cases[3], result(status=AnswerStatus.SUPPORTED, answer="yes")),
    ]
    metrics = compute_metrics(outcomes, cases)

    assert metrics["coverage"] == 0.5
    assert metrics["false_refusal_rate"] == 0.5
    assert metrics["false_answer_rate"] == 0.5
    assert metrics["refusal_precision"] == 0.5
    assert metrics["retrieval_recall"] == 1.0


def test_answering_everything_scores_badly_not_perfectly():
    """The property that makes the suite meaningful: a system that never
    refuses cannot score well, however fluent its answers."""
    cases = [
        case(id="a1", expect_status="answered", expect_text=("x",)),
        case(id="r1", expect_status="refused", expect_text=(), forbid_literals=("SOC 2",)),
        case(id="r2", expect_status="refused", expect_text=(), forbid_literals=("AES",)),
    ]
    outcomes = [
        score_case(cases[0], result(answer="x", evidence=(evidence("x"),), citations=(citation(),))),
        score_case(cases[1], result(answer="We hold SOC 2.", evidence=(evidence("y"),))),
        score_case(cases[2], result(answer="We use AES-256.", evidence=(evidence("y"),))),
    ]
    metrics = compute_metrics(outcomes, cases)
    assert metrics["coverage"] == 1.0, "it answered everything"
    assert metrics["false_answer_rate"] == 1.0, "and every refusal case was wrong"
    assert metrics["hallucination_rate"] > 0


def test_report_formats_without_crashing_on_empty_categories():
    cases = [case(id="r1", expect_status="refused", expect_text=())]
    outcomes = [score_case(cases[0], result(status=AnswerStatus.REFUSED, answer=""))]
    from evals.metrics import Report

    text = format_report(Report(outcomes=tuple(outcomes), dataset="d",
                                metrics=compute_metrics(outcomes, cases)))
    assert "coverage" in text
    assert "n/a" in text, "metrics with no denominator must read n/a, not 0%"


# ----------------------------------------------------------- regression


def test_regression_gate_blocks_more_false_answers():
    problems = check_regression(
        {"false_answer_rate": 0.10, "hallucination_rate": 0.0, "false_refusal_rate": 0.0},
        {"false_answer_rate": 0.00, "hallucination_rate": 0.0, "false_refusal_rate": 0.0},
    )
    assert any("false_answer_rate" in p for p in problems)


def test_regression_gate_tolerates_lower_coverage():
    """Answering less often while being wrong less often is an improvement,
    so coverage is reported but never gates a release."""
    problems = check_regression(
        {"coverage": 0.50, "false_answer_rate": 0.0, "hallucination_rate": 0.0,
         "false_refusal_rate": 0.0},
        {"coverage": 0.90, "false_answer_rate": 0.0, "hallucination_rate": 0.0,
         "false_refusal_rate": 0.0},
    )
    assert problems == []


def test_regression_gate_catches_retrieval_decay():
    problems = check_regression(
        {"retrieval_recall": 0.60}, {"retrieval_recall": 0.95}
    )
    assert any("retrieval_recall" in p for p in problems)


# ------------------------------------------------------------ cassette


def test_cassette_records_then_replays_without_calling_the_provider(tmp_path):
    calls = 0

    class CountingLLM:
        model = "test-model"

        def complete_structured(self, system, user, schema, max_tokens=2048):  # noqa: ANN001
            nonlocal calls
            calls += 1
            return LLMResponse(data={"answer": "yes"}, model=self.model,
                               usage=Usage(10, 5, 1), stop_reason="stop")

    recorder = CassetteLLM(CountingLLM(), tmp_path, mode="record")
    first = recorder.complete_structured("sys", "user", {})
    second = recorder.complete_structured("sys", "user", {})

    assert calls == 1, "the second identical prompt must come from cache"
    assert first.data == second.data
    assert recorder.hits == 1 and recorder.misses == 1
    assert second.usage.requests == 0, "a replay costs nothing and must not be billed"


def test_replay_mode_fails_loudly_on_a_miss(tmp_path):
    replayer = CassetteLLM(None, tmp_path, mode="replay")
    with pytest.raises(ProviderResponseError, match="no recorded response"):
        replayer.complete_structured("sys", "user", {})


def test_changing_the_prompt_misses_the_cache(tmp_path):
    """A prompt edit must not be scored against output recorded for the old
    prompt, which would make a prompt regression invisible."""
    assert prompt_key("sys", "user a", "m") != prompt_key("sys", "user b", "m")
    assert prompt_key("sys", "user", "model-1") != prompt_key("sys", "user", "model-2")


def test_recorded_file_is_readable_json(tmp_path):
    class StubLLM:
        model = "m"

        def complete_structured(self, system, user, schema, max_tokens=2048):  # noqa: ANN001
            return LLMResponse(data={"answer": "x"}, model="m", usage=Usage(1, 1, 1))

    CassetteLLM(StubLLM(), tmp_path, mode="record").complete_structured("s", "u", {})
    files = list(pathlib.Path(tmp_path).glob("*.json"))
    assert len(files) == 1
    stored = json.loads(files[0].read_text())
    assert stored["data"] == {"answer": "x"}
    assert "_prompt_preview" in stored, "a human reading a diff needs to see the prompt"


# ------------------------------------------------- provider error summarising


def test_a_cassette_miss_is_not_reported_as_a_server_error():
    """A prompt hash is 32 hex digits, so it will sooner or later contain
    "500" or "429". Matching those as bare substrings turned a replay miss on
    prompt 5007885c... into "500: 7885c8...", which reads as a provider
    outage that never happened and sends the reader to a status page."""
    from evals.metrics import _summarise_errors

    miss = (
        "the answering model could not be reached: no recorded response for this "
        "prompt (5007885c864e2496c8803b4dd64a6588). Re-run with --record to capture it."
    )
    outcome = score_case(
        case(expect_status="answered", expect_text=("x",)),
        result(failure_stage="provider", reason=miss),
    )
    summary = _summarise_errors([outcome])

    assert len(summary) == 1
    label = next(iter(summary))
    assert label.startswith("cassette miss"), label
    assert not label.startswith("500"), "a prompt hash was read as an HTTP status"


def test_a_real_server_error_is_still_recognised():
    from evals.metrics import _summarise_errors

    for text_value, expected in (
        ("provider returned 500 Internal Server Error", "500"),
        ("provider returned 503 overloaded", "503"),
        ("rate limited: 429 too many requests", "429"),
        ("quota exhausted: RESOURCE_EXHAUSTED daily limit", "RESOURCE_EXHAUSTED"),
    ):
        outcome = score_case(
            case(expect_status="answered", expect_text=("x",)),
            result(failure_stage="provider", reason=text_value),
        )
        label = next(iter(_summarise_errors([outcome])))
        assert label.startswith(expected), f"{text_value!r} summarised as {label!r}"


def test_identical_errors_still_collapse_to_one_line():
    from evals.metrics import _summarise_errors

    outcomes = [
        score_case(
            case(id=f"c{i}", expect_status="answered", expect_text=("x",)),
            result(failure_stage="provider", reason="quota exhausted: RESOURCE_EXHAUSTED"),
        )
        for i in range(3)
    ]
    summary = _summarise_errors(outcomes)
    assert len(summary) == 1 and next(iter(summary.values())) == 3


def test_an_error_signature_is_always_one_line():
    """Providers return errors as pretty-printed JSON. A quota failure
    summarised into four lines of braces defeats the purpose of collapsing
    ten identical errors into one line with a count."""
    from evals.metrics import _summarise_errors

    raw = (
        '/v1beta/models/gemini-2.5-flash:generateContent quota exhausted: {\n'
        '  "error": {\n    "code": 429,\n    "message": "You exceeded your current '
        'quota, please check your plan and billing details."\n  }\n}'
    )
    outcome = score_case(
        case(expect_status="answered", expect_text=("x",)),
        result(failure_stage="provider", reason=raw),
    )
    label = next(iter(_summarise_errors([outcome])))

    assert "\n" not in label, f"signature spans lines: {label!r}"
    assert label.startswith("quota"), label
    assert len(label) <= 120


# --------------------------------- answering without the evidence it needed


def test_answering_without_the_expected_evidence_is_counted():
    """The gap a real run exposed. An answerable case whose expected evidence
    never reached the model, answered anyway, showed as coverage 100% and
    hallucination 0% - because the claims were grounded in whatever adjacent
    text did arrive. The answer identified DePaul's Responsible Officer as
    "the Chief Information Privacy Official": cited, supported, and not what
    was asked, because the chunk naming the post was never retrieved."""
    cases = [
        case(id="got-evidence", expect_status="answered", expect_text=("x",)),
        case(id="no-evidence", expect_status="answered", expect_text=("missing",)),
    ]
    outcomes = [
        score_case(cases[0], result(answer="x", evidence=(evidence("x"),),
                                    citations=(citation(),))),
        # answered, grounded in what arrived, but the required evidence is absent
        score_case(cases[1], result(answer="something adjacent",
                                    evidence=(evidence("unrelated text"),),
                                    citations=(citation(),))),
    ]
    metrics = compute_metrics(outcomes, cases)

    assert metrics["answered_without_evidence_rate"] == 0.5
    assert metrics["coverage"] == 1.0, "coverage alone calls this a success"
    assert metrics["hallucination_rate"] == 0.0, "the claims were grounded"


def test_refusing_when_the_evidence_is_absent_is_not_counted():
    """Refusing is the correct response to missing evidence and must not be
    penalised by this metric, or it would reward answering anyway."""
    cases = [case(id="no-evidence", expect_status="answered", expect_text=("missing",))]
    outcomes = [
        score_case(cases[0], result(status=AnswerStatus.REFUSED, answer="",
                                    evidence=(evidence("unrelated"),)))
    ]
    metrics = compute_metrics(outcomes, cases)
    assert metrics["answered_without_evidence_rate"] == 0.0


def test_refusal_cases_are_excluded_from_the_rate():
    """A refusal case states no retrieval expectation, so it can say nothing
    about whether an answer had its evidence."""
    cases = [case(id="r1", expect_status="refused", expect_text=())]
    outcomes = [score_case(cases[0], result(status=AnswerStatus.REFUSED, answer=""))]
    metrics = compute_metrics(outcomes, cases)
    assert metrics["answered_without_evidence_rate"] is None


def test_the_gate_blocks_more_answers_without_evidence():
    problems = check_regression(
        {"answered_without_evidence_rate": 0.20},
        {"answered_without_evidence_rate": 0.00},
    )
    assert any("answered_without_evidence_rate" in p for p in problems)

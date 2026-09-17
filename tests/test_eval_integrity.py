"""Regression tests for the three Step 7 measurement defects.

Each of these was a real failure that produced plausible-looking numbers
describing something other than the system. They are tested because a
measurement bug is worse than a code bug: the code bug shows up as a
failure, the measurement bug shows up as confidence.
"""

from __future__ import annotations

import pathlib
import uuid

import pytest
from sqlalchemy import text

from evals.cassette import CassetteLLM, prompt_key
from evals.dataset import DatasetError, load_dataset, resolve_documents, tenant_for
from evals.metrics import score_case
from sqc.core.answering.schema import AnswerResult, AnswerStatus, Citation, Claim
from sqc.core.ingestion.pipeline import ingest_bytes
from sqc.core.retrieval.types import Evidence
from sqc.db.engine import get_admin_engine, tenant_session
from sqc.db.repository import list_documents
from sqc.providers.base import LLMResponse, Usage

REPO = pathlib.Path(__file__).resolve().parents[1]
DATASETS = sorted((REPO / "evals" / "datasets").glob("*.yaml"))


def case(**overrides):  # noqa: ANN003, ANN201
    from evals.dataset import EvalCase

    base = dict(id="c1", question="q", expect_status="answered", expect_text=("x",))
    return EvalCase(**{**base, **overrides})


# ------------------------------------------------- defect 1: corpus isolation


@pytest.mark.parametrize("path", DATASETS, ids=lambda p: p.stem)
def test_every_dataset_declares_its_own_corpus(path):
    """Without this, retrieval scores describe whatever happens to be in the
    tenant. A setup flag once put nine documents in one tenant and twelve
    cases written against four fixtures were scored against 164 unrelated
    chunks."""
    dataset = load_dataset(path)
    assert dataset.documents, f"{path.name} declares no documents"


def test_dataset_without_documents_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\ndocument: d.md\ncases:\n"
        "  - id: a\n    question: q?\n    expect_status: refused\n"
    )
    with pytest.raises(DatasetError, match="documents"):
        load_dataset(path)


def test_each_dataset_gets_a_distinct_stable_tenant():
    ids = {load_dataset(p).name: tenant_for(load_dataset(p).name) for p in DATASETS}
    assert len(set(ids.values())) == len(ids), "two datasets share a tenant"
    for name, tenant_id in ids.items():
        assert tenant_for(name) == tenant_id, "tenant id must be stable across calls"


def test_declared_documents_resolve_or_say_why():
    dataset = load_dataset(REPO / "evals/datasets/acme-edge-cases-v1.yaml")
    resolved = resolve_documents(dataset, REPO)
    assert len(resolved) >= 4
    assert all(p.exists() for p in resolved)


def test_missing_corpus_names_the_pattern_that_failed(tmp_path):
    path = tmp_path / "ghost.yaml"
    path.write_text(
        "name: ghost\ndocument: nope.md\ndocuments:\n  - evals/fixtures/nothing-*.md\n"
        "cases:\n  - id: a\n    question: q?\n    expect_status: refused\n"
    )
    with pytest.raises(DatasetError, match="nothing-"):
        resolve_documents(load_dataset(path), REPO)


def test_ingesting_one_dataset_leaves_the_other_tenant_untouched():
    """Isolation proved at the database, not by convention."""
    a, b = tenant_for("iso-test-a"), tenant_for("iso-test-b")
    with get_admin_engine().begin() as conn:
        for tenant_id, name in ((a, "iso-a"), (b, "iso-b")):
            conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :n) "
                     "ON CONFLICT (id) DO NOTHING"),
                {"id": tenant_id, "n": name},
            )
    try:
        from sqc.providers.fake import HashingEmbedder

        embedder = HashingEmbedder(dimension=1024)
        ingest_bytes(tenant_id=a, raw=b"# A\n\nAlpha policy text about encryption.",
                     filename="alpha.md", embedder=embedder)
        with tenant_session(b) as session:
            assert list_documents(session) == []
            assert session.execute(
                text("SELECT count(*) FROM chunks WHERE text ILIKE '%Alpha%'")
            ).scalar_one() == 0
    finally:
        with get_admin_engine().begin() as conn:
            conn.execute(text("DELETE FROM tenants WHERE id = ANY(:ids)"), {"ids": [a, b]})


# --------------------------------------------------- defect 2: replay


class CountingLLM:
    model = "gemini-2.5-flash"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, system, user, schema, max_tokens=2048):  # noqa: ANN001
        self.calls += 1
        return LLMResponse(data={"answer": f"reply {self.calls}"}, model=self.model,
                           usage=Usage(10, 5, 1))


def test_record_then_replay_makes_zero_provider_calls(tmp_path):
    """The defect: the cache key includes the model, and replay had no
    provider to read it from, so it keyed every lookup to 'cassette' and
    missed everything it had just recorded."""
    provider = CountingLLM()
    recorder = CassetteLLM(provider, tmp_path, mode="record", model=provider.model)
    for i in range(3):
        recorder.complete_structured("sys", f"question {i}", {})
    assert provider.calls == 3
    assert recorder.recorded == 3

    replayer = CassetteLLM(None, tmp_path, mode="replay", model=provider.model)
    replies = [replayer.complete_structured("sys", f"question {i}", {}).data for i in range(3)]

    assert provider.calls == 3, "replay must not call the provider at all"
    assert replayer.hits == 3 and replayer.misses == 0
    assert replies == [{"answer": f"reply {i + 1}"} for i in range(3)]


def test_replay_without_the_model_name_would_miss():
    """Pins the cause, so a future refactor that drops the model argument
    fails here rather than silently making replay useless."""
    assert prompt_key("s", "u", "gemini-2.5-flash") != prompt_key("s", "u", "cassette")


def test_replayed_responses_are_not_counted_as_billable_requests(tmp_path):
    provider = CountingLLM()
    CassetteLLM(provider, tmp_path, mode="record",
                model=provider.model).complete_structured("s", "u", {})
    replayed = CassetteLLM(None, tmp_path, mode="replay",
                           model=provider.model).complete_structured("s", "u", {})
    assert replayed.usage.requests == 0


def test_refresh_mode_overwrites_a_recording(tmp_path):
    provider = CountingLLM()
    CassetteLLM(provider, tmp_path, mode="record",
                model=provider.model).complete_structured("s", "u", {})
    CassetteLLM(provider, tmp_path, mode="refresh",
                model=provider.model).complete_structured("s", "u", {})
    assert provider.calls == 2


# ------------------------------------------ defect 3: failure-stage attribution


def answer(**overrides) -> AnswerResult:  # noqa: ANN003
    base = dict(question="q", status=AnswerStatus.REFUSED)
    return AnswerResult(**{**base, **overrides})


def evidence(text_value: str = "policy text") -> Evidence:
    return Evidence(
        evidence_id="E1", chunk_ids=(uuid.uuid4(),), document_id=uuid.uuid4(),
        filename="p.md", text=text_value, heading_path=("P",), page_start=1,
        page_end=1, effective_date=None, score=1.0, citation="p.md p1",
    )


def test_provider_failure_is_not_counted_as_a_model_quality_failure():
    """Five of twelve failures in the first real run were provider errors.
    Counting those as false refusals would have made the model look far more
    conservative than it is."""
    outcome = score_case(
        case(expect_status="answered", expect_text=("x",)),
        answer(failure_stage="provider", reason="the answering model could not be reached",
               evidence=(evidence("x"),)),
    )
    assert outcome.failure_stage == "provider"
    assert not outcome.passed
    assert outcome.hallucinated is False


def test_zero_claims_is_an_answering_failure_not_a_validation_one():
    from sqc.core.answering.validator import validate

    outcome = validate(
        {"answer": "Yes.", "answer_type": "yes", "claims": [],
         "evidence_sufficient": True, "reason": "r"},
        [evidence()],
    )
    assert outcome.status is AnswerStatus.REFUSED
    assert outcome.failure_stage == "answering", (
        "a model that produced nothing to check did not fail validation"
    )


def test_unresolvable_citations_are_a_validation_failure():
    from sqc.core.answering.validator import validate

    outcome = validate(
        {"answer": "Yes.", "answer_type": "yes",
         "claims": [{"text": "invented", "evidence_ids": ["E99"]}],
         "evidence_sufficient": True, "reason": "r"},
        [evidence()],
    )
    assert outcome.status is AnswerStatus.REFUSED
    assert outcome.failure_stage == "validation", (
        "the model produced a claim; validation rejected it"
    )


def test_no_evidence_is_a_retrieval_failure():
    from sqc.core.answering.validator import validate

    outcome = validate(
        {"answer": "", "answer_type": "not_found", "claims": [],
         "evidence_sufficient": False, "reason": "r"},
        [],
    )
    assert outcome.failure_stage == "retrieval"


def test_model_saying_evidence_is_insufficient_is_attributed_to_retrieval():
    """The model had evidence and judged it did not answer the question.
    That is retrieval's problem to fix, not the model's."""
    from sqc.core.answering.validator import validate

    outcome = validate(
        {"answer": "", "answer_type": "not_found", "claims": [],
         "evidence_sufficient": False, "reason": "nothing relevant"},
        [evidence()],
    )
    assert outcome.failure_stage == "retrieval"


def test_malformed_output_is_an_answering_failure():
    from sqc.core.answering.validator import validate

    assert validate("not a dict", [evidence()]).failure_stage == "answering"  # type: ignore[arg-type]


def test_a_clean_answer_has_no_failure_stage():
    from sqc.core.answering.validator import validate

    outcome = validate(
        {"answer": "Policy text applies.", "answer_type": "yes",
         "claims": [{"text": "Policy text applies here.", "evidence_ids": ["E1"]}],
         "evidence_sufficient": True, "reason": "stated"},
        [evidence("Policy text applies here.")],
    )
    assert outcome.status is AnswerStatus.SUPPORTED
    assert outcome.failure_stage is None


def test_scorer_falls_back_to_retrieval_when_expected_evidence_is_absent():
    outcome = score_case(
        case(expect_text=("missing phrase",)),
        answer(status=AnswerStatus.SUPPORTED, answer="something", evidence=(evidence("other"),)),
    )
    assert outcome.retrieval_hit is False
    assert outcome.failure_stage == "retrieval"


def test_scorer_attributes_a_wrong_answer_with_good_evidence_to_answering():
    outcome = score_case(
        case(expect_text=("x",), expect_answer_contains=("expected phrase",)),
        answer(status=AnswerStatus.SUPPORTED, answer="unexpected wording",
               evidence=(evidence("x"),),
               claims=(Claim(text="c", citations=(Citation(
                   evidence_id="E1", chunk_ids=(uuid.uuid4(),), document_id=uuid.uuid4(),
                   filename="p.md", heading_path=(), page_start=1, page_end=1),),
                   supported=True),)),
    )
    assert outcome.retrieval_hit is True
    assert outcome.failure_stage == "answering"


def test_runner_reports_an_absent_corpus_instead_of_crashing(tmp_path):
    """An uncommitted third-party PDF is a normal state. A traceback here
    reads like a broken harness, and a zero score would read like a dataset
    the system failed - neither is true."""
    from evals.run import run_dataset

    path = tmp_path / "ghost.yaml"
    path.write_text(
        "name: ghost-corpus\ndocument: nope.pdf\ndocuments:\n  - tests/data/no-such-*.pdf\n"
        "cases:\n  - id: a\n    question: q?\n    expect_status: refused\n"
    )
    with pytest.raises(SystemExit, match="cannot score"):
        run_dataset(tenant_for("ghost-corpus"), str(path), "replay")


# ------------------------------------ defect 4: provider failures pollute rates


def test_provider_failures_are_excluded_from_quality_rates():
    """A run with 17 of 24 calls failing reported coverage of 16.7% and a
    false-refusal rate of 83.3%. Both described the free tier falling over,
    not the model. Rates must be computed over cases that got an answer."""
    from evals.metrics import compute_metrics

    cases = [
        case(id="a1", expect_status="answered", expect_text=("x",)),
        case(id="a2", expect_status="answered", expect_text=("x",)),
        case(id="a3", expect_status="answered", expect_text=("x",)),
    ]
    outcomes = [
        # one real answer, two calls that never completed
        score_case(cases[0], answer(status=AnswerStatus.SUPPORTED, answer="x",
                                    evidence=(evidence("x"),),
                                    citations=(Citation(
                                        evidence_id="E1", chunk_ids=(uuid.uuid4(),),
                                        document_id=uuid.uuid4(), filename="p.md",
                                        heading_path=(), page_start=1, page_end=1),))),
        score_case(cases[1], answer(failure_stage="provider", evidence=(evidence("x"),))),
        score_case(cases[2], answer(failure_stage="provider", evidence=(evidence("x"),))),
    ]
    metrics = compute_metrics(outcomes, cases)

    assert metrics["provider_failures"] == 2
    assert metrics["scored_cases"] == 1
    assert metrics["cases"] == 3, "the total still reports every case attempted"
    assert metrics["coverage"] == 1.0, "the one completed call was answered"
    assert metrics["false_refusal_rate"] == 0.0, (
        "a provider outage is not the model refusing"
    )


def test_a_run_where_every_call_failed_reports_no_rates_rather_than_zeros():
    """Zeros would read as a system that answers nothing. None reads as what
    it is: nothing was measured."""
    from evals.metrics import compute_metrics

    cases = [case(id="a1", expect_status="answered", expect_text=("x",))]
    outcomes = [score_case(cases[0], answer(failure_stage="provider"))]
    metrics = compute_metrics(outcomes, cases)

    assert metrics["provider_failures"] == 1
    assert metrics["scored_cases"] == 0
    assert metrics["coverage"] is None
    assert metrics["false_refusal_rate"] is None


def test_report_warns_that_rates_exclude_failed_calls():
    from evals.metrics import Report, compute_metrics, format_report

    cases = [case(id="a1", expect_status="answered", expect_text=("x",))]
    outcomes = [score_case(cases[0], answer(failure_stage="provider"))]
    text_out = format_report(Report(outcomes=tuple(outcomes), dataset="d",
                                    metrics=compute_metrics(outcomes, cases)))
    assert "provider failures" in text_out
    assert "EXCLUDED" in text_out


def test_retry_budget_stays_small():
    """Raising it to 8 with a two-second base delay cost two minutes of
    sleeping per failing call, recovered nothing across four runs, and spent
    eight requests per failure against a 250-request daily allowance. The
    lever that mattered was not retrying an exhausted quota at all."""
    from sqc.config import Settings

    settings = Settings()
    assert settings.llm_max_attempts <= 4
    worst_case = sum(
        min(settings.llm_base_delay * (2**a), 30.0)
        for a in range(settings.llm_max_attempts - 1)
    )
    assert worst_case < 10, f"a failing call would sleep {worst_case:.0f}s"


def test_a_provider_failure_never_counts_as_a_passing_refusal():
    """A failed call produces status=refused, which matched any case that
    expected a refusal, so an outage inflated the pass count. One run
    reported 9 passed while only 3 cases had actually been scored."""
    from evals.metrics import compute_metrics

    cases = [case(id=f"r{i}", question="q", expect_status="refused", expect_text=())
             for i in range(3)]
    outcomes = [
        score_case(c, answer(failure_stage="provider",
                             reason="the model could not be reached: 503"))
        for c in cases
    ]
    metrics = compute_metrics(outcomes, cases)
    assert metrics["passed"] == 0, "an outage is not a correct refusal"
    assert metrics["provider_failures"] == 3
    assert all(o.actual == "provider_error" for o in outcomes)


def test_provider_errors_are_summarised_so_the_cause_is_visible():
    """'provider 9' says a call failed, not whether it was throttling, an
    exhausted quota or an outage - three problems needing three responses."""
    from evals.metrics import Report, compute_metrics, format_report

    cases = [case(id=f"a{i}", question="q", expect_status="answered", expect_text=("x",))
             for i in range(4)]
    outcomes = [
        score_case(cases[0], answer(failure_stage="provider",
                                    reason="quota exhausted: RESOURCE_EXHAUSTED daily limit")),
        score_case(cases[1], answer(failure_stage="provider",
                                    reason="quota exhausted: RESOURCE_EXHAUSTED daily limit")),
        score_case(cases[2], answer(failure_stage="provider", reason="returned 503 overloaded")),
        score_case(cases[3], answer(status=AnswerStatus.SUPPORTED, answer="x",
                                    evidence=(evidence("x"),))),
    ]
    metrics = compute_metrics(outcomes, cases)
    summary = metrics["provider_error_summary"]
    assert sum(summary.values()) == 3
    assert max(summary.values()) == 2, "identical errors collapse to one line with a count"

    text_out = format_report(Report(outcomes=tuple(outcomes), dataset="d", metrics=metrics))
    assert "provider errors" in text_out
    assert "RESOURCE_EXHAUSTED" in text_out

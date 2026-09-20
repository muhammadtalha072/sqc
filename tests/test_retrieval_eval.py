"""Tests for the retrieval-only eval mode.

The mode exists to make retrieval measurable without a model, so the failure
that matters is not a wrong number - it is a number that looks like more than
it is. A retrieval-only run reports one stage; if it emitted answer-quality
rates, a reader (or a CI gate) would take a run that called no model as
evidence the system answers correctly.

The other failure it guards against is drift: two definitions of "retrieval
hit" that disagree would make the cheap mode and the full run uncomparable,
which is the whole reason to have the cheap mode.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from evals.dataset import EvalCase, load_dataset, tenant_for
from evals.metrics import (
    RetrievalOutcome,
    RetrievalReport,
    check_retrieval,
    compute_retrieval_metrics,
    format_retrieval_report,
    score_case,
)
from evals.run import run_retrieval_only
from sqc.config import Settings
from sqc.core.answering.schema import AnswerResult, AnswerStatus
from sqc.core.ingestion.pipeline import ingest_bytes
from sqc.core.retrieval.types import Evidence
from sqc.db.engine import get_admin_engine

DIM = 1024

POLICY = b"""# Acme Data Handling Standard

## 1. Encryption

Customer data is encrypted at rest using AES-256. Keys are held in the cloud
key management service and rotated annually.

## 2. Retention

Customer records are retained for 90 days after account closure, unless a
legal hold applies.
"""


def case(**overrides) -> EvalCase:  # noqa: ANN003
    base = dict(id="c1", question="q", expect_status="answered", expect_text=("x",))
    return EvalCase(**{**base, **overrides})


def evidence(text_value: str) -> Evidence:
    return Evidence(
        evidence_id="E1", chunk_ids=(uuid.uuid4(),), document_id=uuid.uuid4(),
        filename="p.md", text=text_value, heading_path=("P",), page_start=1,
        page_end=1, effective_date=None, score=1.0, citation="p.md p1",
    )


def outcome(**overrides) -> RetrievalOutcome:  # noqa: ANN003
    base = dict(case_id="c1", question="q", category="general", hit=True)
    return RetrievalOutcome(**{**base, **overrides})


# ------------------------------------------------------- drift between scorers


@pytest.mark.parametrize(
    ("expect_text", "evidence_text"),
    [
        (("Covered Data",), "Covered Data includes social security numbers."),
        (("Covered Data",), "something entirely unrelated"),
        (("AES-256", "90 days"), "AES-256 is used. Records are kept 90 days."),
        (("AES-256", "90 days"), "AES-256 is used."),
        (("MiXeD CaSe",), "mixed case matching must stay case-insensitive"),
        ((), "refusal cases assert nothing about retrieval"),
    ],
)
def test_helper_agrees_with_the_full_scorer(expect_text, evidence_text):
    """The cheap mode and the full run must never disagree about what counts
    as a retrieval hit, or their numbers stop being comparable and the cheap
    mode becomes a second opinion rather than the same measurement."""
    entry = case(expect_text=expect_text, expect_status="answered" if expect_text else "refused")
    packed = (evidence(evidence_text),)

    hit, _missing = check_retrieval(entry, packed)
    scored = score_case(
        entry,
        AnswerResult(
            question="q",
            status=AnswerStatus.SUPPORTED if expect_text else AnswerStatus.REFUSED,
            answer="",
            evidence=packed,
        ),
    )
    assert hit == scored.retrieval_hit


def test_a_case_with_no_expectation_is_none_not_a_hit():
    """None and False are different states. Averaging a refusal case in as a
    hit would inflate recall with cases that asserted nothing."""
    hit, missing = check_retrieval(case(expect_text=()), (evidence("anything"),))
    assert hit is None
    assert missing == ()


def test_missing_strings_are_reported_individually():
    hit, missing = check_retrieval(
        case(expect_text=("present", "absent")), (evidence("present only"),)
    )
    assert hit is False
    assert missing == ("absent",)


# --------------------------------------------- the mode must not overclaim


def test_retrieval_metrics_omit_every_answer_quality_rate():
    """The property that keeps this mode honest. A run that called no model
    must not report coverage or a false answer rate - not even as zeros,
    which would read as a system that answers nothing dangerously well."""
    metrics = compute_retrieval_metrics([outcome(hit=True), outcome(case_id="c2", hit=False)])

    for forbidden in (
        "coverage",
        "false_answer_rate",
        "false_refusal_rate",
        "refusal_precision",
        "hallucination_rate",
        "citation_rate",
        "review_required_rate",
        "unsupported_claim_rate",
        "answered_without_evidence_rate",
    ):
        assert forbidden not in metrics, f"retrieval-only must not report {forbidden}"

    assert metrics["mode"] == "retrieval-only"
    assert metrics["retrieval_recall"] == 0.5
    assert metrics["retrieval_failure_rate"] == 0.5


def test_recall_is_none_rather_than_zero_when_nothing_was_scored():
    metrics = compute_retrieval_metrics([])
    assert metrics["retrieval_recall"] is None, (
        "zero would read as total retrieval failure; nothing was measured"
    )


def test_report_states_what_it_did_not_measure():
    report = RetrievalReport(
        outcomes=(outcome(hit=False, missing=("3/31/2017",)),),
        dataset="d", embedding_model="fake-hashing-v1", rerank_provider="none",
        metrics=compute_retrieval_metrics([outcome(hit=False, missing=("3/31/2017",))]),
    )
    rendered = format_retrieval_report(report)

    assert "NOT MEASURED" in rendered
    assert "llm=(not called)" in rendered
    assert "3/31/2017" in rendered, "a miss must name what was not found"


def test_skipped_cases_are_reported_not_silently_dropped():
    report = RetrievalReport(
        outcomes=(outcome(),), dataset="d", skipped=6,
        metrics=compute_retrieval_metrics([outcome()]),
    )
    assert "skipped" in format_retrieval_report(report)


# -------------------------------------------------- end to end, against Postgres


@pytest.fixture
def dataset_and_tenant(tmp_path):
    """A one-document corpus and a dataset written against it."""
    path = tmp_path / "retrieval-only-test.yaml"
    path.write_text(
        "name: retrieval-only-test\n"
        "document: retrieval_only_test.md\n"
        "documents:\n"
        "  - tests/data/retrieval_only_test.md\n"
        "cases:\n"
        "  - id: encryption\n"
        "    category: encryption\n"
        "    question: How is customer data encrypted at rest?\n"
        "    expect_status: answered\n"
        "    expect_text:\n"
        "      - AES-256\n"
        "  - id: nothing-to-assert\n"
        "    question: Are you SOC 2 certified?\n"
        "    expect_status: refused\n"
        "    forbid_literals:\n"
        "      - SOC 2\n"
    )
    repo_copy = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tests" / "data" / "retrieval_only_test.md"
    )
    repo_copy.write_bytes(POLICY)

    tenant_id = tenant_for("retrieval-only-test")
    with get_admin_engine().begin() as conn:
        conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :n) "
                 "ON CONFLICT (id) DO NOTHING"),
            {"id": tenant_id, "n": "retrieval-only-test"},
        )
    from sqc.providers.fake import HashingEmbedder

    ingest_bytes(
        tenant_id=tenant_id, raw=POLICY,
        filename="retrieval_only_test.md", embedder=HashingEmbedder(dimension=DIM),
    )
    yield str(path), tenant_id
    with get_admin_engine().begin() as conn:
        conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})
    repo_copy.unlink(missing_ok=True)


def settings_without_any_llm() -> Settings:
    """No key, and an llm_provider that would raise if anything built it."""
    return Settings(
        embedding_provider="fake", rerank_provider="none", embedding_dim=DIM,
        llm_provider="no-such-provider", anthropic_api_key="", gemini_api_key="",
        voyage_api_key="",
    )


def test_retrieval_only_runs_with_no_llm_provider_configured(dataset_and_tenant, monkeypatch):
    """The point of the mode: it must not build a model client. An
    unbuildable provider proves it, and so does refusing to let the runner
    reach build_llm_provider at all."""
    dataset_path, tenant_id = dataset_and_tenant

    import evals.run as run_module

    def explode(*_args, **_kwargs):
        raise AssertionError("retrieval-only must not build an LLM provider")

    monkeypatch.setattr(run_module, "build_llm_provider", explode)

    report = run_retrieval_only(tenant_id, dataset_path, settings=settings_without_any_llm())

    assert report.metrics["retrieval_recall"] == 1.0
    assert report.metrics["cases_with_expectations"] == 1
    assert report.skipped == 1, "the refusal case asserts nothing and is skipped"


def test_retrieval_only_touches_no_cassette(dataset_and_tenant, monkeypatch):
    """A cassette read or write would mean the mode depends on recorded model
    output, which is exactly the coupling it exists to remove."""
    dataset_path, tenant_id = dataset_and_tenant

    import evals.cassette as cassette_module

    def explode(*_args, **_kwargs):
        raise AssertionError("retrieval-only must not touch the cassette layer")

    monkeypatch.setattr(cassette_module.CassetteLLM, "__init__", explode)

    report = run_retrieval_only(tenant_id, dataset_path, settings=settings_without_any_llm())
    assert report.failed == 0


def test_retrieval_only_refuses_a_contaminated_tenant(dataset_and_tenant):
    """A tenant holding more than the dataset declares produces retrieval
    numbers that describe the tenant. The guard applies to this mode too."""
    dataset_path, tenant_id = dataset_and_tenant
    from sqc.providers.fake import HashingEmbedder

    ingest_bytes(
        tenant_id=tenant_id, raw=b"# Unrelated\n\nSome other policy entirely.",
        filename="contaminant.md", embedder=HashingEmbedder(dimension=DIM),
    )
    with pytest.raises(SystemExit, match="corpus mismatch"):
        run_retrieval_only(tenant_id, dataset_path, settings=settings_without_any_llm())


def test_retrieval_only_reports_an_absent_corpus_rather_than_crashing(tmp_path):
    path = tmp_path / "ghost.yaml"
    path.write_text(
        "name: ghost-retrieval\ndocument: nope.pdf\ndocuments:\n"
        "  - tests/data/no-such-*.pdf\n"
        "cases:\n  - id: a\n    question: q?\n    expect_status: refused\n"
    )
    with pytest.raises(SystemExit, match="cannot score"):
        run_retrieval_only(tenant_for("ghost-retrieval"), str(path))


def test_shipped_datasets_have_cases_the_retrieval_mode_can_score():
    """A dataset whose cases all lack expect_text would report recall over
    nothing while looking like a clean run."""
    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    for path in sorted((repo / "evals" / "datasets").glob("*.yaml")):
        dataset = load_dataset(path)
        scorable = [c for c in dataset.cases if c.expect_text]
        assert scorable, f"{path.name} has nothing for --retrieval-only to score"

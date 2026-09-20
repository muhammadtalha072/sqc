"""Run a golden dataset against the pipeline.

    python -m evals.run --tenant <uuid> --dataset evals/datasets/depaul-isp-v1.yaml
    python -m evals.run --tenant <uuid> --record          # call the model, cache replies
    python -m evals.run --tenant <uuid> --replay          # cache only, free and offline
    python -m evals.run --tenant <uuid> --retrieval-only  # retrieval alone, no model
    python -m evals.run --tenant <uuid> --baseline evals/results/baseline.json

Default mode is record: call the model for prompts not yet cached, replay
the rest. That makes the first run cost a handful of requests and every
rerun free.

--retrieval-only scores the retrieval stage by itself. Retrieval ground
truth is a substring assertion against the evidence pack, so it never
depended on the answering model: the mode builds no LLM provider, touches
no cassette, needs no API key and spends no quota. It reports retrieval
metrics only - a run that called no model has nothing to say about answer
quality, and reporting a zero there would read as a system that answers
nothing rather than one that was not asked to answer.

--baseline compares against a saved run and exits non-zero on regression.
The gate is deliberately asymmetric: a rise in false answers or
hallucinations fails immediately, while a fall in coverage is reported but
tolerated, because being wrong less often is worth answering less often.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import uuid
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqc.config import get_settings  # noqa: E402
from sqc.core.answering.pipeline import answer_question  # noqa: E402
from sqc.core.answering.prompt import PROMPT_VERSION  # noqa: E402
from sqc.core.retrieval.pipeline import retrieve  # noqa: E402
from sqc.db.engine import tenant_session  # noqa: E402
from sqc.db.repository import list_documents  # noqa: E402
from sqc.providers.registry import (  # noqa: E402
    build_embedding_provider,
    build_entailment_provider,
    build_llm_provider,
    build_rerank_provider,
)

from evals.cassette import CassetteLLM  # noqa: E402
from evals.dataset import (  # noqa: E402
    DatasetError,
    load_dataset,
    resolve_documents,
    tenant_for,
)
from evals.metrics import (  # noqa: E402
    Report,
    RetrievalOutcome,
    RetrievalReport,
    check_retrieval,
    compute_metrics,
    compute_retrieval_metrics,
    format_report,
    format_retrieval_report,
    score_case,
)

def _configured_model(settings) -> str:  # noqa: ANN001
    """Model id the provider would actually use, mirroring the registry."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        from sqc.providers.gemini import DEFAULT_MODEL

        return (
            settings.llm_model
            if "gemini" in settings.llm_model.lower()
            else DEFAULT_MODEL
        )
    if provider == "anthropic":
        return settings.llm_model
    return provider


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_DATASET = "evals/datasets/depaul-isp-v1.yaml"
CASSETTE_DIR = "evals/cassettes"
RESULTS_DIR = "evals/results"

# Metrics where an increase is a regression, with the tolerance allowed.
# Zero tolerance on the two that matter: answering a question the evidence
# cannot support, and stating something the evidence does not contain.
REGRESSION_GATES = {
    "false_answer_rate": 0.0,
    "hallucination_rate": 0.0,
    # An answerable case whose expected evidence never arrived, answered
    # anyway. Gated with the other two that matter because nothing else
    # reports it: coverage counts it as answered, and hallucination misses it,
    # since the claims are grounded in whatever adjacent text did arrive.
    "answered_without_evidence_rate": 0.0,
    "false_refusal_rate": 0.05,
}


def assert_corpus_matches(tenant_id: uuid.UUID, dataset, dataset_path: str) -> None:  # noqa: ANN001
    """Refuse to score a tenant that does not hold exactly the declared corpus.

    Shared by every mode. A contaminated tenant produces plausible-looking
    retrieval numbers that describe the tenant's contents rather than the
    system, which is the failure this guard exists to prevent - and a
    retrieval-only run is if anything more exposed to it, since retrieval
    recall is the only thing it reports.
    """
    try:
        expected = {p.name for p in resolve_documents(dataset, ROOT)}
    except DatasetError as exc:
        # Third-party policy PDFs are deliberately uncommitted, so an absent
        # corpus is a normal state. Reporting it beats a traceback, and it
        # must never be mistaken for a dataset that scored zero.
        raise SystemExit(f"cannot score '{dataset.name}': {exc}") from None

    with tenant_session(tenant_id) as session:
        present = {d.filename for d in list_documents(session)}

    missing = expected - present
    extra = present - expected
    if missing or extra:
        raise SystemExit(
            f"corpus mismatch for '{dataset.name}'.\n"
            + (f"  missing: {', '.join(sorted(missing))}\n" if missing else "")
            + (f"  unexpected: {', '.join(sorted(extra))}\n" if extra else "")
            + f"Run: python scripts/eval_setup.py --dataset {dataset_path}"
        )


def run_dataset(
    tenant_id: uuid.UUID,
    dataset_path: str,
    mode: str,
    settings=None,  # noqa: ANN001
) -> Report:
    settings = settings or get_settings()
    dataset = load_dataset(dataset_path)
    assert_corpus_matches(tenant_id, dataset, dataset_path)

    inner = None if mode == "replay" else build_llm_provider(settings)
    # The model name is part of the cache key, so replay needs it even
    # though it builds no provider.
    llm = CassetteLLM(inner, CASSETTE_DIR, mode=mode, model=_configured_model(settings))
    embedder = build_embedding_provider(settings)
    reranker = build_rerank_provider(settings)
    entailment = build_entailment_provider(settings)

    outcomes = []
    for index, case in enumerate(dataset.cases, start=1):
        print(f"  [{index}/{len(dataset)}] {case.id}", file=sys.stderr, flush=True)
        result = answer_question(
            tenant_id=tenant_id,
            question=case.question,
            llm=llm,
            embedder=embedder,
            reranker=reranker,
            settings=settings,
            entailment=entailment,
        )
        outcomes.append(score_case(case, result))

    print(
        f"  cassette: {llm.hits} replayed, {llm.recorded} recorded, "
        f"{llm.misses - llm.recorded} missing",
        file=sys.stderr,
        flush=True,
    )
    return Report(
        outcomes=tuple(outcomes),
        dataset=f"{dataset.name} ({len(dataset)} cases)",
        model=llm.model,
        embedding_model=getattr(embedder, "model", "?"),
        rerank_provider=settings.rerank_provider,
        prompt_version=f"{PROMPT_VERSION}+entail:{settings.entailment_provider}",
        metrics=compute_metrics(list(outcomes), list(dataset.cases)),
    )


def run_retrieval_only(
    tenant_id: uuid.UUID,
    dataset_path: str,
    settings=None,  # noqa: ANN001
) -> RetrievalReport:
    """Score retrieval alone: no model call, no cassette, no API quota.

    Retrieval ground truth is a substring assertion against the evidence pack,
    so it never needed the answering model to be measured. Separating it means
    a retrieval change can be evaluated on a free, deterministic, offline run
    instead of re-spending a daily free-tier allowance - and means retrieval
    stays measurable while the end-to-end path is blocked for any reason.

    Cases with no expect_text assert nothing about retrieval and are skipped,
    not scored as hits, which would inflate recall with refusal cases.
    """
    settings = settings or get_settings()
    dataset = load_dataset(dataset_path)
    assert_corpus_matches(tenant_id, dataset, dataset_path)

    # No LLM provider is built. That is the point of the mode, and it is why
    # this path works with no key configured at all.
    embedder = build_embedding_provider(settings)
    reranker = build_rerank_provider(settings)

    outcomes: list[RetrievalOutcome] = []
    skipped = 0
    for index, case in enumerate(dataset.cases, start=1):
        if not case.expect_text:
            skipped += 1
            continue
        print(f"  [{index}/{len(dataset)}] {case.id}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        result = retrieve(
            tenant_id=tenant_id,
            question=case.question,
            embedder=embedder,
            reranker=reranker,
            settings=settings,
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        hit, missing = check_retrieval(case, result.evidence)
        outcomes.append(
            RetrievalOutcome(
                case_id=case.id,
                question=case.question,
                category=case.category,
                hit=bool(hit),
                missing=missing,
                evidence_items=len(result.evidence),
                candidates=len(result.candidates),
                below_floor=result.below_floor,
                refusal_reason=result.refusal_reason or "",
                latency_ms=elapsed,
            )
        )

    print(f"  retrieval only: 0 model calls, {skipped} case(s) skipped",
          file=sys.stderr, flush=True)
    return RetrievalReport(
        outcomes=tuple(outcomes),
        dataset=f"{dataset.name} ({len(dataset)} cases)",
        embedding_model=getattr(embedder, "model", "?"),
        rerank_provider=settings.rerank_provider,
        skipped=skipped,
        metrics=compute_retrieval_metrics(outcomes),
    )


def check_regression(current: dict, baseline: dict) -> list[str]:
    """Compare against a saved run. Returns the regressions found."""
    problems: list[str] = []
    for metric, tolerance in REGRESSION_GATES.items():
        now, before = current.get(metric), baseline.get(metric)
        if now is None or before is None:
            continue
        if now > before + tolerance:
            problems.append(
                f"{metric} rose from {before:.1%} to {now:.1%} "
                f"(tolerance {tolerance:.0%})"
            )
    retrieval_now = current.get("retrieval_recall")
    retrieval_before = baseline.get("retrieval_recall")
    if retrieval_now is not None and retrieval_before is not None:
        if retrieval_now < retrieval_before - 0.05:
            problems.append(
                f"retrieval_recall fell from {retrieval_before:.1%} to {retrieval_now:.1%}"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default=None,
                        help="default: derived from the dataset name")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--replay", action="store_true", help="cache only, no API calls")
    parser.add_argument("--refresh", action="store_true", help="re-record every response")
    parser.add_argument("--attempts", type=int, default=3,
                        help="provider retry budget per call (default 3)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="score retrieval alone: no model call, no cassette, no quota")
    parser.add_argument("--baseline", default=None, help="compare against a saved run")
    parser.add_argument("--save", default=None, help="write the report JSON here")
    args = parser.parse_args()

    if args.retrieval_only and (args.replay or args.refresh):
        # Both flags describe what to do with recorded model responses, and
        # this mode calls no model. Accepting them silently would let a run
        # read as though cassettes were involved in producing its numbers.
        parser.error("--retrieval-only makes no model calls; --replay/--refresh do not apply")

    mode = "replay" if args.replay else "refresh" if args.refresh else "record"
    # A measurement run should wait rather than record an outage as a
    # refusal. Already-recorded cases are skipped, so re-running only
    # retries what is still missing.
    # Backoff is exponential, so a generous budget is expensive: 8 attempts
    # at a 2s base sleeps roughly 254 seconds per failing call, which cost
    # an hour on a run whose failures were persistent rather than transient.
    # Retries help a rate limit; they do nothing for a request the API
    # rejects every time.
    settings = get_settings().model_copy(
        update={"llm_max_attempts": args.attempts, "llm_base_delay": 1.0}
    )
    dataset_name = load_dataset(args.dataset).name
    tenant_id = uuid.UUID(args.tenant) if args.tenant else tenant_for(dataset_name)

    if args.retrieval_only:
        retrieval_report = run_retrieval_only(tenant_id, args.dataset, settings=settings)
        print()
        print(format_retrieval_report(retrieval_report))
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "mode": "retrieval-only",
            "dataset": retrieval_report.dataset,
            "model": None,
            "embedding_model": retrieval_report.embedding_model,
            "rerank_provider": retrieval_report.rerank_provider,
            "skipped_cases": retrieval_report.skipped,
            "metrics": retrieval_report.metrics,
            "cases": [o.as_dict() for o in retrieval_report.outcomes],
        }
        # A separate default file. Writing retrieval-only numbers over
        # latest.json would leave a report carrying no answer metrics where
        # a full run's report is expected.
        default_destination = f"{RESULTS_DIR}/latest-retrieval.json"
        metrics, failed = retrieval_report.metrics, retrieval_report.failed
    else:
        report = run_dataset(tenant_id, args.dataset, mode, settings=settings)
        print()
        print(format_report(report))
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": report.dataset,
            "model": report.model,
            "embedding_model": report.embedding_model,
            "rerank_provider": report.rerank_provider,
            "prompt_version": report.prompt_version,
            "metrics": report.metrics,
            "cases": [o.as_dict() for o in report.outcomes],
        }
        default_destination = f"{RESULTS_DIR}/latest.json"
        metrics, failed = report.metrics, report.failed

    destination = args.save
    if destination is None:
        pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        destination = default_destination
    pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(destination).write_text(json.dumps(payload, indent=2))
    print(f"\nreport written to {destination}")

    exit_code = 0 if failed == 0 else 1

    if args.baseline:
        baseline_path = pathlib.Path(args.baseline)
        if not baseline_path.exists():
            print(f"\nno baseline at {baseline_path}; treating this run as the baseline")
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(payload, indent=2))
        else:
            baseline = json.loads(baseline_path.read_text()).get("metrics", {})
            # check_regression skips any gate absent from either side, so a
            # retrieval-only run compares on retrieval_recall alone rather
            # than reading a full baseline's answer metrics as improvements.
            problems = check_regression(metrics, baseline)
            print()
            if problems:
                print("REGRESSION")
                for problem in problems:
                    print(f"  - {problem}")
                exit_code = 2
            else:
                print("no regression against baseline")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

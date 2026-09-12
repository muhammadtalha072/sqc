"""Answer a questionnaire question end to end.

    python scripts/answer.py --tenant <uuid> "Do you enforce MFA?"
    python scripts/answer.py --tenant <uuid> --audit "Is data encrypted at rest?"

With SQC_LLM_PROVIDER=fake the answering model is a scripted stub that
refuses everything it was not given a response for, so this is only useful
offline for exercising the plumbing. Set SQC_LLM_PROVIDER=anthropic with a
key for real answers.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sqc.config import get_settings  # noqa: E402
from sqc.core.answering.pipeline import answer_question  # noqa: E402
from sqc.core.answering.schema import AnswerStatus  # noqa: E402
from sqc.providers.registry import (  # noqa: E402
    build_embedding_provider,
    build_llm_provider,
    build_rerank_provider,
)

MARKS = {
    AnswerStatus.SUPPORTED: "SUPPORTED",
    AnswerStatus.REVIEW_REQUIRED: "REVIEW REQUIRED",
    AnswerStatus.REFUSED: "REFUSED",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--audit", action="store_true", help="print the audit record as JSON")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.top_k:
        settings = settings.model_copy(update={"evidence_top_k": args.top_k})

    result = answer_question(
        tenant_id=uuid.UUID(args.tenant),
        question=args.question,
        llm=build_llm_provider(settings),
        embedder=build_embedding_provider(settings),
        reranker=build_rerank_provider(settings),
        settings=settings,
    )

    print(f"[{MARKS[result.status]}]  {result.question}\n")
    if result.answer:
        print(f"{result.answer}\n")
    print(f"reason: {result.reason}\n")

    if result.claims:
        print("claims:")
        for claim in result.claims:
            mark = "ok " if claim.supported else "DROPPED"
            handles = ",".join(c.evidence_id for c in claim.citations) or "-"
            print(f"  [{mark}] {claim.text}")
            print(f"          cites {handles}"
                  + (f"  ({claim.problem})" if claim.problem else ""))
        print()

    if result.citations:
        print("citations:")
        for citation in result.citations:
            where = " > ".join(citation.heading_path) or "(no section)"
            pages = f" p{citation.page_start}" if citation.page_start else ""
            print(f"  {citation.evidence_id}: {citation.filename}{pages} - {where}")
        print()

    if result.validation_errors:
        print("validation:")
        for error in result.validation_errors:
            print(f"  - {error}")
        print()

    print(f"signals: {result.retrieval_signals.as_dict()}")
    print(f"model: {result.llm_model}  tokens in/out: "
          f"{result.input_tokens}/{result.output_tokens}  {result.latency_ms}ms")

    if args.audit:
        print("\naudit record:")
        print(json.dumps(result.to_audit_record(), indent=2, default=str))

    return 0 if result.status is AnswerStatus.SUPPORTED else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0) from None

"""Run a retrieval query and show what came back.

    python scripts/search.py --tenant <uuid> "Do you enforce MFA?"
    python scripts/search.py --tenant <uuid> --explain "Where is data stored?"

--explain prints per-retriever scores and ranks, which is how you tell a
lexical hit from a dense one when tuning.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sqc.config import get_settings  # noqa: E402
from sqc.core.retrieval.pipeline import retrieve  # noqa: E402
from sqc.core.retrieval.query import expand_query  # noqa: E402
from sqc.providers.registry import (  # noqa: E402
    build_embedding_provider,
    build_rerank_provider,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="the questionnaire question")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--explain", action="store_true", help="show per-retriever scores")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.top_k:
        settings = settings.model_copy(update={"evidence_top_k": args.top_k})

    expanded, terms = expand_query(args.question)
    if args.explain:
        print(f"terms    : {' '.join(terms)}")
        print(f"expanded : {expanded}\n")

    result = retrieve(
        tenant_id=uuid.UUID(args.tenant),
        question=args.question,
        embedder=build_embedding_provider(settings),
        reranker=None if args.no_rerank else build_rerank_provider(settings),
        settings=settings,
    )

    if not result.has_evidence:
        print(f"NO EVIDENCE: {result.refusal_reason}")
        print(f"signals: {result.signals.as_dict()}")
        return 0

    for item in result.evidence:
        print(f"[{item.evidence_id}] score={item.score:.4f}  {item.citation}")
        body = item.text.replace("\n", " ")
        print(f"     {body[:220]}{'...' if len(body) > 220 else ''}\n")

    if args.explain:
        print("candidates (fused order):")
        for candidate in result.candidates[:12]:
            lex = f"{candidate.lexical_score:.4f}@{candidate.lexical_rank}" if candidate.lexical_score is not None else "-"
            vec = f"{candidate.vector_score:.4f}@{candidate.vector_rank}" if candidate.vector_score is not None else "-"
            rer = f"{candidate.rerank_score:.4f}" if candidate.rerank_score is not None else "-"
            print(f"  rrf={candidate.fusion_score:.6f} lex={lex:>14} vec={vec:>14} "
                  f"rerank={rer:>8}  {candidate.citation[:70]}")
        print()

    print(f"signals: {result.signals.as_dict()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0) from None

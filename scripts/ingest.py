"""Ingest documents into the knowledge base.

    python scripts/ingest.py --tenant <uuid> tests/data/policy.pdf
    python scripts/ingest.py --tenant <uuid> --list
    python scripts/ingest.py --tenant <uuid> --show <document-uuid>

The embedding provider comes from .env. With SQC_EMBEDDING_PROVIDER=fake
this runs offline, which is enough to verify the pipeline end to end;
switch to voyage when measuring real retrieval quality.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402

from sqc.config import get_settings  # noqa: E402
from sqc.core.ingestion.pipeline import IngestionError, ingest_file  # noqa: E402
from sqc.db.engine import tenant_session  # noqa: E402
from sqc.db.repository import list_documents  # noqa: E402
from sqc.providers.registry import build_embedding_provider  # noqa: E402


def show_documents(tenant_id: uuid.UUID) -> None:
    with tenant_session(tenant_id) as session:
        rows = list_documents(session)
    if not rows:
        print("no documents for this tenant")
        return
    print(f"{'document id':38} {'status':9} {'chunks':>6}  {'pages':>5}  {'effective':10}  file")
    for row in rows:
        print(
            f"{str(row.id):38} {row.status:9} {row.chunk_count:>6}  "
            f"{str(row.page_count or '-'):>5}  {str(row.effective_date or '-'):10}  {row.filename}"
        )


def show_chunks(tenant_id: uuid.UUID, document_id: uuid.UUID, limit: int) -> None:
    with tenant_session(tenant_id) as session:
        rows = session.execute(
            text(
                """
                SELECT chunk_index, heading_text, page_start, page_end, token_count,
                       injection_flags, (embedding IS NOT NULL), left(text, 110)
                FROM chunks WHERE document_id = :id
                ORDER BY chunk_index LIMIT :limit
                """
            ),
            {"id": document_id, "limit": limit},
        ).all()
    for index, heading, start, end, tokens, flags, embedded, preview in rows:
        pages = f"p{start}" + (f"-{end}" if end and end != start else "") if start else "p-"
        print(f"[{index:>3}] {pages:>8} {tokens:>4}tok vec={'y' if embedded else 'n'} "
              f"{'FLAGS=' + ','.join(flags) if flags else ''}")
        print(f"      {heading or '(no heading)'}")
        print(f"      {preview.strip()}...")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="files to ingest")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--doc-type", default="unknown",
                        help="policy | soc2 | dpa | questionnaire | other")
    parser.add_argument("--replace", action="store_true",
                        help="re-ingest a file already stored under the same hash")
    parser.add_argument("--no-embed", action="store_true",
                        help="store text only; keyword retrieval still works")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show", default=None, metavar="DOC_ID")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    tenant_id = uuid.UUID(args.tenant)

    if args.list:
        show_documents(tenant_id)
        return 0
    if args.show:
        show_chunks(tenant_id, uuid.UUID(args.show), args.limit)
        return 0
    if not args.paths:
        parser.error("give at least one file, or use --list / --show")

    settings = get_settings()
    embedder = None if args.no_embed else build_embedding_provider(settings)
    if embedder is not None:
        print(f"embedding with {settings.embedding_provider}:{embedder.model} "
              f"({embedder.dimension}d)")

    failures = 0
    for path in args.paths:
        try:
            result = ingest_file(
                tenant_id=tenant_id, path=path, embedder=embedder,
                doc_type=args.doc_type, replace_existing=args.replace,
            )
        except IngestionError as exc:
            print(f"FAILED  {path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if result.skipped_duplicate:
            print(f"SKIP    {result.filename}: already stored as {result.document_id} "
                  "(use --replace to re-ingest)")
            continue

        print(f"OK      {result.filename}: {result.chunk_count} chunks -> {result.document_id}")
        if result.parsed is not None:
            print(f"        pages={result.parsed.page_count} "
                  f"effective_date={result.parsed.effective_date} "
                  f"version={result.parsed.version_label}")
        if result.injection_flagged:
            print(f"        {result.injection_flagged} chunk(s) flagged for injection-like text")
        for warning in result.warnings:
            print(f"        warning: {warning}")

    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Piping into head/less closes stdout early. Exit quietly instead of
        # dumping a traceback over the user's terminal.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0) from None

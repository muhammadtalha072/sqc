"""Provision an isolated tenant per evaluation dataset and ingest its corpus.

    python scripts/eval_setup.py                                  # every dataset
    python scripts/eval_setup.py --dataset evals/datasets/foo.yaml

Each dataset gets its own tenant, derived from its name, containing exactly
the documents it declares and nothing else. Documents already present that
the dataset does not declare are removed.

The previous version took a --with-real flag and ingested every document on
disk into one shared tenant. Twelve cases written against four small fixture
policies were then scored while competing with 164 chunks of five unrelated
university policies, and the resulting retrieval recall described the setup
script rather than the system.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from sqc.config import get_settings  # noqa: E402
from sqc.core.ingestion.pipeline import IngestionError, ingest_file  # noqa: E402
from sqc.db.engine import get_admin_engine, tenant_session  # noqa: E402
from sqc.db.repository import delete_document, ensure_tenant, list_documents  # noqa: E402
from sqc.providers.registry import build_embedding_provider  # noqa: E402

from evals.dataset import (  # noqa: E402
    DatasetError,
    load_dataset,
    resolve_documents,
    tenant_for,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "evals" / "datasets"


def provision(dataset_path: pathlib.Path) -> int:
    dataset = load_dataset(dataset_path)
    tenant_id = tenant_for(dataset.name)
    try:
        paths = resolve_documents(dataset, ROOT)
    except DatasetError as exc:
        # Third-party policy PDFs are deliberately not committed, so a
        # dataset whose corpus is absent is skipped rather than fatal. The
        # runner still refuses to score that dataset.
        print(f"\n{dataset.name}  SKIPPED: {exc}", file=sys.stderr)
        return 0
    expected = {p.name for p in paths}

    with get_admin_engine().begin() as conn:
        ensure_tenant(conn, tenant_id, f"eval:{dataset.name}")

    # Anything not declared is removed, so a tenant reused from an earlier
    # contaminated run cannot quietly keep scoring against extra documents.
    removed = 0
    with tenant_session(tenant_id) as session:
        for document in list_documents(session):
            if document.filename not in expected:
                delete_document(session, document.id)
                removed += 1

    embedder = build_embedding_provider(get_settings())
    failures = 0
    print(f"\n{dataset.name}  tenant={tenant_id}")
    if removed:
        print(f"  removed {removed} document(s) not declared by this dataset")

    for path in paths:
        try:
            result = ingest_file(
                tenant_id=tenant_id, path=str(path), embedder=embedder,
                doc_type="policy", replace_existing=True,
            )
        except IngestionError as exc:
            print(f"  FAILED  {path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        flagged = (
            f"  ({result.injection_flagged} chunk(s) flagged for injection)"
            if result.injection_flagged else ""
        )
        print(f"  OK      {path.name}: {result.chunk_count} chunks{flagged}")

    with tenant_session(tenant_id) as session:
        present = {d.filename for d in list_documents(session)}
        chunks = session.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    if present != expected:
        print(f"  ISOLATION FAILED: tenant holds {sorted(present)}, "
              f"dataset declares {sorted(expected)}", file=sys.stderr)
        failures += 1
    else:
        print(f"  isolated: {len(present)} document(s), {chunks} chunks")
    print(f"  export TENANT={tenant_id}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="one dataset; default is all")
    args = parser.parse_args()

    paths = (
        [pathlib.Path(args.dataset)] if args.dataset else sorted(DATASET_DIR.glob("*.yaml"))
    )
    if not paths:
        print(f"no datasets found in {DATASET_DIR}", file=sys.stderr)
        return 1
    return 1 if sum(provision(p) for p in paths) else 0


if __name__ == "__main__":
    raise SystemExit(main())

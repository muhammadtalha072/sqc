"""Ground-truth invariants for the multi-document eval corpus.

university-isp-multi-v1 scores cases written against one university's policy
inside a tenant holding five. That is the point - a real customer has many
overlapping documents and the hard problem is answering from the right one -
but it puts two obligations on the golden file that a single-document dataset
never had.

Both were checked by hand when the dataset was written. They are asserted here
so a later edit cannot quietly break them:

1. Every expect_text string must be unique to the named document. A string
   that also appears in another university's policy lets a case pass by
   retrieving the wrong document, which is the exact failure the corpus exists
   to detect.

2. No forbid_literal may appear in the named document. A refusal case whose
   forbidden string is in the very policy it asks about is unpassable, and the
   failure would look like a model defect. This caught a real mistake: "AWS"
   as a forbidden literal matches inside the word "laws", which DePaul's own
   text contains.

A third test pins the reason the dataset exists at all - the corpus has to be
large enough that retrieval can fail.

The five policies are third-party PDFs and deliberately uncommitted, so every
test here skips rather than fails when the corpus is absent.
"""

from __future__ import annotations

import pathlib
import uuid

import pytest
from sqlalchemy import text

from evals.dataset import load_dataset, resolve_documents
from sqc.config import Settings
from sqc.core.ingestion.pipeline import ingest_file
from sqc.core.retrieval.pipeline import retrieve
from sqc.db.engine import get_admin_engine, tenant_session
from sqc.providers.fake import HashingEmbedder

REPO = pathlib.Path(__file__).resolve().parents[1]
DATASET = REPO / "evals" / "datasets" / "university-isp-multi-v1.yaml"
DIM = 1024

# Retrieval returning a large slice of the corpus cannot fail in an
# interesting way. Measured on the single-document datasets the evidence pack
# was 34-40% of all chunks and vector-only retrieval scored 100%, so the
# ablation had no power to tell a good fusion change from deleting a
# retriever. On this corpus it is about 3%.
MAX_EVIDENCE_SHARE = 0.10


def _dataset():  # noqa: ANN202
    if not DATASET.exists():
        pytest.skip(f"{DATASET.name} not present")
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def corpus():  # noqa: ANN201
    """The declared five-document corpus in a throwaway tenant."""
    dataset = _dataset()
    try:
        paths = resolve_documents(dataset, REPO)
    except Exception as exc:  # DatasetError: third-party PDFs are uncommitted
        pytest.skip(f"corpus unavailable: {exc}")

    tenant_id = uuid.uuid4()
    with get_admin_engine().begin() as conn:
        conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :n)"),
            {"id": tenant_id, "n": "multi-doc-corpus-test"},
        )
    embedder = HashingEmbedder(dimension=DIM)
    try:
        for path in paths:
            ingest_file(tenant_id=tenant_id, path=str(path), embedder=embedder,
                        doc_type="policy", replace_existing=True)
        with tenant_session(tenant_id) as session:
            rows = session.execute(
                text("SELECT d.filename, c.text FROM chunks c "
                     "JOIN documents d ON d.id = c.document_id")
            ).all()
        yield tenant_id, dataset, rows
    finally:
        with get_admin_engine().begin() as conn:
            conn.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})


def _documents_containing(rows, needle: str) -> set[str]:  # noqa: ANN001
    return {
        filename
        for filename, body in rows
        if needle.lower() in (body or "").lower()
    }


def test_every_expect_text_is_unique_to_the_named_document(corpus):
    """Otherwise a case passes by retrieving another organisation's text."""
    _tenant_id, dataset, rows = corpus
    named = dataset.document

    problems = []
    for case in dataset.cases:
        for needle in case.expect_text:
            found = _documents_containing(rows, needle)
            if named not in found:
                problems.append(f"{case.id}: {needle!r} is not in {named}")
            elif found - {named}:
                problems.append(
                    f"{case.id}: {needle!r} also appears in {sorted(found - {named})} - "
                    "the case could pass on the wrong document"
                )
    assert not problems, "\n".join(problems)


def test_no_forbid_literal_appears_in_the_named_document(corpus):
    """A refusal case whose forbidden string is in the policy it asks about
    cannot be passed, and the failure would be blamed on the model."""
    _tenant_id, dataset, rows = corpus
    named = dataset.document

    problems = [
        f"{case.id}: forbidden literal {needle!r} appears in {named} itself"
        for case in dataset.cases
        for needle in case.forbid_literals
        if named in _documents_containing(rows, needle)
    ]
    assert not problems, "\n".join(problems)


def test_unanswerable_cases_have_something_to_resist(corpus):
    """The corpus earns its cost through the cases where a wrong answer is
    retrievable. At least some refusal cases must have their forbidden text
    present in another document, or this is just a bigger easy corpus."""
    _tenant_id, dataset, rows = corpus
    named = dataset.document

    traps = {
        case.id
        for case in dataset.cases
        if not case.answerable
        for needle in case.forbid_literals
        if _documents_containing(rows, needle) - {named}
    }
    assert len(traps) >= 3, (
        f"only {len(traps)} refusal case(s) have tempting evidence elsewhere in the "
        "tenant; on a single-document corpus refusing was free because the text did "
        "not exist at all"
    )


def test_corpus_is_large_enough_for_retrieval_to_fail(corpus):
    """The reason this dataset exists. If the evidence pack is a large share
    of the corpus, recall measures the corpus size rather than the system."""
    tenant_id, dataset, rows = corpus
    total_chunks = len(rows)
    assert total_chunks >= 100, f"corpus is only {total_chunks} chunks"

    settings = Settings(
        embedding_provider="fake", rerank_provider="none", llm_provider="fake",
        embedding_dim=DIM, candidates_per_retriever=40, evidence_top_k=6,
    )
    embedder = HashingEmbedder(dimension=DIM)
    shares = []
    for case in dataset.cases:
        result = retrieve(tenant_id=tenant_id, question=case.question,
                          embedder=embedder, reranker=None, settings=settings)
        packed = sum(len(item.chunk_ids) for item in result.evidence)
        shares.append(packed / total_chunks)

    mean_share = sum(shares) / len(shares)
    assert mean_share <= MAX_EVIDENCE_SHARE, (
        f"evidence pack averages {mean_share:.0%} of the corpus; above "
        f"{MAX_EVIDENCE_SHARE:.0%} retrieval is returning a slice of the corpus "
        "rather than selecting from it"
    )

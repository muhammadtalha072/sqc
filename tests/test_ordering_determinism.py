"""Regression tests for the Step 8b ordering defect.

The defect: evidence order tie-broke on `chunks.id`, which is
`gen_random_uuid()`. Two ingests of byte-identical files produced different
orders among tied rows, which changed the evidence pack, which changed the
prompt, which orphaned every recorded cassette. Measured before the fix:
15 of 24 prompt keys changed across two consecutive ingests with no code
change at all.

Ties are not an edge case here. The hashing embedder returns exactly equal
cosine similarity for many chunks, and RRF assigns 1/(k+rank), which is the
same value for any two chunks holding the same rank in their one retriever.
The tie-break therefore decides real evidence order most of the time.

What the fix does and does not buy is asserted below rather than assumed:
it makes ordering total and deterministic for a given database state, and
across re-ingests of a single-document corpus. It does not make a
multi-document corpus stable across re-ingests, because `documents.id` is
itself regenerated per ingest.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from sqc.config import Settings
from sqc.core.ingestion.pipeline import ingest_bytes
from sqc.core.retrieval.pipeline import retrieve
from sqc.core.retrieval.search import lexical_search, reciprocal_rank_fusion, vector_search
from sqc.core.retrieval.types import Candidate
from sqc.db.engine import get_admin_engine, tenant_session
from sqc.db.repository import format_vector
from sqc.providers.fake import HashingEmbedder

DIM = 1024

# One document, several sections. Deliberately repetitive: every chunk carries
# the document title in its heading, so a question made of title words matches
# the whole document at nearly equal rank - the shape that made tie-breaking
# decide the evidence pack in the first place.
SINGLE_DOC = b"""# Information Security Policy

## Definitions

Covered data includes social security numbers and financial account numbers.

## Responsibilities

The Responsible Officer for this information security policy is the Vice
President for Information Services.

## Reporting

Report a suspected breach to the Director of Information Security.

## Retention

Records are retained for ninety days after account closure.
"""

DOC_A = b"""# Acme Access Control Policy

## Authentication

Multi-factor authentication is required for all employee accounts.

## Exceptions

Break-glass service accounts are exempt from multi-factor authentication.
"""

DOC_B = b"""# Acme Data Handling Standard

## Encryption

Customer data is encrypted at rest using AES-256.

## Backups

Backups run every four hours.
"""

QUESTIONS = (
    "What does this information security policy say?",
    "Who is responsible for information security?",
    "What is required for authentication?",
    "How is customer data handled?",
)


def make_settings(**overrides) -> Settings:  # noqa: ANN003
    base = {
        "embedding_provider": "fake",
        "rerank_provider": "none",
        "llm_provider": "fake",
        "embedding_dim": DIM,
        "candidates_per_retriever": 40,
        "evidence_top_k": 6,
    }
    return Settings(**{**base, **overrides})


@pytest.fixture(scope="module")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=DIM)


def _new_tenant(name: str) -> uuid.UUID:
    tenant_id = uuid.uuid4()
    with get_admin_engine().begin() as conn:
        conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :n)"),
            {"id": tenant_id, "n": name},
        )
    return tenant_id


def _drop(*tenant_ids: uuid.UUID) -> None:
    with get_admin_engine().begin() as conn:
        conn.execute(text("DELETE FROM tenants WHERE id = ANY(:ids)"), {"ids": list(tenant_ids)})


def _texts(result) -> list[str]:  # noqa: ANN001
    """Evidence identified by content, which is the only thing stable across
    ingests. Comparing chunk_id or document_id would compare a fresh random
    value against another fresh random value and assert nothing."""
    return [e.text for e in result.evidence]


@pytest.fixture(scope="module")
def single_doc_twice(embedder):  # noqa: ANN001
    """The same one-document corpus ingested into two separate tenants.

    Separate ingests mean separate random document ids and chunk ids, which is
    exactly the condition that used to reorder the evidence.
    """
    first, second = _new_tenant("order-single-1"), _new_tenant("order-single-2")
    for tenant_id in (first, second):
        ingest_bytes(tenant_id=tenant_id, raw=SINGLE_DOC, filename="isp.md", embedder=embedder)
    yield first, second
    _drop(first, second)


# ------------------------------------------------- deterministic for one state


def test_repeated_queries_return_identical_order(single_doc_twice, embedder):
    """Within one database state the order must never move. This is what makes
    a recorded cassette replayable at all."""
    tenant_id, _ = single_doc_twice
    settings = make_settings()
    for question in QUESTIONS:
        runs = [
            _texts(retrieve(tenant_id=tenant_id, question=question,
                            embedder=embedder, reranker=None, settings=settings))
            for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2], f"order moved between runs for {question!r}"


def test_same_document_ingested_twice_retrieves_in_the_same_order(single_doc_twice, embedder):
    """The defect, directly. Same bytes, two ingests, different random ids -
    the evidence order must be identical. Before the fix this was the case
    that changed 15 of 24 prompt keys."""
    first, second = single_doc_twice
    settings = make_settings()
    for question in QUESTIONS:
        left = _texts(retrieve(tenant_id=first, question=question,
                               embedder=embedder, reranker=None, settings=settings))
        right = _texts(retrieve(tenant_id=second, question=question,
                                embedder=embedder, reranker=None, settings=settings))
        assert left == right, f"re-ingest reordered evidence for {question!r}"


def test_both_retrievers_order_identically_across_two_ingests(single_doc_twice, embedder):
    """Asserted at the retriever level too, so a failure points at which SQL
    ordering regressed rather than only at the packed result."""
    first, second = single_doc_twice
    question = QUESTIONS[0]
    query_vector = format_vector(embedder.embed_query(question))

    with tenant_session(first) as session:
        lex_a = [c.text for c in lexical_search(session, question, 40, None)]
        vec_a = [c.text for c in vector_search(session, query_vector, 40, None)]
    with tenant_session(second) as session:
        lex_b = [c.text for c in lexical_search(session, question, 40, None)]
        vec_b = [c.text for c in vector_search(session, query_vector, 40, None)]

    assert lex_a and vec_a, "fixture retrieved nothing; the assertion would be vacuous"
    assert lex_a == lex_b, "lexical ordering differed across ingests"
    assert vec_a == vec_b, "vector ordering differed across ingests"


def test_chunk_index_is_read_from_the_right_column(single_doc_twice, embedder):
    """Guards the column shift the fix introduced. chunk_index was added to the
    shared SELECT, which moves the score column; reading the wrong index would
    silently turn a chunk position into a relevance score."""
    tenant_id, _ = single_doc_twice
    question = QUESTIONS[0]
    query_vector = format_vector(embedder.embed_query(question))
    with tenant_session(tenant_id) as session:
        lexical = lexical_search(session, question, 40, None)
        vector = vector_search(session, query_vector, 40, None)
        expected = {
            row[0]: row[1]
            for row in session.execute(text("SELECT id, chunk_index FROM chunks")).all()
        }

    assert lexical and vector
    for candidate in lexical:
        assert candidate.chunk_index == expected[candidate.chunk_id]
        assert candidate.lexical_score is not None and candidate.lexical_score > 0.0, (
            "lexical score looks like a chunk index, not a rank"
        )
    for candidate in vector:
        assert candidate.chunk_index == expected[candidate.chunk_id]
        assert candidate.vector_score is not None and -1.0 <= candidate.vector_score <= 1.0, (
            f"vector score {candidate.vector_score} is outside cosine range"
        )


# ------------------------------------------------------------- fusion ordering


def _candidate(document: int, index: int, chunk_id: int, **overrides) -> Candidate:  # noqa: ANN003
    base = dict(
        chunk_id=uuid.UUID(int=chunk_id),
        document_id=uuid.UUID(int=document),
        chunk_index=index,
        filename=f"doc{document}.md",
        text=f"d{document}c{index}",
        parent_text=f"d{document}c{index}",
        heading_path=(),
        section=None,
        page_start=None,
        page_end=None,
        effective_date=None,
    )
    return Candidate(**{**base, **overrides})


def test_fusion_tie_break_ignores_chunk_id():
    """Tied fusion scores must order by (document_id, chunk_index). The chunk
    ids below are deliberately in the opposite order, so a sort that still
    looked at chunk_id would reverse the result."""
    ranking = [
        _candidate(document=1, index=0, chunk_id=999, lexical_score=1.0),
        _candidate(document=1, index=1, chunk_id=500, lexical_score=1.0),
        _candidate(document=1, index=2, chunk_id=1, lexical_score=1.0),
    ]
    # One retriever, so every chunk holds a distinct rank and scores differ;
    # feed the same list twice at swapped positions to force exact ties.
    fused = reciprocal_rank_fusion([ranking, list(reversed(ranking))], k=60)
    scores = {c.text: c.fusion_score for c in fused}
    assert len(set(scores.values())) == 2, "expected the middle chunk to tie with itself"

    tied = [c for c in fused if c.fusion_score == max(scores.values())]
    assert [c.chunk_index for c in tied] == sorted(c.chunk_index for c in tied), (
        "tied candidates are not in chunk_index order"
    )


def test_fusion_tie_break_never_overrides_score():
    """Ranking behaviour apart from tie-breaking must be unchanged: a higher
    fusion score always wins, whatever the tie-break key would prefer."""
    # chunk_index 0 would sort first on the tie-break, but it is ranked last
    # by the only retriever, so it must come last.
    ranking = [
        _candidate(document=1, index=9, chunk_id=1, lexical_score=1.0),
        _candidate(document=1, index=5, chunk_id=2, lexical_score=0.9),
        _candidate(document=1, index=0, chunk_id=3, lexical_score=0.8),
    ]
    fused = reciprocal_rank_fusion([ranking], k=60)
    assert [c.chunk_index for c in fused] == [9, 5, 0]
    assert [c.fusion_score for c in fused] == sorted(
        (c.fusion_score for c in fused), reverse=True
    )


def test_fusion_order_is_independent_of_input_chunk_ids():
    """The same corpus ingested twice differs only in its random ids. Fusion
    given the same scores must produce the same order regardless."""
    def build(id_base: int) -> list[Candidate]:
        return [
            _candidate(document=1, index=i, chunk_id=id_base + i, lexical_score=1.0)
            for i in range(5)
        ]

    first = [c.text for c in reciprocal_rank_fusion([build(100)], k=60)]
    second = [c.text for c in reciprocal_rank_fusion([build(900)], k=60)]
    assert first == second


# --------------------------------------------- the limit of what this fix buys


def test_cross_document_ties_still_depend_on_a_random_document_id(embedder):
    """Honest boundary. `documents.id` is gen_random_uuid(), so a tie spanning
    two documents orders by a value that is regenerated on every ingest.

    This is not asserted as instability - a given pair of random ids may sort
    either way - but as the mechanism: swapping only the document ids, with
    identical scores and chunk indices, changes the order. Measured on the
    multi-document eval corpus, 11 of 12 prompt keys still moved across two
    ingests after this fix, while the single-document corpus went to 0 of 12.

    Fixing it means ordering on something content-derived rather than on an
    id, which is a larger change and is not part of Step 8b.
    """
    low, high = uuid.UUID(int=1), uuid.UUID(int=2)
    pair = [
        _candidate(document=2, index=0, chunk_id=10, lexical_score=1.0),
        _candidate(document=1, index=0, chunk_id=11, lexical_score=1.0),
    ]
    fused = reciprocal_rank_fusion([pair, list(reversed(pair))], k=60)
    top_two = [c.document_id for c in fused[:2]]
    assert set(top_two) == {low, high}
    assert top_two == sorted(top_two, key=str), (
        "ordering should follow document_id, which is exactly why a re-ingest "
        "can reorder chunks that belong to different documents"
    )


def test_multi_document_ordering_is_stable_within_one_ingest(embedder):
    """The multi-document case is not stable across ingests, but it must still
    be stable for a given database state, or nothing is reproducible at all."""
    tenant_id = _new_tenant("order-multi")
    try:
        ingest_bytes(tenant_id=tenant_id, raw=DOC_A, filename="a.md", embedder=embedder)
        ingest_bytes(tenant_id=tenant_id, raw=DOC_B, filename="b.md", embedder=embedder)
        settings = make_settings()
        for question in QUESTIONS:
            runs = [
                _texts(retrieve(tenant_id=tenant_id, question=question,
                                embedder=embedder, reranker=None, settings=settings))
                for _ in range(3)
            ]
            assert runs[0] == runs[1] == runs[2]
    finally:
        _drop(tenant_id)

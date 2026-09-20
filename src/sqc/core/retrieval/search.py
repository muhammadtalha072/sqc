"""The two retrievers and their fusion.

Both queries are ordinary SELECTs with no tenant predicate. That is
deliberate: row-level security already scopes them, and adding a redundant
`WHERE tenant_id = ...` would suggest the policy is optional. If the filter
ever became the real control, forgetting it once would be a data leak.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

from sqc.core.retrieval.query import build_tsquery
from sqc.core.retrieval.types import Candidate

_SELECT_COLUMNS = """
    c.id, c.document_id, c.chunk_index, d.filename, c.text, c.parent_text,
    c.heading_path, c.section, c.page_start, c.page_end, d.effective_date,
    c.injection_flags
"""

_SCORE = 12
"""Index of the score column, which every query appends after the shared
columns above. Named so that adding a column moves one constant rather than
silently shifting what the callers read as a relevance score."""

_ORDER_TIE_BREAK = "c.document_id, c.chunk_index"
"""Tie-break for equal scores.

Not c.id: chunk ids are gen_random_uuid(), so two ingests of the same bytes
produced different orders among tied rows, which changed the evidence pack,
which changed the prompt, which orphaned every recorded cassette. Ties are
common here - the hashing embedder returns exactly equal cosine similarity
for many chunks, and a query made of document-title words scores nearly flat
across a whole document.

(document_id, chunk_index) is unique by schema constraint, so this is a total
order. document_id is itself regenerated per ingest, so this makes ordering
deterministic within a database state and across re-ingests of a
single-document corpus; ties that span two documents can still swap when the
documents are re-ingested. Fixing that would mean ordering on something
content-derived, which is a larger change than this one."""


def _row_to_candidate(row, *, lexical: float | None, vector: float | None) -> Candidate:  # noqa: ANN001
    return Candidate(
        chunk_id=row[0],
        document_id=row[1],
        chunk_index=row[2],
        filename=row[3],
        text=row[4],
        parent_text=row[5] or row[4],
        heading_path=tuple(row[6] or ()),
        section=row[7],
        page_start=row[8],
        page_end=row[9],
        effective_date=row[10],
        injection_flags=tuple(row[11] or ()),
        lexical_score=lexical,
        vector_score=vector,
    )


def lexical_search(
    session: Session,
    question: str,
    limit: int,
    document_ids: list[uuid.UUID] | None = None,
) -> list[Candidate]:
    """Full-text search over the generated tsvector.

    ts_rank_cd rather than ts_rank: the cover-density variant rewards
    query terms appearing close together, which separates a chunk actually
    about encryption at rest from one that mentions encryption in passing
    and storage somewhere else entirely.
    """
    tsquery = build_tsquery(question)
    if tsquery is None:
        return []

    sql = f"""
        SELECT {_SELECT_COLUMNS},
               ts_rank_cd(c.tsv, query, 32) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN to_tsquery('english', :tsquery) AS query
        WHERE c.tsv @@ query
          {"AND c.document_id = ANY(:document_ids)" if document_ids else ""}
        ORDER BY score DESC, {_ORDER_TIE_BREAK}
        LIMIT :limit
    """
    params: dict = {"tsquery": tsquery, "limit": limit}
    if document_ids:
        params["document_ids"] = document_ids
    rows = session.execute(text(sql), params).all()
    return [_row_to_candidate(row, lexical=float(row[_SCORE]), vector=None) for row in rows]


def vector_search(
    session: Session,
    query_vector: str,
    limit: int,
    document_ids: list[uuid.UUID] | None = None,
) -> list[Candidate]:
    """Nearest neighbours by cosine distance.

    The stored score is similarity (1 - distance) so that larger is better
    for every retriever, which keeps the fusion and signal code from having
    to remember which direction each score runs in.
    """
    sql = f"""
        SELECT {_SELECT_COLUMNS},
               1 - (c.embedding <=> CAST(:qv AS vector)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
          {"AND c.document_id = ANY(:document_ids)" if document_ids else ""}
        ORDER BY c.embedding <=> CAST(:qv AS vector), {_ORDER_TIE_BREAK}
        LIMIT :limit
    """
    params: dict = {"qv": query_vector, "limit": limit}
    if document_ids:
        params["document_ids"] = document_ids
    rows = session.execute(text(sql), params).all()
    return [_row_to_candidate(row, lexical=None, vector=float(row[_SCORE])) for row in rows]


def reciprocal_rank_fusion(
    rankings: list[list[Candidate]],
    k: int = 60,
) -> list[Candidate]:
    """Merge ranked lists by reciprocal rank: score = sum 1 / (k + rank).

    Rank-based rather than score-based on purpose. ts_rank_cd is unbounded
    and corpus-dependent while cosine similarity sits in [0, 1]; normalising
    them against each other is unstable, especially when one retriever
    returns two results and the other returns forty. RRF only needs the
    orderings, which is the one thing both retrievers agree on the meaning of.

    Raw per-retriever scores survive on the merged candidate, because
    confidence needs magnitude even though ranking does not.
    """
    merged: dict[uuid.UUID, Candidate] = {}
    scores: dict[uuid.UUID, float] = {}

    for ranking in rankings:
        for position, candidate in enumerate(ranking, start=1):
            key = candidate.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position)
            merged[key] = _merge(merged.get(key), candidate, position)

    # Tie-break matches the SQL: (document_id, chunk_index), never chunk_id.
    # Fusion scores collide constantly - 1/(k+rank) is the same value for any
    # two chunks holding the same rank in their one retriever - so this sort
    # decides real evidence order, not a rare edge case.
    ordered = sorted(
        merged.values(),
        key=lambda c: (-scores[c.chunk_id], str(c.document_id), c.chunk_index),
    )
    return [
        Candidate(**{**_fields(c), "fusion_score": round(scores[c.chunk_id], 8)})
        for c in ordered
    ]


def _fields(candidate: Candidate) -> dict:
    return {slot: getattr(candidate, slot) for slot in Candidate.__slots__}


def _merge(existing: Candidate | None, incoming: Candidate, position: int) -> Candidate:
    """Fold one retriever's view of a chunk into what we already have.

    Keeps both retrievers' raw scores and ranks on the same candidate, which
    is what lets the agreement signal exist downstream.
    """
    fields = _fields(existing if existing is not None else incoming)
    if incoming.lexical_score is not None:
        fields["lexical_score"] = incoming.lexical_score
        fields["lexical_rank"] = position
    if incoming.vector_score is not None:
        fields["vector_score"] = incoming.vector_score
        fields["vector_rank"] = position
    return Candidate(**fields)

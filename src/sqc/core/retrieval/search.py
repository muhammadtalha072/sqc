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
    c.id, c.document_id, d.filename, c.text, c.parent_text, c.heading_path,
    c.section, c.page_start, c.page_end, d.effective_date, c.injection_flags
"""


def _row_to_candidate(row, *, lexical: float | None, vector: float | None) -> Candidate:  # noqa: ANN001
    return Candidate(
        chunk_id=row[0],
        document_id=row[1],
        filename=row[2],
        text=row[3],
        parent_text=row[4] or row[3],
        heading_path=tuple(row[5] or ()),
        section=row[6],
        page_start=row[7],
        page_end=row[8],
        effective_date=row[9],
        injection_flags=tuple(row[10] or ()),
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
        ORDER BY score DESC, c.id
        LIMIT :limit
    """
    params: dict = {"tsquery": tsquery, "limit": limit}
    if document_ids:
        params["document_ids"] = document_ids
    rows = session.execute(text(sql), params).all()
    return [_row_to_candidate(row, lexical=float(row[11]), vector=None) for row in rows]


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
        ORDER BY c.embedding <=> CAST(:qv AS vector), c.id
        LIMIT :limit
    """
    params: dict = {"qv": query_vector, "limit": limit}
    if document_ids:
        params["document_ids"] = document_ids
    rows = session.execute(text(sql), params).all()
    return [_row_to_candidate(row, lexical=None, vector=float(row[11])) for row in rows]


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

    ordered = sorted(merged.values(), key=lambda c: (-scores[c.chunk_id], str(c.chunk_id)))
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

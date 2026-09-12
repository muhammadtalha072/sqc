"""Retrieval orchestration.

    question -> lexical + vector -> RRF -> rerank -> floor gate -> evidence pack

The floor gate is the first of the two refusal points in the system. If
nothing clears it, the caller refuses without ever calling the answering
model: there is no evidence to ground an answer in, and skipping the call
also means not paying for a refusal.
"""

from __future__ import annotations

import uuid

from sqc.config import Settings, get_settings
from sqc.core.ingestion.chunking import estimate_tokens
from sqc.core.retrieval.query import extract_terms
from sqc.core.retrieval.search import (
    lexical_search,
    reciprocal_rank_fusion,
    vector_search,
)
from sqc.core.retrieval.types import Candidate, Evidence, RetrievalResult, RetrievalSignals
from sqc.db.engine import tenant_session
from sqc.db.repository import format_vector
from sqc.providers.base import EmbeddingProvider, ProviderError, RerankProvider

MAX_EVIDENCE_TOKENS = 4000
"""Budget for the evidence pack. Caps cost per question and keeps the
answering prompt inside a size where the model reliably attends to all of
it rather than to the ends."""


def retrieve(
    *,
    tenant_id: uuid.UUID,
    question: str,
    embedder: EmbeddingProvider | None = None,
    reranker: RerankProvider | None = None,
    settings: Settings | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> RetrievalResult:
    """Find evidence for one question within one tenant."""
    settings = settings or get_settings()
    per_retriever = settings.candidates_per_retriever

    # No content words means no query. Lexical search already returns nothing,
    # but dense search always returns its k nearest neighbours however
    # meaningless the input, so a spreadsheet header row or an "N/A" cell
    # would retrieve arbitrary policy text and look answerable.
    if not extract_terms(question):
        return RetrievalResult(
            question=question,
            below_floor=True,
            refusal_reason="the question contains no searchable terms",
            signals=RetrievalSignals(),
        )

    query_vector: str | None = None
    if embedder is not None:
        try:
            query_vector = format_vector(embedder.embed_query(question))
        except ProviderError:
            # Degrade to lexical only. A dense-retrieval outage should cost
            # recall, not availability: BM25-style search still answers most
            # keyword-heavy questionnaire items.
            query_vector = None

    with tenant_session(tenant_id) as session:
        lexical = lexical_search(session, question, per_retriever, document_ids)
        vector = (
            vector_search(session, query_vector, per_retriever, document_ids)
            if query_vector
            else []
        )

    if not lexical and not vector:
        return RetrievalResult(
            question=question,
            below_floor=True,
            refusal_reason="no evidence retrieved for this question",
            signals=RetrievalSignals(),
        )

    fused = reciprocal_rank_fusion([lexical, vector], k=settings.rrf_k)

    # Injection-flagged chunks are retrieved but never packed as evidence.
    # They stay searchable so the customer can find and review them; they do
    # not reach the answering model.
    usable = [c for c in fused if not c.injection_flags]
    excluded = len(fused) - len(usable)

    if not usable:
        return RetrievalResult(
            question=question,
            candidates=tuple(fused),
            below_floor=True,
            refusal_reason=(
                "every retrieved passage is flagged as containing instruction-like "
                "text and was withheld from the answer"
            ),
            signals=RetrievalSignals(
                candidates_found=len(fused),
                lexical_hits=len(lexical),
                vector_hits=len(vector),
                injection_excluded=excluded,
            ),
        )

    ranked, reranked = _apply_rerank(reranker, question, usable, settings.evidence_top_k)
    top = ranked[: settings.evidence_top_k]
    # Returned candidates come from the reranked list so the rerank score
    # survives for audit and tuning. Withheld chunks are appended rather
    # than dropped: an operator debugging a refusal needs to see them.
    all_candidates = ranked + [c for c in fused if c.injection_flags]

    scores = [_score_of(c, reranked) for c in top]
    top_score = scores[0] if scores else 0.0
    gap = (scores[0] - scores[1]) if len(scores) > 1 else scores[0] if scores else 0.0

    agreement = sum(
        1 for c in fused if c.lexical_score is not None and c.vector_score is not None
    )
    evidence = _pack_evidence(top)

    signals = RetrievalSignals(
        candidates_found=len(fused),
        lexical_hits=len(lexical),
        vector_hits=len(vector),
        agreement=agreement,
        top_score=top_score,
        score_gap=gap,
        distinct_documents=len({c.document_id for c in top}),
        injection_excluded=excluded,
        reranked=reranked,
        evidence_tokens=sum(estimate_tokens(e.text) for e in evidence),
    )

    # The floor only applies when a reranker produced a calibrated relevance
    # score. RRF scores are relative to the result set, so thresholding them
    # would reject a good answer simply for having few competitors.
    below_floor = reranked and top_score < settings.retrieval_floor
    return RetrievalResult(
        question=question,
        evidence=() if below_floor else tuple(evidence),
        candidates=tuple(all_candidates),
        signals=signals,
        below_floor=below_floor,
        refusal_reason=(
            f"best passage scored {top_score:.3f}, below the retrieval floor "
            f"of {settings.retrieval_floor:.2f}"
            if below_floor
            else None
        ),
    )


def _apply_rerank(
    reranker: RerankProvider | None,
    question: str,
    candidates: list[Candidate],
    top_k: int,
) -> tuple[list[Candidate], bool]:
    """Reorder by a cross-encoder when one is configured.

    Reranking sees more candidates than it returns: its job is to promote
    the right chunk from deep in the fused list, which it cannot do if only
    the top few are shown to it.
    """
    if reranker is None or not candidates or getattr(reranker, "is_noop", False):
        # NoOpReranker preserves fusion order and returns flat zeros, which
        # are not relevance signals. Detected by marker rather than by score
        # pattern: a real reranker scoring everything zero is a meaningful
        # verdict that nothing retrieved is relevant, and discarding it would
        # hide exactly the case the floor gate exists to catch.
        return candidates, False
    try:
        ranked = reranker.rerank(question, [c.parent_text for c in candidates], max(top_k * 3, top_k))
    except ProviderError:
        # Rerank failure degrades to fusion order rather than failing the query.
        return candidates, False
    if not ranked:
        return candidates, False

    by_index = {item.index: item.score for item in ranked}

    scored = [
        Candidate(
            **{
                **{slot: getattr(c, slot) for slot in Candidate.__slots__},
                "rerank_score": by_index.get(index),
            }
        )
        for index, c in enumerate(candidates)
    ]
    scored.sort(
        key=lambda c: (-(c.rerank_score if c.rerank_score is not None else -1.0), -c.fusion_score)
    )
    return scored, True


def _score_of(candidate: Candidate, reranked: bool) -> float:
    if reranked and candidate.rerank_score is not None:
        return candidate.rerank_score
    return candidate.fusion_score


def _pack_evidence(candidates: list[Candidate]) -> list[Evidence]:
    """Group leaves that share a parent section into one evidence item.

    Two leaves from the same section carry the same parent_text, so sending
    both would pay twice for identical text and give the model two handles
    for one passage. Merging also keeps a rule and its exception in a single
    citable unit, which is the whole reason parent text exists.
    """
    packed: list[Evidence] = []
    by_section: dict[tuple[uuid.UUID, str], list[Candidate]] = {}
    order: list[tuple[uuid.UUID, str]] = []

    for candidate in candidates:
        key = (candidate.document_id, candidate.parent_text)
        if key not in by_section:
            by_section[key] = []
            order.append(key)
        by_section[key].append(candidate)

    budget = MAX_EVIDENCE_TOKENS
    for position, key in enumerate(order, start=1):
        group = by_section[key]
        head = group[0]
        body = head.parent_text or head.text
        cost = estimate_tokens(body)
        if cost > budget and packed:
            break
        budget -= cost
        pages = [p for c in group for p in (c.page_start, c.page_end) if p is not None]
        packed.append(
            Evidence(
                evidence_id=f"E{position}",
                chunk_ids=tuple(c.chunk_id for c in group),
                document_id=head.document_id,
                filename=head.filename,
                text=body,
                heading_path=head.heading_path,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                effective_date=head.effective_date,
                score=max(
                    c.rerank_score if c.rerank_score is not None else c.fusion_score
                    for c in group
                ),
                citation=head.citation,
            )
        )
    return packed

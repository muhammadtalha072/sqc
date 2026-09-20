"""Types returned by retrieval.

Every score the pipeline computes is kept, not just the ranking. Confidence
downstream is derived from measurable signals, so if a signal is discarded
here it cannot be used there, and the only thing left would be asking the
answering model how sure it feels.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class Candidate:
    """One chunk returned by one retriever, with that retriever's score."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    """Position of this chunk within its document. Carried so that ordering
    can tie-break on (filename, chunk_index) rather than on chunk_id or
    document_id, both of which are fresh random UUIDs on every ingest. Every
    key in that tie-break survives a re-ingest, which is what makes an
    evidence pack - and therefore a prompt, and therefore a recorded
    cassette - reproducible across ingests and across machines."""
    filename: str
    text: str
    parent_text: str
    heading_path: tuple[str, ...]
    section: str | None
    page_start: int | None
    page_end: int | None
    effective_date: date | None
    injection_flags: tuple[str, ...] = ()
    lexical_score: float | None = None
    vector_score: float | None = None
    lexical_rank: int | None = None
    vector_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        where = " > ".join(self.heading_path) if self.heading_path else "(no section)"
        if self.page_start is None:
            return f"{self.filename} - {where}"
        pages = (
            f"p{self.page_start}"
            if self.page_end in (None, self.page_start)
            else f"pp{self.page_start}-{self.page_end}"
        )
        return f"{self.filename} {pages} - {where}"


@dataclass(frozen=True, slots=True)
class Evidence:
    """A packed unit of evidence handed to the answering model.

    May cover several leaf chunks that share one parent section: sending the
    section once, with all its chunk ids, avoids paying for the same text
    twice and keeps a rule together with its exceptions.
    """

    evidence_id: str
    """Short stable handle the model cites, e.g. 'E1'. UUIDs invite
    transcription errors and waste tokens."""
    chunk_ids: tuple[uuid.UUID, ...]
    document_id: uuid.UUID
    filename: str
    text: str
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    effective_date: date | None
    score: float
    citation: str


@dataclass(frozen=True, slots=True)
class RetrievalSignals:
    """Measurable inputs to confidence. No model opinion appears here."""

    candidates_found: int = 0
    lexical_hits: int = 0
    vector_hits: int = 0
    agreement: int = 0
    """Chunks returned by both retrievers. Independent agreement is a
    stronger signal than a high score from either alone."""
    top_score: float = 0.0
    score_gap: float = 0.0
    """Top score minus second. A flat distribution means nothing stood out."""
    distinct_documents: int = 0
    injection_excluded: int = 0
    reranked: bool = False
    evidence_tokens: int = 0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "candidates_found": self.candidates_found,
            "lexical_hits": self.lexical_hits,
            "vector_hits": self.vector_hits,
            "agreement": self.agreement,
            "top_score": round(self.top_score, 6),
            "score_gap": round(self.score_gap, 6),
            "distinct_documents": self.distinct_documents,
            "injection_excluded": self.injection_excluded,
            "reranked": self.reranked,
            "evidence_tokens": self.evidence_tokens,
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    question: str
    evidence: tuple[Evidence, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    signals: RetrievalSignals = field(default_factory=RetrievalSignals)
    below_floor: bool = False
    """Nothing cleared the retrieval threshold. The caller must refuse
    without calling the answering model: no evidence means no answer, and
    skipping the call also avoids paying for a refusal."""
    refusal_reason: str | None = None

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence) and not self.below_floor

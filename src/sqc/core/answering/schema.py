"""The answering contract.

Two things are deliberately absent from the schema the model fills in.

There is no confidence field. Asking a language model how sure it is
produces a number that correlates with fluency, not with evidence. Status
and confidence are computed by the validator from measurable signals.

There is no chunk id field either. The model cites short handles - E1, E2 -
and the validator maps those back to real chunk ids. A model that cannot
see a UUID cannot invent one, so citation integrity holds by construction
rather than by checking afterwards.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqc.core.retrieval.types import Evidence, RetrievalSignals


class AnswerStatus(StrEnum):
    SUPPORTED = "supported"
    REVIEW_REQUIRED = "review_required"
    REFUSED = "refused"


class AnswerType(StrEnum):
    YES = "yes"
    NO = "no"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "The answer to the question, written only from the supplied "
                "evidence. Empty string if the evidence does not support an answer."
            ),
        },
        "answer_type": {
            "type": "string",
            "enum": [t.value for t in AnswerType],
            "description": (
                "yes/no only when the evidence settles the question as asked. "
                "partial when the evidence covers some cases but not all, for "
                "example a control that applies to administrators when the "
                "question asks about all users. not_found when the evidence "
                "does not address the question."
            ),
        },
        "claims": {
            "type": "array",
            "description": "Each factual statement in the answer, with its evidence.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Evidence handles such as E1. Never invent one.",
                    },
                },
                "required": ["text", "evidence_ids"],
            },
        },
        "evidence_sufficient": {
            "type": "boolean",
            "description": "False when the evidence does not settle the question.",
        },
        "conflict_detected": {
            "type": "boolean",
            "description": "True when two pieces of evidence disagree.",
        },
        "conflict_note": {"type": "string"},
        "reason": {
            "type": "string",
            "description": "One sentence on why this answer or refusal follows from the evidence.",
        },
    },
    "required": ["answer", "answer_type", "claims", "evidence_sufficient", "reason"],
}


@dataclass(frozen=True, slots=True)
class Citation:
    """A citation the validator resolved back to real retrieved evidence."""

    evidence_id: str
    chunk_ids: tuple[uuid.UUID, ...]
    document_id: uuid.UUID
    filename: str
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    effective_date: Any = None
    quote: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "chunk_ids": [str(c) for c in self.chunk_ids],
            "document_id": str(self.document_id),
            "filename": self.filename,
            "heading_path": list(self.heading_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "effective_date": str(self.effective_date) if self.effective_date else None,
        }


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    citations: tuple[Citation, ...] = ()
    supported: bool = True
    problem: str | None = None
    """Why the claim was rejected, when it was."""
    entailment: str | None = None
    """supported | unsupported | contradicted | unknown, or None when no
    entailment checker ran. A citation alone no longer confers support."""
    entailment_score: float | None = None
    entailment_detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "supported": self.supported,
            "problem": self.problem,
            "entailment": self.entailment,
            "entailment_score": self.entailment_score,
            "entailment_detail": self.entailment_detail,
            "citations": [c.as_dict() for c in self.citations],
        }


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Everything needed to display, audit and evaluate one answered question."""

    question: str
    status: AnswerStatus
    answer: str = ""
    answer_type: AnswerType | None = None
    claims: tuple[Claim, ...] = ()
    citations: tuple[Citation, ...] = ()
    evidence_chunk_ids: tuple[uuid.UUID, ...] = ()
    reason: str = ""
    validation_errors: tuple[str, ...] = ()
    validation_signals: dict[str, Any] = field(default_factory=dict)
    """Every check the validator ran and what it found. Kept so the eval
    suite can tell which check caused a status, and so a threshold can be
    calibrated later from recorded distributions rather than guessed."""
    failure_stage: str | None = None
    """retrieval | answering | validation | provider, when the outcome was
    not a clean answer. Separating these is the difference between knowing
    the system failed and knowing where."""
    evidence: tuple[Evidence, ...] = ()
    retrieval_signals: RetrievalSignals = field(default_factory=RetrievalSignals)
    llm_model: str | None = None
    llm_provider: str | None = None
    embedding_model: str | None = None
    prompt_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    created_at: datetime | None = None

    @property
    def answered(self) -> bool:
        return self.status is AnswerStatus.SUPPORTED

    @property
    def needs_human(self) -> bool:
        return self.status is not AnswerStatus.SUPPORTED

    def to_audit_record(self) -> dict[str, Any]:
        """Shape matching the answer_runs table, ready to persist.

        Built here rather than in the database layer so that an answer can be
        serialised and compared in the eval suite without a database at all.
        """
        return {
            "question": self.question,
            "outcome": (
                "answered" if self.status is AnswerStatus.SUPPORTED else self.status.value
            ),
            "answer_text": self.answer or None,
            "refusal_reason": self.reason if self.needs_human else None,
            "retrieved_chunk_ids": [str(c) for c in self.evidence_chunk_ids],
            "citations": [c.as_dict() for c in self.citations],
            "signals": {
                **self.retrieval_signals.as_dict(),
                "claims_total": len(self.claims),
                "claims_unsupported": sum(1 for c in self.claims if not c.supported),
                "validation_errors": list(self.validation_errors),
                "answer_type": self.answer_type.value if self.answer_type else None,
                "failure_stage": self.failure_stage,
                **{f"check_{k}": v for k, v in self.validation_signals.items()},
            },
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "prompt_version": self.prompt_version,
            "latency_ms": self.latency_ms,
            "confidence_label": self.status.value,
            "confidence_score": None,
            "created_at": self.created_at,
        }

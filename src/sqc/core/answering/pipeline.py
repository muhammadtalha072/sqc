"""Answering pipeline.

    question -> retrieval -> evidence pack -> LLM -> validator -> AnswerResult

Every failure path ends in a refusal carrying a reason, never in an
exception reaching the caller and never in an unvalidated answer. A provider
timeout and a hallucinated citation both produce the same externally visible
outcome: no answer, and a statement of why.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from sqc.config import Settings, get_settings
from sqc.core.answering.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from sqc.core.answering.schema import (
    ANSWER_SCHEMA,
    AnswerResult,
    AnswerStatus,
    Citation,
)
from sqc.core.answering.entailment import EntailmentProvider
from sqc.core.answering.validator import validate
from sqc.core.retrieval.pipeline import retrieve
from sqc.core.retrieval.types import RetrievalResult
from sqc.providers.base import EmbeddingProvider, LLMProvider, ProviderError, RerankProvider

MAX_ANSWER_TOKENS = 2048


def answer_question(
    *,
    tenant_id: uuid.UUID,
    question: str,
    llm: LLMProvider,
    embedder: EmbeddingProvider | None = None,
    reranker: RerankProvider | None = None,
    settings: Settings | None = None,
    document_ids: list[uuid.UUID] | None = None,
    entailment: EntailmentProvider | None = None,
) -> AnswerResult:
    """Answer one questionnaire question from one tenant's evidence."""
    settings = settings or get_settings()
    started = time.perf_counter()

    retrieval = retrieve(
        tenant_id=tenant_id,
        question=question,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
        document_ids=document_ids,
    )

    if not retrieval.has_evidence:
        # Refuse without calling the model. There is nothing to ground an
        # answer in, and a refusal should not cost an API call.
        return _refusal(
            question=question,
            reason=retrieval.refusal_reason or "no evidence was retrieved for this question",
            retrieval=retrieval,
            embedder=embedder,
            started=started,
        )

    user_prompt = build_user_prompt(question, retrieval.evidence)

    try:
        response = llm.complete_structured(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=ANSWER_SCHEMA,
            max_tokens=MAX_ANSWER_TOKENS,
        )
    except ProviderError as exc:
        return _refusal(
            question=question,
            reason=f"the answering model could not be reached: {exc}",
            stage="provider",
            retrieval=retrieval,
            embedder=embedder,
            started=started,
            llm_model=getattr(llm, "model", None),
            errors=("provider error",),
        )
    except Exception as exc:  # noqa: BLE001 - an unexpected fault must still refuse safely
        return _refusal(
            question=question,
            reason="the answering model failed unexpectedly",
            stage="provider",
            retrieval=retrieval,
            embedder=embedder,
            started=started,
            llm_model=getattr(llm, "model", None),
            errors=(f"unhandled provider fault: {type(exc).__name__}",),
        )

    outcome = validate(
        response.data, retrieval.evidence, entailment=entailment, question=question
    )
    status, answer, answer_type = outcome.status, outcome.answer, outcome.answer_type
    claims, errors, reason = outcome.claims, outcome.errors, outcome.reason

    citations: list[Citation] = []
    seen: set[str] = set()
    for claim in claims:
        for citation in claim.citations:
            if citation.evidence_id not in seen:
                seen.add(citation.evidence_id)
                citations.append(citation)

    return AnswerResult(
        question=question,
        status=status,
        answer=answer,
        answer_type=answer_type,
        claims=tuple(claims),
        citations=tuple(citations),
        evidence_chunk_ids=tuple(
            chunk_id for item in retrieval.evidence for chunk_id in item.chunk_ids
        ),
        reason=reason,
        validation_errors=tuple(errors),
        validation_signals=outcome.signals,
        failure_stage=outcome.failure_stage,
        evidence=retrieval.evidence,
        retrieval_signals=retrieval.signals,
        llm_model=response.model,
        llm_provider=type(llm).__name__,
        embedding_model=getattr(embedder, "model", None),
        prompt_version=PROMPT_VERSION,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
        created_at=datetime.now(UTC),
    )


def _refusal(
    *,
    question: str,
    reason: str,
    retrieval: RetrievalResult,
    embedder: EmbeddingProvider | None,
    started: float,
    llm_model: str | None = None,
    errors: tuple[str, ...] = (),
    stage: str = "retrieval",
) -> AnswerResult:
    """A refusal that still carries its evidence and signals.

    Refusals are audited as carefully as answers: without the retrieval
    signals behind them there is no way to tell a correct refusal from an
    over-cautious one, and no way to tune the difference.
    """
    return AnswerResult(
        question=question,
        status=AnswerStatus.REFUSED,
        reason=reason,
        evidence=retrieval.evidence,
        evidence_chunk_ids=tuple(
            chunk_id for item in retrieval.evidence for chunk_id in item.chunk_ids
        ),
        retrieval_signals=retrieval.signals,
        validation_errors=errors,
        failure_stage=stage,
        llm_model=llm_model,
        embedding_model=getattr(embedder, "model", None),
        prompt_version=PROMPT_VERSION,
        latency_ms=int((time.perf_counter() - started) * 1000),
        created_at=datetime.now(UTC),
    )

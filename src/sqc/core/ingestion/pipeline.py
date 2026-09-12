"""Ingestion pipeline: bytes in, indexed chunks out.

    parse -> chunk -> embed -> persist

The whole thing runs in one transaction. A document is either fully
searchable or absent; there is no state where half a policy is indexed and
the answering engine confidently cites the half that made it in while the
contradicting clause is missing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy.exc import IntegrityError

from sqc.core.ingestion.chunking import chunk_document
from sqc.core.ingestion.model import LeafChunk, ParsedDocument
from sqc.core.ingestion.parsers import DocumentParseError, parse_bytes
from sqc.db.engine import tenant_session
from sqc.db.repository import (
    delete_document,
    find_document_by_hash,
    insert_chunks,
    insert_document,
    mark_document_indexed,
)
from sqc.providers.base import EmbeddingProvider, ProviderError


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: uuid.UUID | None
    filename: str
    chunk_count: int = 0
    embedded: bool = False
    skipped_duplicate: bool = False
    warnings: tuple[str, ...] = field(default=())
    injection_flagged: int = 0
    parsed: ParsedDocument | None = None

    @property
    def ok(self) -> bool:
        return self.document_id is not None


class IngestionError(RuntimeError):
    pass


def ingest_bytes(
    *,
    tenant_id: uuid.UUID,
    raw: bytes,
    filename: str,
    embedder: EmbeddingProvider | None,
    doc_type: str = "unknown",
    replace_existing: bool = False,
) -> IngestionResult:
    """Parse, chunk, embed and store one document.

    Embedding happens before the transaction opens. A provider call can take
    tens of seconds across many batches, and holding a database transaction
    open across a slow network call is how connection pools die.
    """
    try:
        parsed = parse_bytes(raw, filename)
    except DocumentParseError as exc:
        raise IngestionError(str(exc)) from exc

    chunks: list[LeafChunk] = chunk_document(parsed)
    if not chunks:
        raise IngestionError(
            f"{filename} produced no chunks; "
            + (parsed.warnings[0] if parsed.warnings else "the document appears to be empty")
        )

    # Duplicate check comes before embedding, not after. The hash falls out
    # of parsing for free, while embedding a long policy costs real money and
    # minutes. Re-ingesting a stored file must not spend either.
    if not replace_existing:
        with tenant_session(tenant_id) as session:
            existing = find_document_by_hash(session, parsed.content_sha256)
        if existing is not None:
            return IngestionResult(
                document_id=existing,
                filename=filename,
                skipped_duplicate=True,
                warnings=parsed.warnings,
                parsed=parsed,
            )

    vectors: list[tuple[float, ...]] | None = None
    embedding_model: str | None = None
    if embedder is not None:
        try:
            batch = embedder.embed_documents([c.embed_text for c in chunks])
        except ProviderError as exc:
            raise IngestionError(f"embedding {filename} failed: {exc}") from exc
        if len(batch.vectors) != len(chunks):
            raise IngestionError(
                f"embedder returned {len(batch.vectors)} vectors for {len(chunks)} chunks"
            )
        vectors = list(batch.vectors)
        embedding_model = batch.model

    try:
        with tenant_session(tenant_id) as session:
            if replace_existing:
                existing = find_document_by_hash(session, parsed.content_sha256)
                if existing is not None:
                    # Replace rather than update: re-chunking can produce a
                    # different number of chunks, and stale chunk rows would
                    # keep being cited long after the text changed.
                    delete_document(session, existing)

            document_id = insert_document(
                session,
                tenant_id=tenant_id,
                filename=filename,
                content_sha256=parsed.content_sha256,
                doc_type=doc_type,
                page_count=parsed.page_count,
                effective_date=parsed.effective_date,
                version_label=parsed.version_label,
            )
            insert_chunks(
                session,
                tenant_id=tenant_id,
                document_id=document_id,
                chunks=chunks,
                vectors=vectors,
                embedding_model=embedding_model,
            )
            mark_document_indexed(session, document_id)
    except IntegrityError:
        # Two ingests of the same file raced between the check above and this
        # insert. The unique index settled it; report the winner.
        with tenant_session(tenant_id) as session:
            winner = find_document_by_hash(session, parsed.content_sha256)
        if winner is None:
            raise
        return IngestionResult(
            document_id=winner,
            filename=filename,
            skipped_duplicate=True,
            warnings=parsed.warnings,
            parsed=parsed,
        )

    return IngestionResult(
        document_id=document_id,
        filename=filename,
        chunk_count=len(chunks),
        embedded=vectors is not None,
        warnings=parsed.warnings,
        injection_flagged=sum(1 for c in chunks if c.injection_flags),
        parsed=parsed,
    )


def ingest_file(
    *,
    tenant_id: uuid.UUID,
    path: str,
    embedder: EmbeddingProvider | None,
    doc_type: str = "unknown",
    replace_existing: bool = False,
) -> IngestionResult:
    import pathlib

    file_path = pathlib.Path(path)
    return ingest_bytes(
        tenant_id=tenant_id,
        raw=file_path.read_bytes(),
        filename=file_path.name,
        embedder=embedder,
        doc_type=doc_type,
        replace_existing=replace_existing,
    )

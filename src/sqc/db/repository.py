"""Data access for documents and chunks.

Every function takes a Session from `tenant_session`, so RLS is already in
force. No function here filters by tenant_id in its WHERE clause: doing so
would imply the policy is optional, and a forgotten filter would then be a
data leak rather than a redundancy.

Vectors are bound as text and cast with ::vector rather than going through
a pgvector type adapter. One less version-sensitive dependency in the write
path, and the cast fails loudly on a malformed vector instead of silently
coercing it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

from sqc.core.ingestion.model import LeafChunk


@dataclass(frozen=True, slots=True)
class DocumentRow:
    id: uuid.UUID
    filename: str
    doc_type: str
    status: str
    content_sha256: str
    page_count: int | None
    effective_date: date | None
    version_label: str | None
    chunk_count: int = 0


def format_vector(vector: tuple[float, ...] | list[float]) -> str:
    """pgvector's text input format: [0.1,0.2,0.3]."""
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def find_document_by_hash(session: Session, content_sha256: str) -> uuid.UUID | None:
    row = session.execute(
        text("SELECT id FROM documents WHERE content_sha256 = :h"), {"h": content_sha256}
    ).first()
    return row[0] if row else None


def insert_document(
    session: Session,
    *,
    tenant_id: uuid.UUID,
    filename: str,
    content_sha256: str,
    doc_type: str = "unknown",
    page_count: int | None = None,
    effective_date: date | None = None,
    version_label: str | None = None,
) -> uuid.UUID:
    """Create a document in 'pending'. It becomes 'indexed' only once its
    chunks are committed, so a crash mid-ingest leaves an obviously
    incomplete row rather than a document that looks searchable but is not."""
    return session.execute(
        text(
            """
            INSERT INTO documents
                (tenant_id, filename, doc_type, content_sha256, page_count,
                 effective_date, version_label, status)
            VALUES (:tenant_id, :filename, :doc_type, :sha, :pages,
                    :effective_date, :version, 'pending')
            RETURNING id
            """
        ),
        {
            "tenant_id": tenant_id,
            "filename": filename,
            "doc_type": doc_type,
            "sha": content_sha256,
            "pages": page_count,
            "effective_date": effective_date,
            "version": version_label,
        },
    ).scalar_one()


def insert_chunks(
    session: Session,
    *,
    tenant_id: uuid.UUID,
    document_id: uuid.UUID,
    chunks: list[LeafChunk],
    vectors: list[tuple[float, ...]] | None,
    embedding_model: str | None,
) -> int:
    """Bulk-insert chunks. vectors may be None to store text only.

    Storing text without vectors is a legitimate state: keyword retrieval
    still works, and it lets ingestion proceed when an embedding provider
    is down, with a re-embed pass later.
    """
    if not chunks:
        return 0
    if vectors is not None and len(vectors) != len(chunks):
        raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks")

    rows = []
    for position, chunk in enumerate(chunks):
        rows.append(
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "parent_text": chunk.parent_text,
                "embed_text": chunk.embed_text,
                "heading_path": list(chunk.heading_path),
                "heading_text": chunk.heading_text,
                "section": chunk.section,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
                "injection_flags": list(chunk.injection_flags),
                "embedding": format_vector(vectors[position]) if vectors else None,
                "embedding_model": embedding_model if vectors else None,
            }
        )

    session.execute(
        text(
            """
            INSERT INTO chunks
                (tenant_id, document_id, chunk_index, text, parent_text, embed_text,
                 heading_path, heading_text, section, page_start, page_end,
                 token_count, injection_flags, embedding, embedding_model)
            VALUES
                (:tenant_id, :document_id, :chunk_index, :text, :parent_text, :embed_text,
                 :heading_path, :heading_text, :section, :page_start, :page_end,
                 :token_count, :injection_flags, CAST(:embedding AS vector), :embedding_model)
            """
        ),
        rows,
    )
    return len(rows)


def mark_document_indexed(session: Session, document_id: uuid.UUID) -> None:
    session.execute(
        text(
            "UPDATE documents SET status = 'indexed', indexed_at = now(),"
            " error_message = NULL WHERE id = :id"
        ),
        {"id": document_id},
    )


def mark_document_failed(session: Session, document_id: uuid.UUID, message: str) -> None:
    session.execute(
        text("UPDATE documents SET status = 'failed', error_message = :msg WHERE id = :id"),
        {"id": document_id, "msg": message[:2000]},
    )


def delete_document(session: Session, document_id: uuid.UUID) -> None:
    """Chunks disappear with it via ON DELETE CASCADE."""
    session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})


def get_document(session: Session, document_id: uuid.UUID) -> DocumentRow | None:
    row = session.execute(
        text(
            """
            SELECT d.id, d.filename, d.doc_type, d.status, d.content_sha256,
                   d.page_count, d.effective_date, d.version_label,
                   (SELECT count(*) FROM chunks c WHERE c.document_id = d.id)
            FROM documents d WHERE d.id = :id
            """
        ),
        {"id": document_id},
    ).first()
    return DocumentRow(*row) if row else None


def list_documents(session: Session) -> list[DocumentRow]:
    rows = session.execute(
        text(
            """
            SELECT d.id, d.filename, d.doc_type, d.status, d.content_sha256,
                   d.page_count, d.effective_date, d.version_label,
                   (SELECT count(*) FROM chunks c WHERE c.document_id = d.id)
            FROM documents d ORDER BY d.uploaded_at DESC
            """
        )
    ).all()
    return [DocumentRow(*row) for row in rows]


def count_chunks(session: Session) -> int:
    return session.execute(text("SELECT count(*) FROM chunks")).scalar_one()


def ensure_tenant(session_or_conn, tenant_id: uuid.UUID, name: str) -> None:  # noqa: ANN001
    """Insert a tenant if absent. Runs on the admin connection: the tenants
    table is the registry that RLS policies are keyed against, so it cannot
    itself be tenant-scoped."""
    session_or_conn.execute(
        text(
            "INSERT INTO tenants (id, name) VALUES (:id, :name)"
            " ON CONFLICT (id) DO NOTHING"
        ),
        {"id": tenant_id, "name": name},
    )

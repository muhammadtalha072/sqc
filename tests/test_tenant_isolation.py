"""Tenant isolation is a security control, so it is tested against a real
database, not mocked. These tests must run in CI against Postgres.

They assert three things:
  1. a tenant sees only its own rows
  2. a tenant cannot write rows belonging to another tenant
  3. with no tenant set, the app role sees nothing (default-deny)
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from sqc.db.engine import get_admin_engine, get_engine, tenant_session


@pytest.fixture(scope="module")
def two_tenants() -> tuple[uuid.UUID, uuid.UUID]:
    """Create two tenants, each with one document and one chunk."""
    a, b = uuid.uuid4(), uuid.uuid4()

    # The tenants table carries no RLS: it is the registry itself, written
    # only by provisioning.
    with get_admin_engine().begin() as conn:
        for tid, label in ((a, "Tenant A"), (b, "Tenant B")):
            conn.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tid, "name": label},
            )

    # Everything tenant-owned is written through the tenant-scoped session.
    # Because RLS is FORCEd, even the schema owner has no bypass, so this is
    # the only write path that exists anywhere in the system.
    for tid, label in ((a, "Tenant A"), (b, "Tenant B")):
        with tenant_session(tid) as s:
            doc = uuid.uuid4()
            s.execute(
                text(
                    "INSERT INTO documents (id, tenant_id, filename, content_sha256)"
                    " VALUES (:id, :t, :f, :s)"
                ),
                {"id": doc, "t": tid, "f": f"{label}.pdf", "s": uuid.uuid4().hex},
            )
            s.execute(
                text(
                    "INSERT INTO chunks (tenant_id, document_id, chunk_index,"
                    " text, embed_text) VALUES (:t, :d, 0, :x, :x)"
                ),
                {"t": tid, "d": doc, "x": f"{label} encrypts data at rest with AES-256."},
            )
    yield a, b
    with get_admin_engine().begin() as conn:
        conn.execute(text("DELETE FROM tenants WHERE id = ANY(:ids)"), {"ids": [a, b]})


def test_tenant_sees_only_own_documents(two_tenants):
    a, b = two_tenants
    with tenant_session(a) as s:
        names = [r[0] for r in s.execute(text("SELECT filename FROM documents"))]
    assert names == ["Tenant A.pdf"]

    with tenant_session(b) as s:
        names = [r[0] for r in s.execute(text("SELECT filename FROM documents"))]
    assert names == ["Tenant B.pdf"]


def test_explicit_cross_tenant_query_returns_nothing(two_tenants):
    """Even naming the other tenant's id directly returns zero rows:
    the policy filters before the WHERE clause is considered."""
    a, b = two_tenants
    with tenant_session(a) as s:
        rows = s.execute(
            text("SELECT filename FROM documents WHERE tenant_id = :other"), {"other": b}
        ).all()
    assert rows == []


def test_chunk_text_does_not_leak_across_tenants(two_tenants):
    a, b = two_tenants
    with tenant_session(a) as s:
        texts = [r[0] for r in s.execute(text("SELECT text FROM chunks"))]
    assert len(texts) == 1
    assert "Tenant B" not in texts[0]


def test_cannot_insert_row_for_another_tenant(two_tenants):
    """WITH CHECK blocks writing a row the writer could not read back."""
    a, b = two_tenants
    with pytest.raises(Exception) as exc:
        with tenant_session(a) as s:
            s.execute(
                text(
                    "INSERT INTO documents (tenant_id, filename, content_sha256)"
                    " VALUES (:t, 'smuggled.pdf', :s)"
                ),
                {"t": b, "s": uuid.uuid4().hex},
            )
    assert "row-level security" in str(exc.value).lower()


def test_default_deny_without_tenant_setting(two_tenants):
    """A query that forgets to scope the tenant must return nothing.
    This is why app.tenant_id is read with the missing_ok flag: unset
    yields NULL, the policy is false, and the query fails closed."""
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT count(*) FROM documents")).scalar_one()
    assert rows == 0

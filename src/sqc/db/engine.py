"""Database engines and the tenant-scoped session.

Application code must never issue a query outside `tenant_session`. That is the
only place `app.tenant_id` is set, and RLS returns zero rows without it, so a
forgotten scope fails closed rather than leaking.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from sqc.config import get_settings

_app_engine: Engine | None = None
_admin_engine: Engine | None = None


def get_engine() -> Engine:
    """Engine for the restricted app role. RLS applies to every statement."""
    global _app_engine
    if _app_engine is None:
        _app_engine = create_engine(get_settings().database_url, pool_pre_ping=True)
    return _app_engine


def get_admin_engine() -> Engine:
    """Engine for migrations and tenant provisioning only. Never per-request."""
    global _admin_engine
    if _admin_engine is None:
        _admin_engine = create_engine(get_settings().admin_database_url, pool_pre_ping=True)
    return _admin_engine


@contextmanager
def tenant_session(tenant_id: uuid.UUID | str) -> Iterator[Session]:
    """Open a session bound to one tenant for the life of the transaction.

    SET LOCAL scopes the setting to this transaction, so a pooled connection
    handed to another request cannot inherit a previous tenant's id.
    """
    tid = str(uuid.UUID(str(tenant_id)))  # reject anything that is not a UUID
    maker = sessionmaker(bind=get_engine(), expire_on_commit=False)
    session = maker()
    try:
        # SET LOCAL takes no bind parameters, so set_config is used instead:
        # is_local=true gives the same transaction scope while keeping the
        # tenant id a bound value that never reaches the SQL parser as text.
        session.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": tid}
        )
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engines() -> None:
    """Drop cached engines. Used by tests after changing settings."""
    global _app_engine, _admin_engine
    for engine in (_app_engine, _admin_engine):
        if engine is not None:
            engine.dispose()
    _app_engine = None
    _admin_engine = None

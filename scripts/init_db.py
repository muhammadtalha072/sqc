"""Apply schema.sql to the database. Run with admin credentials.

    python scripts/init_db.py

The embedding dimension is substituted here rather than hard-coded in SQL,
so switching embedding provider is a config change plus a re-index, not a
schema rewrite.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from sqlalchemy import text

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sqc.config import get_settings  # noqa: E402
from sqc.db.engine import get_admin_engine  # noqa: E402

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "sqc" / "db" / "schema.sql"


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    settings = get_settings()
    sql = SCHEMA_PATH.read_text().replace("{{EMBEDDING_DIM}}", str(settings.embedding_dim))

    # One transaction: either the whole schema applies or none of it does.
    with get_admin_engine().begin() as conn:
        conn.execute(text(sql))

    print(f"schema applied (embedding_dim={settings.embedding_dim})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

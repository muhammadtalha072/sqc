"""Create a tenant.

    python scripts/create_tenant.py "Acme Corp"

Prints the tenant id. Keep it: every other command needs it.
Runs on the admin connection because `tenants` is the registry that RLS
policies are keyed against and so cannot be tenant-scoped itself.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sqc.db.engine import get_admin_engine  # noqa: E402
from sqc.db.repository import ensure_tenant  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="display name, e.g. 'Acme Corp'")
    parser.add_argument("--id", default=None, help="reuse a specific UUID")
    args = parser.parse_args()

    tenant_id = uuid.UUID(args.id) if args.id else uuid.uuid4()
    with get_admin_engine().begin() as conn:
        ensure_tenant(conn, tenant_id, args.name)

    print(tenant_id)
    print(f"# export SQC_TENANT={tenant_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

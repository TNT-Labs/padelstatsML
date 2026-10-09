"""Recover access: create or reactivate an administrator.

    python scripts/create_admin.py <username>
    docker compose exec api python scripts/create_admin.py admin

Prints a temporary password, to be replaced at the first sign-in. Every open
session of that user is closed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip())
        return 2
    from app.core.accounts import adopt_orphans, reset_admin
    from app.core.database import init_schema, sync_session

    init_schema()
    try:
        with sync_session() as session:
            password = reset_admin(session, sys.argv[1])
            adopt_orphans(session)
    except ValueError as exc:
        print(f"Errore: {exc}")
        return 1
    print(f"Amministratore «{sys.argv[1].strip().lower()}» pronto.")
    print(f"Password provvisoria: {password}")
    print("Al primo accesso verrà chiesto di sceglierne una nuova.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

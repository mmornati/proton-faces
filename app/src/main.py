"""Entry point: serves the FastAPI app (and optionally the indexer threads).

Phase 1 of issue #4: by default this process only serves uvicorn. The
indexing pipeline now runs in a separate `indexer` container (see
`indexer_main.py`) so the API event loop isn't preempted by face detection.

Set `RUN_INDEXER=1` to keep the legacy in-process indexer behaviour — useful
as a fallback while the new container is being rolled out, and for local
single-process dev setups.

Also handles one-shot admin commands used at first-time setup:
  --create-admin  : create the first user (role=admin) interactively
  --reset-password: reset an existing user's password (admin recovery)
"""
from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys

import uvicorn

from api import app  # noqa: F401  (registers FastAPI routes; also runs init)
import store
from auth import hash_password

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _create_admin(username: str, display_name: str | None) -> int:
    store.init_db()
    if store.get_user_by_username(username) is not None:
        print(f"user '{username}' already exists", file=sys.stderr)
        return 2
    pw = os.environ.get("ADMIN_PASSWORD") or getpass.getpass("password: ")
    if len(pw) < 8:
        print("password must be at least 8 characters", file=sys.stderr)
        return 2
    user_id = store.create_user(
        username=username,
        password_hash=hash_password(pw),
        role="admin",
        display_name=display_name or username,
    )
    # One-time: copy any pre-existing photos.favorited=1 rows into the new
    # admin's per-user favorites so no stars are lost on upgrade.
    backfilled = store.backfill_legacy_favorites(user_id)
    print(f"admin user '{username}' created (id={user_id})")
    if backfilled:
        print(f"backfilled {backfilled} legacy favorite(s) into this user's favorites")
    return 0


def _reset_password(username: str) -> int:
    store.init_db()
    row = store.get_user_by_username(username)
    if row is None:
        print(f"user '{username}' not found", file=sys.stderr)
        return 2
    pw = os.environ.get("ADMIN_PASSWORD") or getpass.getpass("new password: ")
    if len(pw) < 8:
        print("password must be at least 8 characters", file=sys.stderr)
        return 2
    store.update_user(row["id"], password_hash=hash_password(pw))
    n = store.revoke_all_tokens(row["id"])
    print(f"password reset for '{username}'; {n} active session(s) revoked")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="proton-faces")
    parser.add_argument(
        "--backfill-gps",
        action="store_true",
        help="One-shot: attach GPS/place data from the local Takeout export (PHOTOS_DIR), then exit.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="With --backfill-gps: ignore the sha1->GPS cache and rehash the local Takeout export.",
    )
    parser.add_argument(
        "--create-admin",
        nargs="?",
        const="__prompt__",
        default=None,
        metavar="USERNAME",
        help="One-shot: create an admin user (prompts for username/password).",
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help="With --create-admin: set the user's display name (defaults to username).",
    )
    parser.add_argument(
        "--reset-password",
        default=None,
        metavar="USERNAME",
        help="One-shot: reset an existing user's password.",
    )
    args = parser.parse_args()

    if args.create_admin is not None:
        username = args.create_admin
        if username == "__prompt__":
            username = input("username: ").strip()
        sys.exit(_create_admin(username, args.display_name))

    if args.reset_password is not None:
        sys.exit(_reset_password(args.reset_password))

    if args.backfill_gps:
        from indexer import backfill_gps, enrich_places
        matched = backfill_gps(rebuild_cache=args.rebuild_cache)
        logging.getLogger(__name__).info("GPS backfill done: %d photos matched", matched)
        enriched = enrich_places()
        logging.getLogger(__name__).info("GPS place enrichment done: %d photos", enriched)
        return

    if _env_bool("RUN_INDEXER", False):
        logging.getLogger(__name__).info(
            "RUN_INDEXER=1: starting in-process indexer alongside uvicorn"
        )
        from indexer import start
        start()
    else:
        logging.getLogger(__name__).info(
            "RUN_INDEXER unset: serving API only; indexer runs in its own container"
        )

    from config import settings
    # Start the admin's auto-backup daemon (schedule read from admin_config.json).
    from admin import start_backup_worker
    start_backup_worker()
    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
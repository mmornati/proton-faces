#!/usr/bin/env python3
"""Recover a person whose faces were absorbed by a mistaken merge.

When two people are merged, `merge_person` re-parents the source person's
faces to the target (`UPDATE faces SET person_id=? WHERE person_id=?`) and
deletes the source row. Face rows keep their immutable AUTOINCREMENT `id`, so
a pre-merge backup's face-ID list maps 1:1 onto live face rows. This script
uses that to move the faces back to a recreated person row.

Workflow (run on the host, with the indexer and app stopped):

    docker compose stop indexer app
    cp /media/12tb/photos-index/index.sqlite3 \\
       /media/12tb/photos-index/_backups/index-pre-recover-$(date +%Y%m%d-%H%M%S).sqlite3
    python3 scripts/recover-merged-person.py \\
        --backup /media/12tb/photos-index/_backups/index-20260910-062505.sqlite3 \\
        --name "Gaia Mornati"
    docker compose start indexer app

The indexer rewrites the mmap face sidecar on its debounced schedule after
startup, so the API picks up the new face->person mapping within a couple of
minutes. Use --dry-run to preview without writing.

Run from the repo root:
    python scripts/recover-merged-person.py --backup <path> --name "Gaia Mornati"
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

# SQLite caps parameter placeholders at 999; keep the chunk small enough that
# every generated query stays well below it (mirrors store._SQL_CHUNK).
_SQL_CHUNK = 500


def _connect(path: Path, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        uri = f"file:{path}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    return sqlite3.connect(path)


def _find_person(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, name, cover_uid, cover_face_id, face_count, photo_count "
        "FROM people WHERE name = ? ORDER BY id",
        (name,),
    ).fetchone()


def _face_ids_for_person(conn: sqlite3.Connection, person_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT id FROM faces WHERE person_id = ? ORDER BY id", (person_id,)
    ).fetchall()
    return [r["id"] for r in rows]


def _recount(conn: sqlite3.Connection, person_id: int) -> None:
    """Mirror store._recount_person: refresh denormalized counts."""
    conn.execute(
        """UPDATE people
           SET face_count = (SELECT COUNT(*) FROM faces f WHERE f.person_id = people.id),
               photo_count = (SELECT COUNT(DISTINCT f.photo_uid) FROM faces f
                              WHERE f.person_id = people.id AND f.photo_uid IS NOT NULL)
           WHERE id = ?""",
        (person_id,),
    )


def _chunked(ids: list[int], size: int = _SQL_CHUNK) -> list[list[int]]:
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move a person's faces back from a merge, using a pre-merge backup."
    )
    parser.add_argument(
        "--backup",
        type=Path,
        required=True,
        help="Path to a pre-merge backup SQLite DB (e.g. _backups/index-20260910-062505.sqlite3)",
    )
    parser.add_argument(
        "--name",
        default="Gaia Mornati",
        help="Name of the person to recover (must exist in both backup and live DB)",
    )
    parser.add_argument(
        "--live",
        type=Path,
        default=Path("/media/12tb/photos-index/index.sqlite3"),
        help="Path to the live index DB (default: /media/12tb/photos-index/index.sqlite3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and validate without writing anything",
    )
    args = parser.parse_args()

    if not args.backup.exists():
        print(f"error: backup DB not found: {args.backup}", file=sys.stderr)
        return 2
    if not args.live.exists():
        print(f"error: live DB not found: {args.live}", file=sys.stderr)
        return 2

    backup = _connect(args.backup, read_only=True)
    backup.row_factory = sqlite3.Row
    live = _connect(args.live)
    live.row_factory = sqlite3.Row

    src = _find_person(backup, args.name)
    if src is None:
        print(f"error: person {args.name!r} not found in backup DB", file=sys.stderr)
        return 2
    backup_face_ids = _face_ids_for_person(backup, src["id"])
    if not backup_face_ids:
        print(f"error: person {args.name!r} (id {src['id']}) has no faces in the backup", file=sys.stderr)
        return 2

    tgt = _find_person(live, args.name)
    if tgt is None:
        print(
            f"error: person {args.name!r} not found in live DB — recreate the person "
            "in the UI first (or pass a different --name)",
            file=sys.stderr,
        )
        return 2

    # Pre-flight: every backup face ID must exist in the live DB, and we must
    # know who currently owns each one.
    placeholders = ",".join("?" * len(backup_face_ids))
    live_rows = live.execute(
        f"SELECT id, person_id FROM faces WHERE id IN ({placeholders})",
        backup_face_ids,
    ).fetchall()
    live_by_id = {r["id"]: r["person_id"] for r in live_rows}
    missing = [fid for fid in backup_face_ids if fid not in live_by_id]
    if missing:
        print(
            f"error: {len(missing)} backup face IDs are missing from the live DB "
            f"(first: {missing[:5]}...) — aborting",
            file=sys.stderr,
        )
        return 2

    already = [fid for fid in backup_face_ids if live_by_id[fid] == tgt["id"]]
    to_move = [fid for fid in backup_face_ids if live_by_id[fid] != tgt["id"]]
    donors: dict[int, int] = {}
    for fid in to_move:
        pid = live_by_id[fid]
        if pid is not None:
            donors[pid] = donors.get(pid, 0) + 1

    print(f"backup person : {args.name!r} (id {src['id']}) — {len(backup_face_ids)} faces")
    print(f"live target   : {args.name!r} (id {tgt['id']}) — {tgt['face_count']} faces")
    print(f"  already on target : {len(already)}")
    print(f"  to move back      : {len(to_move)}")
    for pid, n in sorted(donors.items(), key=lambda kv: -kv[1]):
        print(f"  donor person {pid}: {n} faces")
    if not to_move:
        print("nothing to do: all backup faces are already on the target")
        return 0

    if args.dry_run:
        print("\ndry-run: no changes written")
        return 0

    # Snapshot the live DB before touching it (same convention as backup.sh).
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snap = args.live.parent / "_backups" / f"index-pre-recover-{stamp}.sqlite3"
    snap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.live, snap)
    print(f"\nsnapshot: {snap}")

    with live:
        # Backfill name/cover from the backup only when the target has none
        # (mirrors merge_person's backfill direction).
        if not tgt["name"] and src["name"]:
            live.execute("UPDATE people SET name=? WHERE id=?", (src["name"], tgt["id"]))
        if not tgt["cover_uid"] and src["cover_uid"]:
            live.execute("UPDATE people SET cover_uid=? WHERE id=?", (src["cover_uid"], tgt["id"]))
        if not tgt["cover_face_id"] and src["cover_face_id"]:
            live.execute(
                "UPDATE people SET cover_face_id=? WHERE id=?",
                (src["cover_face_id"], tgt["id"]),
            )
        # Re-parent the backup face IDs that are not already on the target.
        for chunk in _chunked(to_move):
            ph = ",".join("?" * len(chunk))
            live.execute(
                f"UPDATE faces SET person_id=? WHERE id IN ({ph})", [tgt["id"], *chunk]
            )
        # Recount the target and every donor.
        _recount(live, tgt["id"])
        for pid in donors:
            _recount(live, pid)

    live.close()
    backup.close()

    # Report the new state.
    live2 = _connect(args.live)
    live2.row_factory = sqlite3.Row
    for pid, label in [(tgt["id"], "target"), *[(pid, f"donor {pid}") for pid in donors]]:
        row = live2.execute(
            "SELECT face_count, photo_count FROM people WHERE id=?", (pid,)
        ).fetchone()
        print(f"after  {label:>10}: {row['face_count']} faces / {row['photo_count']} photos")
    live2.close()
    print("\ndone. Restart the indexer so it rewrites the face sidecar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

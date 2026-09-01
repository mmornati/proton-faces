"""One-shot reclaim script for the Sep 1 false-deletion cascade.

The 40,310 photos marked `status='deleted'` on Aug 30-31 during the
Frankfurt cooling failure are still in Proton Drive. We force a reclaim
by:

  1. Loading all currently-deleted uids from the local DB.
  2. Asking the bridge's `/nodes` endpoint for full metadata in batches.
  3. For each uid the bridge still knows about, calling
     `store.upsert_photos([row])` so the existing reclaim CASE clause
     resets status='deleted' -> 'new' and clears was_deleted_at.

The script is idempotent — re-running it on rows that have already
been reclaimed is a no-op (the CASE clause is gated on the current
status).

Uids the bridge no longer recognises (e.g. Proton actually deleted
them) are left as 'deleted' with was_deleted_at intact for historical
record. Run as:

  docker compose exec indexer python3 /app/src/reclaim_deleted.py

Logs progress every 500 uids and exits with the count reclaimed vs.
count still-deleted-after.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx  # noqa: E402

from config import settings  # noqa: E402
from store import get_conn, upsert_photos  # noqa: E402

log = logging.getLogger("reclaim")
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

BRIDGE_URL = settings.bridge_url
BATCH = 100  # bridge's /nodes accepts a list of uids
TIMEOUT = 120.0


def _nodes(uids: list[str]) -> list[dict]:
    """Fetch full metadata for uids via the bridge. Returns one dict per uid.

    Bridge streams the response as NDJSON (one node per line). Comment lines
    starting with '#' are progress markers and are skipped.
    """
    r = httpx.post(
        f"{BRIDGE_URL}/nodes",
        json={"uids": uids},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    items = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(json.loads(line))
    return items


def _row_from_node(node: dict) -> dict:
    """Bridge returns one node dict per uid (modificationTime, captureTime
    are ISO strings; we strip them down to what upsert_photos accepts)."""
    from datetime import datetime
    capture_time_iso = node.get("captureTime") or ""
    if capture_time_iso:
        try:
            capture_time = int(
                datetime.fromisoformat(capture_time_iso.replace("Z", "+00:00")).timestamp()
            )
        except ValueError:
            capture_time = 0
    else:
        capture_time = 0
    return {
        "uid": node["uid"],
        "name": node.get("name"),
        "media_type": node.get("mediaType"),
        "capture_time": capture_time,
        "sha1": node.get("sha1"),
        "albums": node.get("albums") or [],
        "size": node.get("size"),
    }


def main() -> int:
    with get_conn() as conn:
        deleted = [r[0] for r in conn.execute(
            "SELECT uid FROM photos WHERE status='deleted' ORDER BY capture_time"
        ).fetchall()]
    total = len(deleted)
    log.info("starting reclaim: %d uids in status='deleted'", total)
    if total == 0:
        return 0

    reclaimed = 0
    still_deleted = 0
    errors = 0
    t0 = time.time()
    for i in range(0, total, BATCH):
        chunk = deleted[i:i + BATCH]
        try:
            nodes = _nodes(chunk)
        except Exception as exc:
            log.warning("bridge /nodes batch %d failed: %s", i // BATCH, exc)
            errors += len(chunk)
            time.sleep(5)
            continue
        # Bridge returns nodes only for uids it still knows about. Build a
        # set so we can detect uids the bridge has actually forgotten.
        seen_uids = {n["uid"] for n in nodes}
        rows = [_row_from_node(n) for n in nodes]
        if rows:
            upsert_photos(rows)
            reclaimed += len(rows)
        # Uids not in `seen_uids` are truly gone from Proton (e.g. you
        # actually deleted them, or they expired). Leave as 'deleted'
        # with was_deleted_at preserved.
        not_seen = [u for u in chunk if u not in seen_uids]
        still_deleted += len(not_seen)

        if (i // BATCH) % 5 == 0:
            elapsed = time.time() - t0
            rate = (i + len(chunk)) / max(elapsed, 0.001)
            log.info(
                "  progress %d/%d (%.1f uids/sec, reclaimed=%d, still_deleted=%d, errors=%d)",
                i + len(chunk), total, rate, reclaimed, still_deleted, errors,
            )

    elapsed = time.time() - t0
    log.info(
        "done in %.1fs: reclaimed=%d, still_deleted=%d, errors=%d",
        elapsed, reclaimed, still_deleted, errors,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

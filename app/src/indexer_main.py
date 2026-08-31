"""Indexer-only entry point: runs the indexing pipeline in its own container.

No uvicorn, no FastAPI. Designed to share SQLite + DATA_DIR with the `app`
container so that face detection happens on dedicated CPU cores and the
API event loop never has to wait for recognition.

Cross-process safety: writes to `photos` / `clips` are guarded by per-row
atomic claims (`UPDATE ... WHERE status='new'`) and SQLite is in WAL mode
with a 30 s busy_timeout, so the API process and the indexer process
interleave cleanly without extra coordination.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys

from config import settings
from indexer import backfill_gps, enrich_places, start

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("indexer_main")


def main() -> None:
    parser = argparse.ArgumentParser(prog="proton-faces-indexer")
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
    args = parser.parse_args()

    if args.backfill_gps:
        matched = backfill_gps(rebuild_cache=args.rebuild_cache)
        log.info("GPS backfill done: %d photos matched", matched)
        enriched = enrich_places()
        log.info("GPS place enrichment done: %d photos", enriched)
        return

    threads = start()
    log.info("indexer pipeline up (%d background threads)", len(threads))

    # Park the main thread on a sleep loop until SIGTERM/SIGINT. uvicorn-style
    # signal handling isn't needed here because we never bind a port.
    stop = False

    def _shutdown(signum, _frame):
        log.info("received signal %s, shutting down", signum)
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while not stop:
            # Daemon threads exit automatically when the process does; we just
            # need to keep the main thread alive and forward signals.
            signal.pause()
    except (KeyboardInterrupt, SystemExit):
        pass

    log.info("indexer_main exiting")
    sys.exit(0)


if __name__ == "__main__":
    main()
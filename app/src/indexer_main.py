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
import os
import signal
import sys
import threading

import uvicorn

from config import settings
from indexer import backfill_gps, enrich_places, start
from indexer_status import app as status_app

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
        help="One-shot: attach GPS/place data from a legacy local Takeout export (PHOTOS_DIR), then exit.",
    )
    parser.add_argument(
        "--backfill-gps-exif",
        action="store_true",
        help="One-shot: re-download photos without GPS and read coordinates from their EXIF, then exit.",
    )
    parser.add_argument(
        "--gps-media-type",
        default=None,
        metavar="PREFIX",
        help="With --backfill-gps-exif: only sweep media types with this prefix (e.g. image/heic).",
    )
    parser.add_argument(
        "--gps-limit",
        type=int,
        default=0,
        help="With --backfill-gps-exif: cap the number of photos processed (0 = unlimited).",
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

    if args.backfill_gps_exif:
        from indexer import backfill_gps_exif
        matched = backfill_gps_exif(media_type=args.gps_media_type, limit=args.gps_limit)
        log.info("GPS EXIF backfill done: %d photos matched", matched)
        enriched = enrich_places()
        log.info("GPS place enrichment done: %d photos", enriched)
        return

    # DEMO_MODE: bootstrap the default admin user before starting the pipeline
    # so the web UI's login screen is reachable without manual setup.
    if os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from demo import ensure_default_admin

            ensure_default_admin()
        except Exception as exc:  # pragma: no cover
            log.warning("DEMO_MODE admin bootstrap failed: %s", exc)

    threads = start()
    log.info("indexer pipeline up (%d background threads)", len(threads))

    # Tiny internal-only HTTP server that exposes the indexer's live
    # state to the `app` container (see indexer_status.py). Bound to
    # 0.0.0.0 so the `app` service on the compose `internal` network can
    # reach it at http://indexer:8091/status. There is no host port
    # mapping for 8091 in compose.yml, so this is still not reachable
    # from outside the compose network. Every route except /healthz is
    # gated on the shared `INDEXER_TOKEN` secret (issue #42); missing
    # token outside DEMO_MODE raises RuntimeError below.
    status_port = settings.indexer_status_port

    # Fail closed if the operator forgot to set INDEXER_TOKEN — never
    # silently leave the control API open. _indexer_token() in
    # indexer_status does the same check, but validating here surfaces
    # the misconfig in the main process's log at startup instead of as
    # a 500 from the status server thread.
    import indexer_status  # local import to avoid pulling fastapi on the CLI helpers
    indexer_status._indexer_token()

    def _run_status_server() -> None:
        cfg = uvicorn.Config(
            status_app,
            host="0.0.0.0",
            port=status_port,
            log_level=settings.log_level.lower(),
            access_log=False,
        )
        uvicorn.Server(cfg).run()

    status_thread = threading.Thread(
        target=_run_status_server, name="indexer-status", daemon=True
    )
    status_thread.start()
    log.info("indexer status endpoint listening on 0.0.0.0:%d", status_port)

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

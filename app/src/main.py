"""Entry point: serves the FastAPI app (and optionally the indexer threads).

Phase 1 of issue #4: by default this process only serves uvicorn. The
indexing pipeline now runs in a separate `indexer` container (see
`indexer_main.py`) so the API event loop isn't preempted by face detection.

Set `RUN_INDEXER=1` to keep the legacy in-process indexer behaviour — useful
as a fallback while the new container is being rolled out, and for local
single-process dev setups.
"""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from api import app
from config import settings
from indexer import backfill_gps, enrich_places, start

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


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
    args = parser.parse_args()

    if args.backfill_gps:
        matched = backfill_gps(rebuild_cache=args.rebuild_cache)
        logging.getLogger(__name__).info("GPS backfill done: %d photos matched", matched)
        enriched = enrich_places()
        logging.getLogger(__name__).info("GPS place enrichment done: %d photos", enriched)
        return

    if _env_bool("RUN_INDEXER", False):
        logging.getLogger(__name__).info(
            "RUN_INDEXER=1: starting in-process indexer alongside uvicorn"
        )
        start()
    else:
        logging.getLogger(__name__).info(
            "RUN_INDEXER unset: serving API only; indexer runs in its own container"
        )

    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
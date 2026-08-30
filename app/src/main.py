"""Entry point: starts the indexer background threads and serves the API."""
from __future__ import annotations

import argparse
import logging

import uvicorn

from api import app
from config import settings
from indexer import backfill_gps, start

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="proton-faces")
    parser.add_argument(
        "--backfill-gps",
        action="store_true",
        help="One-shot: attach GPS/place data from the local Takeout export (PHOTOS_DIR), then exit.",
    )
    args = parser.parse_args()

    if args.backfill_gps:
        matched = backfill_gps()
        logging.getLogger(__name__).info("GPS backfill done: %d photos matched", matched)
        return

    start()
    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
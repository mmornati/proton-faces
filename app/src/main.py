"""Entry point: starts the indexer background threads and serves the API."""
from __future__ import annotations

import logging

import uvicorn

from api import app
from config import settings
from indexer import start

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    start()
    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
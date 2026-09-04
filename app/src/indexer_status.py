"""Tiny HTTP server exposing the indexer's live runtime state.

Runs alongside the indexer pipeline inside the `indexer` container. The
companion `app` container proxies `GET /status` to this endpoint so the
footer/health modal can show real timestamps, live queue depth, and
thread liveness — none of which would otherwise be reachable from the
separate process that owns the FastAPI app.

Internal-only: bound to 127.0.0.1 inside the indexer container, with no
host port mapping in compose.yml. Only the `app` service on the same
compose network can reach it (via the DNS name `indexer`).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

import indexer
from store import stats

log = logging.getLogger("indexer_status")

app = FastAPI(title="proton-faces-indexer-status", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe — useful for compose's healthcheck if we ever want one."""
    return {"ok": True}


@app.get("/status")
def status() -> dict:
    """Snapshot of the indexer's runtime state.

    Combines the in-memory `_runtime` snapshot (live queue depth, thread
    liveness, last-sync timestamps) with the durable SQLite pending count
    so the UI can show both metrics side by side.
    """
    rt = indexer.get_indexer_state()
    try:
        pending_db = (stats().get("photos") or {}).get("pending", 0)
    except Exception as exc:  # pragma: no cover
        log.warning("stats() failed inside indexer status: %s", exc)
        pending_db = 0
    rt["pending_db"] = int(pending_db)
    return rt


@app.post("/trigger-sync")
def trigger_sync() -> dict:
    """Ask the sync loop to run a full scan on its next iteration."""
    indexer.request_full_sync()
    return {"ok": True}


@app.get("/sync-config")
def sync_config() -> dict:
    """Return the live sync configuration (env defaults + file overrides)."""
    return indexer.get_sync_config()


@app.put("/sync-config")
def update_sync_config(body: dict) -> dict:
    """Validate, persist and return the merged sync configuration."""
    try:
        return indexer.set_sync_config(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

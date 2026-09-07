"""Tiny HTTP server exposing the indexer's live runtime state.

Runs alongside the indexer pipeline inside the `indexer` container. The
companion `app` container proxies `GET /status` to this endpoint so the
footer/health modal can show real timestamps, live queue depth, and
thread liveness — none of which would otherwise be reachable from the
separate process that owns the FastAPI app.

Network exposure: bound to 0.0.0.0 inside the indexer container so the
`app` service on the compose `internal` network can reach it via the
DNS name `indexer`. No host port is published in compose.yml, so this
endpoint is NOT reachable from outside the compose network. The only
container that should talk to it is `app` (and compose's optional
healthcheck, which only hits `/healthz`).

Authentication: every route except `GET /healthz` requires the shared
secret `INDEXER_TOKEN` in the `X-Indexer-Token` header. The token is
validated with `hmac.compare_digest` to avoid timing leaks. Outside
DEMO_MODE the indexer refuses to start without it (the `app` container
uses the same env var to authenticate its proxy calls).

Issue: #42 — indexer control API (:8091) was previously unauthenticated
and reachable from any container on the compose network.
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi import status as http_status

import indexer
from store import stats

log = logging.getLogger("indexer_status")

app = FastAPI(title="proton-faces-indexer-status", version="0.1.0")


def _indexer_token() -> bytes:
    """Return the configured `INDEXER_TOKEN` as bytes.

    The same value is plumbed through to the `app` container via compose
    and attached to every proxied request (`api._indexer_proxy_json`).
    Without an explicit token the indexer refuses to start (fail closed),
    so a missing env var can't silently leave the control API open.

    DEMO_MODE is the only path that may fall back to a per-boot random
    secret: demo traffic is public by design, but the token still must
    exist in-process so `app` and `indexer` agree on the same value
    within a single boot.
    """
    tok = os.environ.get("INDEXER_TOKEN", "").strip()
    if tok:
        return tok.encode("utf-8")
    if not _is_demo_mode():
        raise RuntimeError(
            "INDEXER_TOKEN is not set. Set it in your environment (.env) before "
            "starting the indexer, e.g. `openssl rand -hex 32`. Refusing to start: "
            "without an explicit token, the control API on :8091 would accept "
            "unauthenticated calls from any container on the compose network."
        )
    if not hasattr(_indexer_token, "_ephemeral"):
        import secrets
        _indexer_token._ephemeral = secrets.token_hex(32).encode("utf-8")  # type: ignore[attr-defined]
        log.warning(
            "INDEXER_TOKEN not set; DEMO_MODE using a per-boot ephemeral token. "
            "Set INDEXER_TOKEN explicitly so the app container can authenticate "
            "its proxy calls across restarts."
        )
    return _indexer_token._ephemeral  # type: ignore[attr-defined]


def _is_demo_mode() -> bool:
    return os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def _require_indexer_token(request: Request) -> None:
    """Constant-time check of the `X-Indexer-Token` request header.

    Returns 401 when the header is missing or doesn't match `INDEXER_TOKEN`.
    Used as a FastAPI dependency on every route except `GET /healthz`
    (which is intentionally auth-free so compose healthchecks still work
    without needing the token).
    """
    presented = request.headers.get("X-Indexer-Token", "")
    if not presented:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="missing X-Indexer-Token",
            headers={"WWW-Authenticate": "X-Indexer-Token"},
        )
    expected = _indexer_token()
    try:
        presented_bytes = presented.encode("ascii")
    except UnicodeEncodeError:
        presented_bytes = b"\x00"
    if not hmac.compare_digest(expected, presented_bytes):
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="invalid X-Indexer-Token",
            headers={"WWW-Authenticate": "X-Indexer-Token"},
        )


# Routes that touch state require the shared INDEXER_TOKEN (see
# `_require_indexer_token`). `GET /healthz` is intentionally auth-free so
# compose healthchecks (or any other in-cluster probe) can ping it
# without needing to know the secret.


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe — auth-free so compose healthchecks work without the token."""
    return {"ok": True}


@app.get("/status", dependencies=[Depends(_require_indexer_token)])
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


@app.post("/trigger-sync", dependencies=[Depends(_require_indexer_token)])
def trigger_sync() -> dict:
    """Ask the sync loop to run a full scan on its next iteration."""
    indexer.request_full_sync()
    return {"ok": True}


@app.get("/sync-config", dependencies=[Depends(_require_indexer_token)])
def sync_config() -> dict:
    """Return the live sync configuration (env defaults + file overrides)."""
    return indexer.get_sync_config()


@app.put("/sync-config", dependencies=[Depends(_require_indexer_token)])
def update_sync_config(body: dict) -> dict:
    """Validate, persist and return the merged sync configuration."""
    try:
        return indexer.set_sync_config(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

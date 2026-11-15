"""Status route handlers: health, stats, bridge_health, status.

Handlers are defined as module-level functions so they can be re-exported
from ``api.py`` (tests call them as ``api.api_health(...)`` etc.).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

import api  # noqa: E402  (intentional: see api_common docstring)
from auth import require_user

log = logging.getLogger("api")

router = APIRouter()


@router.get("/api/health", dependencies=[])
def api_health():
    """Liveness probe — always 200 if the process is up."""
    # Cached: the route is unauthenticated, so it must not turn every
    # anonymous request into a bridge round trip.
    try:
        reachable, logged_in = api._cached_bridge_health()
    except Exception:
        reachable, logged_in = False, False
    return {"ok": True, "bridge": {"reachable": reachable, "loggedIn": logged_in}}


@router.get("/api/stats")
def api_stats(_: object = Depends(require_user)):
    """Indexing stats (counts, last sync, etc.)."""
    return api._cached_stats()


@router.get("/api/bridge_health")
def api_bridge_health(_: object = Depends(require_user)):
    """Bridge health snapshot."""
    return api._cached_bridge_health()


@router.get("/api/status", dependencies=[])
def api_status(request: Request) -> dict:
    """Aggregated status snapshot for the bottom status bar / details overlay."""
    import time as _time
    try:
        bridge_ok, bridge_logged_in = api._cached_bridge_health()
    except Exception:
        bridge_ok = False
        bridge_logged_in = False
    out = {"now": _time.time(), "bridge": {"reachable": bridge_ok, "loggedIn": bridge_logged_in}}
    # Anonymous callers (compose healthcheck, the login screen) get reachability
    # only. Library size, disk usage, thread state and raw sync error strings
    # describe a private archive and its internals, so they need a live token.
    try:
        authed = api._bearer_is_valid(request)
    except Exception:
        authed = False
    if not authed:
        return out
    s = api._cached_stats()
    rt = api._merged_indexer_state()
    disk = rt.get("disk") or {}
    if "thumb_dir_bytes" in disk:
        thumbs_bytes = disk["thumb_dir_bytes"]
    else:
        from admin import cached_dir_size
        thumbs_bytes = cached_dir_size(api.settings.thumb_dir)
    if "db_bytes" in disk:
        db_bytes = disk["db_bytes"]
    else:
        db_bytes = api.settings.db_path.stat().st_size if api.settings.db_path.exists() else 0
    out.update({
        "stats": s,
        "indexer": rt,
        "disk": {
            "thumb_dir_bytes": thumbs_bytes,
            "db_bytes": db_bytes,
        },
        "config": {
            "sync_interval": api.settings.sync_interval,
            "cluster_interval": api.settings.cluster_interval,
            "gps_interval": api.settings.gps_interval,
            "workers": api.settings.workers,
            "face_sim_threshold": api.settings.face_sim_threshold,
            "min_cluster_size": api.settings.min_cluster_size,
            "photos_dir": api.settings.photos_dir or None,
        },
    })
    return out

"""FastAPI application: search API + static web UI."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
from fastapi import Body, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import clip
import faces
import indexer
from auth import (
    ROLE_RANK,
    CurrentUser,
    allow_public_thumbs,
    demo_disable_admin_user_management,
    demo_disable_backups,
    hash_password,
    login,
    make_signed_token,
    require_role,
    require_signing_secret,
    require_user,
    signed_or_token,
)
from auth import (
    access_ttl as auth_access_ttl,
)
from auth import (
    refresh as refresh_tokens,
)
from auth import (
    refresh_ttl as auth_refresh_ttl,
)
from bridge_client import (
    BridgeTransientError,
    get_bridge,
    is_valid_uid,
    uid_invalid_reason,
)
from clip import embed_text
from compression import CompressionMiddleware
from config import settings
from faces import embed_query_face
from indexer import get_indexer_state
from sidecar import read_clip_sidecar
from store import (
    _embedding_cache_data,
    album_names,
    album_photos,
    all_albums,
    all_clips,
    all_people,
    all_tags,
    archived_photos,
    assign_face_person,
    clip_count,
    count_faces_for_person,
    create_person,
    create_user,
    delete_empty_people,
    delete_user,
    done_photos,
    duplicate_groups,
    face_embedding,
    face_ids_for_people,
    faces_for_person,
    faces_for_photo,
    favorite_photo,
    favorite_uids,
    find_person_by_name,
    get_person,
    get_photo,
    get_photos_batch,
    get_tags,
    get_user_by_id,
    get_user_by_username,
    is_favorite,
    list_users,
    map_markers,
    memories_for_today,
    merge_people_bulk,
    merge_person,
    people_by_ids,
    person_map_markers,
    person_mean_embedding,
    person_mean_embeddings_from_cache,
    person_mean_matrix_from_cache,
    photo_anchors,
    photos_by_tag,
    photos_for_person,
    place_stats,
    rename_person,
    revoke_all_tokens,
    revoke_token,
    search_photos_by_place,
    set_archived,
    set_hidden,
    set_person_cover_face,
    set_tags,
    similar_faces,
    stats,
    unassign_face,
    unassigned_faces,
    unfavorite_photo,
    update_user,
)

log = logging.getLogger("api")

# P-01: in production, hide the interactive API docs (Swagger UI + ReDoc)
# and the raw OpenAPI schema. The default `DEMO_HARDENING_MODE=1` flips this
# on for public demos; production deployments with private credentials
# get the same default. Set `EXPOSE_API_DOCS=1` to keep the docs visible
# (useful when the operator wants to share the schema with their own
# front-end team behind a separate auth gate).
_EXPOSE_API_DOCS = os.environ.get("EXPOSE_API_DOCS", "").strip().lower() in (
    "1", "true", "yes", "on",
)

@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Fail closed before serving traffic.

    Signed binary URLs require an explicit SIGNING_SECRET outside DEMO_MODE
    (a known default is forgerable and an ephemeral per-worker secret breaks
    signed URLs across uvicorn workers). Raises so the process refuses to
    boot instead of serving 500s on the first /thumb request.

    After the fail-closed check, each uvicorn worker pre-loads the ML
    sessions (issue #97) so the first /api/search and /api/search/face
    don't pay a multi-second model load inside a user request.
    """
    require_signing_secret()
    warm_models()
    yield


def warm_models() -> None:
    """Pre-load the CLIP + InsightFace sessions once per worker process.

    Every uvicorn worker lazily loads its own sessions on the first user
    request, which after a deploy/restart turns /api/search and
    /api/search/face into multi-second latency spikes and repeats the
    weight loads N workers times. Called from the lifespan (once per
    worker process) before traffic is served. Failures are logged and
    swallowed so a missing model never blocks boot — the lazy path remains
    as fallback. Gated by WARM_MODELS (default on; off in tests/CLI).
    """
    if not settings.warm_models:
        log.info("WARM_MODELS=0: skipping ML session warm-up")
        return
    for name, fn in (
        ("clip", clip.warm_up),
        ("insightface", faces.warm_up),
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - warm-up must never block boot
            log.warning("warm-up for %s failed (lazy load remains as fallback): %s", name, exc)


app = FastAPI(
    title="proton-faces",
    version="0.1.0",
    dependencies=[Depends(require_user)],
    lifespan=_lifespan,
    docs_url="/docs" if _EXPOSE_API_DOCS else None,
    redoc_url="/redoc" if _EXPOSE_API_DOCS else None,
    openapi_url="/openapi.json" if _EXPOSE_API_DOCS else None,
)

app.add_middleware(CompressionMiddleware, minimum_size=1024)

_STATIC = Path(__file__).parent / "static"

# TTL cache for the (expensive) people-duplicates computation. The blockwise
# pairwise scan over ~30 k person-means takes ~40 s, so it is cached well
# beyond a single page view — both the 1:1 `/duplicates` endpoint and the
# person-centric `/suggested-merges` discovery share this budget.
_DUP_CACHE_TTL = 600.0
_dups_cache: tuple[float, dict] | None = None
_dups_cache_lock = threading.Lock()

# Person-centric view of the same scan, keyed by threshold (each threshold
# materialises its own ranked row list once, then requests only slice it).
_SUGGESTED_CACHE_TTL = 600.0
_suggested_cache: dict[float, tuple[float, list]] = {}
_suggested_cache_lock = threading.Lock()

# TTL cache for the photo-duplicates endpoint (issue #90). Groups are
# user-independent (favorites are merged per request), so one entry per
# `limit` page-size serves every authenticated user.
_PHOTO_DUPS_CACHE_TTL = 30.0
_photo_dups_cache: tuple[float, dict[int, list]] | None = None
_photo_dups_cache_lock = threading.Lock()

# TTL caches (cheap, frequently re-requested on navigation).
_ANCHORS_CACHE_TTL = 60.0
_anchors_cache: tuple[float, dict] | None = None

_PEOPLE_CACHE_TTL = 10.0
_PEOPLE_CACHE_MAX = 32
# Small LRU keyed by the name prefix (None = the unfiltered full list).
# Typeahead keystrokes that move forward/back between prefixes hit warm
# entries instead of re-running the people query on every keystroke.
_people_cache: OrderedDict[str | None, tuple[float, list]] = OrderedDict()
_people_cache_lock = threading.Lock()

# Hard cap on how long `/api/photos/{uid}/full` is allowed to take before we
# give up and return 504 to the user. The bridge's /photo/{uid}/full endpoint
# occasionally hangs for 30+ seconds when Proton's downloader endpoint is
# degraded; without this cap the FastAPI handler blocks indefinitely and the
# browser's loading spinner never resolves. We use an async semaphore for
# admission control (doesn't consume a thread) and enforce the timeout via
# asyncio.wait_for so the event loop stays responsive.
_FULL_TIMEOUT_SEC = 30.0
_FULL_SEMAPHORE_MAX = int(os.environ.get("FULL_SEMAPHORE_MAX", "8"))
_full_semaphore = asyncio.Semaphore(_FULL_SEMAPHORE_MAX)

# Rolling log of /full proxy failures (504 timeouts + 502 bridge errors +
# BridgeTransientErrors). The admin "Bridge cache" check consults this so
# that an idle bridge isn't flagged just because its SDK caches happen to
# be old — we only flag stale when full-res is also failing, which is the
# signature of the getFileDownloader-waitForCondition2 hang.
_FULL_RES_FAILURE_WINDOW_SEC = 15 * 60
_full_res_failure_ts: list[float] = []
_FULL_RES_FAILURE_LOG_LOCK = threading.Lock()


def _record_full_res_failure() -> None:
    cutoff = time.time() - _FULL_RES_FAILURE_WINDOW_SEC
    with _FULL_RES_FAILURE_LOG_LOCK:
        _full_res_failure_ts.append(time.time())
        # prune in place — cheap because the list stays tiny
        if _full_res_failure_ts and _full_res_failure_ts[0] < cutoff:
            _full_res_failure_ts[:] = [t for t in _full_res_failure_ts if t >= cutoff]


def _recent_full_res_failures() -> int:
    cutoff = time.time() - _FULL_RES_FAILURE_WINDOW_SEC
    with _FULL_RES_FAILURE_LOG_LOCK:
        # prune as we read so memory stays bounded
        while _full_res_failure_ts and _full_res_failure_ts[0] < cutoff:
            _full_res_failure_ts.pop(0)
        return len(_full_res_failure_ts)

# In-memory CLIP matrix cache: avoids rebuilding an 88 MB numpy stack on
# every text-search request. Rebuilds only when the clip row count changes
# or after TTL.
_CLIP_CACHE_TTL = 60.0

# Cache for stats() and disk-size walks so the periodic /api/status poll
# doesn't re-pay for 4× COUNT(*) + GROUP BY + a 44 k stat() walk every 15 s.
_STATS_CACHE_TTL = 5.0
_stats_cache: tuple[float, dict] | None = None
_stats_cache_lock = threading.Lock()
_DIRSIZE_CACHE_TTL = 300.0
_dirsize_cache: dict[str, tuple[float, int]] = {}
_clip_cache: tuple[float, int, list[str], np.ndarray] | None = None
_clip_cache_lock = threading.Lock()

# Disk + lock for face crops (computed lazily, then served as plain files).
_crop_lock = threading.Lock()

_IMMUTABLE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}


def _extract_bearer(request: Request) -> str | None:
    """Pull the bearer token out of the Authorization header, if present.

    Used by endpoints that need to know "is there any auth here?" without
    going through FastAPI's dependency machinery (e.g. `/api/status` to
    decide whether to include the config block).
    """
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(None, 1)[1].strip()
    return token or None


def _invalidate_dups_cache() -> None:
    global _dups_cache, _suggested_cache
    _dups_cache = None
    _suggested_cache = {}


def _invalidate_photo_dups_cache() -> None:
    global _photo_dups_cache
    _photo_dups_cache = None


def _invalidate_people_cache() -> None:
    global _people_cache
    _people_cache = OrderedDict()


def _invalidate_clip_cache() -> None:
    global _clip_cache
    _clip_cache = None


def _crop_cache_path(face_id: int) -> Path:
    return settings.crops_dir / f"{face_id}.jpg"


def _drop_crop_cache(face_id: int) -> None:
    """Remove a single cached face crop (e.g. after the face was assigned)."""
    with _crop_lock:
        try:
            _crop_cache_path(face_id).unlink(missing_ok=True)
        except OSError:
            pass


def _drop_person_crops(person_id: int) -> None:
    """Remove cached face crops for every face belonging to a person."""
    from store import get_conn

    with get_conn() as conn:
        face_ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM faces WHERE person_id=?", (person_id,)
            ).fetchall()
        ]
    if not face_ids:
        return
    with _crop_lock:
        for fid in face_ids:
            try:
                _crop_cache_path(fid).unlink(missing_ok=True)
            except OSError:
                pass


def _drop_people_crops(person_ids: list[int]) -> None:
    """Remove cached face crops for every face of many people (single query).

    Bulk variant of `_drop_person_crops` so a many-source merge drops all crop
    files with one chunked SELECT instead of one connection per person.
    """
    face_ids = face_ids_for_people(person_ids)
    if not face_ids:
        return
    with _crop_lock:
        for fid in face_ids:
            try:
                _crop_cache_path(fid).unlink(missing_ok=True)
            except OSError:
                pass


# --- helpers ---------------------------------------------------------------

def _sign_if_needed(url: str | None) -> str | None:
    """Append ?sig=&exp= to a binary endpoint URL when prod-mode is on.

    Returns the URL unchanged when DEMO_ALLOW_PUBLIC_THUMBS=1 (or when the
    URL is None). Used by every endpoint that hands a /thumb /full /cover
    /crop URL back to the front-end so <img src=...> just works.
    """
    if not url or allow_public_thumbs():
        return url
    sig, exp = make_signed_token(url, ttl_seconds=300)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sig={sig}&exp={exp}"


def _row_to_dict(row) -> dict:
    d = dict(row)
    d.pop("embedding", None)
    if d.get("thumb_path"):
        d["thumb_url"] = _sign_if_needed(f"/api/photos/{d['uid']}/thumb")
    else:
        d["thumb_url"] = None
    mt = d.get("media_type") or ""
    if mt.startswith("video/"):
        d["kind"] = "video"
    elif mt.startswith("image/"):
        d["kind"] = "image"
    else:
        d["kind"] = "other"
    # Surface local-only metadata flags + tags as plain JSON-friendly values.
    # `favorited` is the legacy "anyone starred this" boolean (always 0 after
    # PR-9, but kept for backward-compat). `favorited_by_me` is added per-list
    # in `_user_photos()` so list endpoints don't issue N per-row queries.
    d["favorited"] = bool(d.get("favorited"))
    d["favorited_by_me"] = False
    d["archived"] = bool(d.get("archived"))
    d["hidden"] = bool(d.get("hidden"))
    raw_tags = d.get("tags")
    if raw_tags:
        try:
            d["tags"] = list(json.loads(raw_tags))
        except Exception:
            d["tags"] = []
    else:
        d["tags"] = []
    return d


def _user_photos(user_id: int, rows, fav_set: set[str] | None = None) -> list[dict]:
    """Serialize a list of photo rows for `user_id`, marking favorited_by_me.

    Issues a single batched query against user_favorites for the page of uids
    so list endpoints stay O(1) round-trips. A precomputed `fav_set` skips
    that query so callers can batch one favorites lookup across many row
    groups (used by the Duplicates endpoint).
    """
    if fav_set is None:
        uids = [r["uid"] for r in rows]
        fav_set = favorite_uids(user_id, uids) if uids else set()
    out = []
    for r in rows:
        d = _row_to_dict(r)
        d["favorited_by_me"] = r["uid"] in fav_set
        out.append(d)
    return out


def _single_user_photo(user_id: int, row) -> dict:
    d = _row_to_dict(row)
    d["favorited_by_me"] = is_favorite(user_id, row["uid"])
    return d


# --- public auth endpoints -------------------------------------------------
# Declared first so they don't accidentally inherit the global require_user
# dep. Each route sets dependencies=[] explicitly so it stays public even if
# route ordering changes.

# FP-1: the refresh token lives in an HttpOnly cookie so it can't be exfiltrated
# by XSS (localStorage theft was the pre-fix vector). SameSite=Strict blocks
# cross-site sends; the SPA additionally gates all state-changing calls behind
# a custom header, so the CSRF surface is closed for the cookie path.
_REFRESH_COOKIE = "pf_refresh"


def _set_refresh_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(
        _REFRESH_COOKIE,
        token,
        max_age=auth_refresh_ttl(),
        httponly=True,
        samesite="strict",
        secure=settings.auth_cookie_secure,
        path="/",
    )


def _clear_refresh_cookie(resp: Response) -> None:
    resp.delete_cookie(_REFRESH_COOKIE, path="/", samesite="strict")


@app.post("/api/auth/login", dependencies=[])
def api_login(request: Request, body: dict = Body(...)):
    """Exchange username+password for an access+refresh token pair.

    The refresh token is returned in the JSON body (backwards-compatible)
    AND set as an HttpOnly SameSite=Strict cookie (FP-1). The cookie is the
    primary channel for the browser SPA; the body field exists for API
    consumers and the hardening script.
    """
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if not username or not password:
        raise HTTPException(400, "username and password required")
    ua = request.headers.get("user-agent")
    ip = request.client.host if request.client else None
    access, refresh, user = login(username, password, user_agent=ua, ip=ip)
    resp = JSONResponse({
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_in": auth_access_ttl(),
        "user": {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
        },
    })
    _set_refresh_cookie(resp, refresh)
    return resp


@app.post("/api/auth/refresh", dependencies=[])
def api_refresh(request: Request, body: dict = Body(default={})):
    """Issue a new (access, refresh) pair. The refresh token is **rotated**
    on every successful call (P-02 from the 2026-09-01 pen test):
    the old refresh token is revoked and a new one is minted. The front-end
    MUST overwrite its stored refresh token with the new one.

    The refresh token is read from the HttpOnly cookie (FP-1) with a
    fallback to the JSON body for backwards compatibility with API
    consumers and the hardening script.
    """
    rt = (request.cookies.get(_REFRESH_COOKIE) or "").strip()
    if not rt:
        rt = (body.get("refresh_token") or "").strip()
    if not rt:
        raise HTTPException(400, "refresh_token required")
    ua = request.headers.get("user-agent")
    ip = request.client.host if request.client else None
    access, new_refresh, user = refresh_tokens(rt, user_agent=ua, ip=ip)
    resp = JSONResponse({
        "access_token": access,
        "refresh_token": new_refresh,
        "token_type": "Bearer",
        "expires_in": auth_access_ttl(),
        "user": {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
        },
    })
    _set_refresh_cookie(resp, new_refresh)
    return resp


@app.post("/api/auth/logout")
def api_logout(request: Request, user: CurrentUser = Depends(require_user)):
    """Invalidate the bearer token used for this request.

    Also revokes the refresh token (F-08) and clears the refresh cookie
    (FP-1) so a logout ends the whole session, not just the access token.
    """
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.split(None, 1)[1].strip() if auth_header.lower().startswith("bearer ") else None
    if token:
        revoke_token(token)
    rt = (request.cookies.get(_REFRESH_COOKIE) or "").strip()
    if rt:
        revoke_token(rt)
    resp = JSONResponse({"ok": True})
    _clear_refresh_cookie(resp)
    return resp


@app.get("/api/auth/me")
def api_me(user: CurrentUser = Depends(require_user)):
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
    }


@app.get("/api/auth/limits", dependencies=[])
def api_limits():
    """Public — UI uses this to render the login screen with the right labels."""
    return {"min_username": 2, "min_password": 8}


@app.post("/api/sign")
def api_sign(request: Request,
              body: dict = Body(...),
              user: CurrentUser = Depends(require_user)):
    """Issue short-lived signed URLs for binary endpoints (/thumb /full /cover /crop).

    Body: {"paths": ["/api/photos/<uid>/thumb", "/api/photos/<uid>/full", ...]}
    Returns: {"urls": [{"path": "...", "sig": "...", "exp": 1234567890}, ...]}

    The signed URL is world-readable until the next hour boundary — up to
    ~60 minutes, with a minimum lifetime of ``ttl`` seconds. Hour-quantized
    expiry keeps the ``?sig=&exp=`` pair byte-identical for the rest of the
    current hour so the binary endpoints' ``Cache-Control: immutable`` headers
    actually get used (otherwise every page load would mint a new URL and
    force a re-download). This lets the front-end embed
    <img src="/api/photos/{uid}/thumb?sig=...&exp=..."> without ever needing
    to attach an Authorization header to a static tag.

    In DEMO_ALLOW_PUBLIC_THUMBS=1 mode, signing is optional; the binary
    endpoints stay world-readable even without a signature. In prod mode
    (DEMO_ALLOW_PUBLIC_THUMBS=0), signing is the only way to embed <img>
    assets without a same-origin fetch first.
    """
    paths = body.get("paths")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(400, "paths must be a non-empty list")
    ttl = 300
    if isinstance(body.get("ttl"), int):
        ttl = max(30, min(3600, body["ttl"]))
    out = []
    for p in paths:
        if not isinstance(p, str) or not p.startswith("/api/"):
            raise HTTPException(400, f"invalid path: {p!r}")
        # Whitelist the binary endpoint suffixes.
        if not any(p.endswith(s) for s in ("/thumb", "/full", "/cover", "/crop")):
            raise HTTPException(400, f"path not signable: {p!r}")
        sig, exp = make_signed_token(p, ttl_seconds=ttl)
        out.append({"path": p, "sig": sig, "exp": exp})
    return {"urls": out, "ttl": ttl}


# --- public status (so the login screen can show "bridge online" before auth) ---

@app.get("/api/health", dependencies=[])
def health() -> dict:
    try:
        b = get_bridge().health()
        bridge_ok = bool(b.get("ok"))
        bridge_logged_in = bool(b.get("loggedIn"))
    except Exception as exc:
        bridge_ok = False
        bridge_logged_in = False
        log.warning("bridge health failed: %s", exc)
    return {
        "ok": True,
        "bridge": {"reachable": bridge_ok, "loggedIn": bridge_logged_in},
    }


@app.get("/api/stats")
def api_stats() -> dict:
    return stats()


def _dir_size_bytes(path: Path) -> int:
    """Cheap directory size in bytes (sum of immediate children). Best-effort."""
    if not path.exists():
        return 0
    total = 0
    try:
        for entry in path.iterdir():
            try:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += _dir_size_bytes(entry)
            except OSError:
                continue
    except OSError:
        return total
    return total


def _cached_dir_size(path: Path) -> int:
    """Disk-walk with a 30 s TTL — the thumb dir has tens of thousands of
    files and a full `stat()` walk is expensive when polled every 15 s."""
    key = str(path)
    now = time.time()
    hit = _dirsize_cache.get(key)
    if hit is not None and now - hit[0] < _DIRSIZE_CACHE_TTL:
        return hit[1]
    n = _dir_size_bytes(path)
    _dirsize_cache[key] = (now, n)
    return n


def _cached_stats() -> dict:
    """Cached `stats()` so the periodic `/api/status` poll doesn't pay for
    4 COUNT(*) + 1 GROUP BY on every request. 5 s TTL is well under the
    user-perceived staleness of the bottom status bar."""
    global _stats_cache
    now = time.time()
    if _stats_cache is not None and now - _stats_cache[0] < _STATS_CACHE_TTL:
        return _stats_cache[1]
    with _stats_cache_lock:
        now = time.time()
        if _stats_cache is not None and now - _stats_cache[0] < _STATS_CACHE_TTL:
            return _stats_cache[1]
        payload = stats()
        _stats_cache = (now, payload)
        return payload


# Cached bridge health so the periodic /api/status poll doesn't pay a
# bridge HTTP round-trip (measured ~100-300 ms) on every request.
_BRIDGE_HEALTH_CACHE_TTL = 30.0
_bridge_health_cache: tuple[float, tuple[bool, bool]] | None = None
_bridge_health_cache_lock = threading.Lock()


def _cached_bridge_health() -> tuple[bool, bool]:
    """Cached (reachable, logged_in)."""
    global _bridge_health_cache
    now = time.time()
    if _bridge_health_cache is not None and now - _bridge_health_cache[0] < _BRIDGE_HEALTH_CACHE_TTL:
        return _bridge_health_cache[1]
    with _bridge_health_cache_lock:
        now = time.time()
        if _bridge_health_cache is not None and now - _bridge_health_cache[0] < _BRIDGE_HEALTH_CACHE_TTL:
            return _bridge_health_cache[1]
        try:
            b = get_bridge().health()
            state = (bool(b.get("ok")), bool(b.get("loggedIn")))
        except Exception as exc:
            log.warning("bridge health failed: %s", exc)
            state = (False, False)
        _bridge_health_cache = (now, state)
        return state


# Proxy cache for the indexer container's live runtime state. The API
# process never runs the indexer itself (RUN_INDEXER=0 by default), so
# `get_indexer_state()` would otherwise return an empty stub. Instead we
# HTTP GET the dedicated /status endpoint that the indexer container
# exposes on its internal network. The frontend polls /api/status every
# 60 s; status is not realtime-critical, so a 30 s TTL means at most one
# proxied round-trip per poll instead of a fresh HTTP call every 2 s.
_INDEXER_PROXY_CACHE_TTL = 30.0
_INDEXER_PROXY_TIMEOUT = 3.0  # seconds; the endpoint computes store.stats() and can be slow under load
_indexer_proxy_cache: tuple[float, dict] | None = None
# Once the indexer proxy starts failing, suppress repeat warning logs
# for this many seconds. The frontend still renders the empty stub
# either way, and the cache TTL keeps the failure state sticky.
_INDEXER_PROXY_LOG_THROTTLE = 30.0
_indexer_proxy_last_warn: float = 0.0

# Pooled HTTP client for the indexer-status proxy (mirrors the
# `bridge_client._bridge` singleton pattern). The old urllib path opened a
# fresh TCP connection per call; the API container polls this endpoint
# every /api/status refresh, so keep-alive reuse matters. Short timeout:
# a hung indexer must degrade to the local stub, not stall the poll.
_indexer_proxy_client: httpx.Client | None = None
_indexer_proxy_client_lock = threading.Lock()


def _get_indexer_proxy_client() -> httpx.Client:
    global _indexer_proxy_client
    if _indexer_proxy_client is None:
        with _indexer_proxy_client_lock:
            if _indexer_proxy_client is None:
                _indexer_proxy_client = httpx.Client(
                    timeout=httpx.Timeout(_INDEXER_PROXY_TIMEOUT),
                    headers={"Accept": "application/json"},
                )
    return _indexer_proxy_client


def _indexer_is_local() -> bool:
    """True when this process is running the indexer threads (RUN_INDEXER=1)."""
    return bool(getattr(indexer, "_runtime", {}).get("threads"))


def _empty_indexer_state(pending_db: int | None = None) -> dict:
    """Fallback stub used when the indexer container is unreachable.

    Returns the same shape as `get_indexer_state()` so the frontend can
    keep rendering, plus an optional `pending_db` count read straight
    from SQLite (still useful when the indexer is down).
    """
    out = {
        "started_at": None,
        "last_sync": None,
        "last_sync_error": None,
        "last_cluster": None,
        "last_gps": None,
        "pending_in_queue": 0,
        "threads": {},
        "remote": True,
    }
    if pending_db is not None:
        out["pending_db"] = int(pending_db)
    return out


def _fetch_remote_indexer_state() -> dict:
    """Proxy the indexer container's /status endpoint (cached + timeout).

    On any failure (DNS, refused, timeout, non-2xx, bad JSON) returns the
    empty stub augmented with `pending_db` from local SQLite so the
    durable count is still surfaced, and tags the payload with
    `proxy_ok=False` so the UI can surface the failure instead of
    showing a silent "—".

    The indexer's /status payload already carries the durable `pending_db`
    count (computed in the indexer process), so the fresh/valid path never
    runs the local `stats()` queries. Degraded mode falls back to
    `_cached_stats()`, so `pending_db` reuses the same cached computation
    as the response's `stats` block — at most one `stats()` per request.
    """
    global _indexer_proxy_cache, _indexer_proxy_last_warn
    now = time.time()
    if _indexer_proxy_cache is not None and now - _indexer_proxy_cache[0] < _INDEXER_PROXY_CACHE_TTL:
        return _indexer_proxy_cache[1]
    url = settings.indexer_status_url.rstrip("/") + "/status"
    headers = {}
    token = (settings.indexer_token or "").strip()
    if token:
        headers["X-Indexer-Token"] = token
    try:
        resp = _get_indexer_proxy_client().get(url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        if now - _indexer_proxy_last_warn >= _INDEXER_PROXY_LOG_THROTTLE:
            log.warning("indexer status proxy failed (%s): %s", url, exc)
            _indexer_proxy_last_warn = now
        try:
            pending_db = (_cached_stats().get("photos") or {}).get("pending", 0)
        except Exception:
            pending_db = None
        payload = _empty_indexer_state(pending_db=pending_db)
        payload["proxy_ok"] = False
        payload["proxy_error"] = type(exc).__name__
    payload.setdefault("pending_db", 0)
    payload.setdefault("pending_in_queue", 0)
    payload.setdefault("proxy_ok", True)
    _indexer_proxy_cache = (now, payload)
    return payload


def _indexer_proxy_json(method: str, path: str, body: dict | None = None) -> dict:
    """Proxy a JSON request to the indexer container's status HTTP server.

    Used by the admin sync-control endpoints. Raises HTTPException(502) on any
    transport, HTTP or parse failure so the admin UI gets a clean error.
    Auth: every proxied call carries the shared `INDEXER_TOKEN` in the
    `X-Indexer-Token` header (issue #42). A missing/empty token is a
    configuration bug — we surface it as 502 to the admin UI rather than
    letting the indexer return 401, since the admin can't fix that without
    a deployment fix anyway.
    """
    url = settings.indexer_status_url.rstrip("/") + path
    headers = {}
    token = (settings.indexer_token or "").strip()
    if token:
        headers["X-Indexer-Token"] = token
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        resp = _get_indexer_proxy_client().request(method, url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise HTTPException(502, f"indexer proxy {method} {path} failed: {exc}")


def _merged_indexer_state() -> dict:
    """Return the best indexer state for `/api/status`.

    - In-process (RUN_INDEXER=1): local `_runtime` snapshot.
    - Otherwise: proxy the indexer container's /status endpoint.
    """
    if _indexer_is_local():
        return get_indexer_state()
    return _fetch_remote_indexer_state()


@app.get("/api/status", dependencies=[])
def api_status(request: Request) -> dict:
    """Aggregated status snapshot for the bottom status bar / details overlay.

    Combines bridge health, indexer stats, runtime state (thread liveness,
    last-sync timestamps, pending queue), and data-dir disk usage. Designed
    to be cheap to poll every few seconds.

    The `config` block leaks operational details (sync_interval, workers,
    face_sim_threshold, photos_dir) useful for follow-on recon. Hide it
    behind auth: anonymous callers get everything except `config`.
    """
    try:
        bridge_ok, bridge_logged_in = _cached_bridge_health()
    except Exception:
        bridge_ok = False
        bridge_logged_in = False
    s = _cached_stats()
    rt = _merged_indexer_state()
    thumbs_bytes = _cached_dir_size(settings.thumb_dir)
    db_bytes = settings.db_path.stat().st_size if settings.db_path.exists() else 0
    out = {
        "now": time.time(),
        "bridge": {"reachable": bridge_ok, "loggedIn": bridge_logged_in},
        "stats": s,
        "indexer": rt,
        "disk": {
            "thumb_dir_bytes": thumbs_bytes,
            "db_bytes": db_bytes,
        },
    }
    # Only authenticated callers see the operational config block.
    try:
        if _extract_bearer(request):
            out["config"] = {
                "sync_interval": settings.sync_interval,
                "cluster_interval": settings.cluster_interval,
                "gps_interval": settings.gps_interval,
                "workers": settings.workers,
                "face_sim_threshold": settings.face_sim_threshold,
                "min_cluster_size": settings.min_cluster_size,
                "photos_dir": settings.photos_dir or None,
            }
    except Exception:
        pass
    return out


# --- photos ----------------------------------------------------------------

@app.get("/api/photos")
def api_photos(limit: int = 200, offset: int = 0, place: str | None = None,
               before: int | None = None, only_favorites: bool = False,
               include_archived: bool = True, tag: str | None = None,
               user: CurrentUser = Depends(require_user)):
    if tag:
        rows = photos_by_tag(tag, limit=limit, offset=offset)
    elif place:
        rows = search_photos_by_place(place, limit=limit, offset=offset)
    else:
        rows = done_photos(limit=limit, offset=offset, before=before,
                            only_favorites=only_favorites,
                            include_archived=include_archived,
                            user_id=user.id)
    return {"photos": _user_photos(user.id, rows)}


@app.get("/api/photos/archived")
def api_archived_photos(limit: int = 200, offset: int = 0,
                         user: CurrentUser = Depends(require_user)):
    return {"photos": _user_photos(user.id, archived_photos(limit=limit, offset=offset))}


@app.get("/api/memories")
def api_memories(month: int | None = None, day: int | None = None, limit: int = 60,
                  user: CurrentUser = Depends(require_user)):
    """Photos captured on (month, day) in previous years — "on this day".

    Defaults to today's calendar date in UTC so the UI can just call the
    endpoint without arguments. Each result includes ``age_days`` so the UI
    can render "5 years ago today".
    """
    import datetime as _dt
    now = _dt.datetime.utcnow()
    m = month if month is not None else now.month
    d = day if day is not None else now.day
    rows = memories_for_today(m, d, limit=limit)
    photos = _user_photos(user.id, rows)
    for photo, r in zip(photos, rows):
        age_days = int(r["age_days"]) if r["age_days"] is not None else None
        photo["age_days"] = age_days
        photo["age_years"] = int(age_days // 365) if age_days is not None else None
    return {"month": m, "day": d, "photos": photos}


def _duplicate_groups_cached(limit: int) -> list[list]:
    """Photo-duplicate groups with a short single-flight TTL cache.

    Double-checked locking keyed by `limit` so concurrent requests share one
    store round-trip while different page sizes keep their own entries. The
    groups are user-independent — the endpoint merges per-user favorites on
    top, so one entry serves every authenticated caller.
    """
    global _photo_dups_cache
    now = time.time()
    cache = _photo_dups_cache
    if cache is not None and now - cache[0] < _PHOTO_DUPS_CACHE_TTL and limit in cache[1]:
        return cache[1][limit]
    with _photo_dups_cache_lock:
        now = time.time()
        cache = _photo_dups_cache
        if cache is not None and now - cache[0] < _PHOTO_DUPS_CACHE_TTL and limit in cache[1]:
            return cache[1][limit]
        if cache is None or now - cache[0] >= _PHOTO_DUPS_CACHE_TTL:
            cache = (now, {})
            _photo_dups_cache = cache
        groups = duplicate_groups(limit=limit)
        cache[1][limit] = groups
        return groups


@app.get("/api/duplicates")
def api_duplicates(limit: int = 200, user: CurrentUser = Depends(require_user)):
    """Groups of photos that share a Proton content-hash (sha1).

    Each group is rendered side-by-side in the Duplicates tab; users can
    hide individual copies (``hidden=1``) so they don't re-appear.

    One store round-trip returns every group with its members (self-join,
    issue #90) and the favorite flags come from a single batched query over
    all the returned uids — not one per-group query. Groups themselves are cached
    for `_PHOTO_DUPS_CACHE_TTL` seconds.
    """
    groups = _duplicate_groups_cached(limit)
    all_uids = [r["uid"] for members in groups for r in members]
    fav_set = favorite_uids(user.id, all_uids) if all_uids else set()
    out = []
    for members in groups:
        out.append({
            "sha1": members[0]["sha1"],
            "count": len(members),
            "photos": _user_photos(user.id, members, fav_set=fav_set),
        })
    return {"groups": out}


@app.get("/api/tags")
def api_tags():
    return {"tags": [{"name": r[0], "count": r[1]} for r in all_tags()]}


@app.patch("/api/photos/{uid}")
def api_patch_photo(uid: str, body: dict = Body(...),
                     user: CurrentUser = Depends(require_role("write"))):
    """Set local-only metadata flags on a photo: favorited, archived, hidden.

    Body keys are all optional; only the provided ones are updated. Returns
    the updated photo row.

    `favorited` is per-user (stored in user_favorites); `archived` and `hidden`
    remain shared so the family can keep a single archive view.
    """
    if get_photo(uid) is None:
        raise HTTPException(404, "photo not found")
    if "favorited" in body:
        if bool(body["favorited"]):
            favorite_photo(user.id, uid)
        else:
            unfavorite_photo(user.id, uid)
    if "archived" in body:
        set_archived(uid, bool(body["archived"]))
    if "hidden" in body:
        set_hidden(uid, bool(body["hidden"]))
        # Hidden members are included (at the end of their group); invalidate
        # so the Duplicates tab reflects the new order right away.
        _invalidate_photo_dups_cache()
    row = get_photo(uid)
    return _single_user_photo(user.id, row)


@app.put("/api/photos/{uid}/tags")
def api_set_tags(uid: str, body: dict = Body(...),
                  user: CurrentUser = Depends(require_role("write"))):
    """Replace the freeform tag set for a photo. ``tags`` is a list of strings."""
    if get_photo(uid) is None:
        raise HTTPException(404, "photo not found")
    tags = body.get("tags") or []
    if not isinstance(tags, list):
        raise HTTPException(400, "tags must be a list of strings")
    clean = set_tags(uid, tags)
    return {"uid": uid, "tags": clean}


@app.get("/api/photos/{uid}/tags")
def api_get_tags(uid: str):
    if get_photo(uid) is None:
        raise HTTPException(404, "photo not found")
    return {"uid": uid, "tags": get_tags(uid)}


@app.get("/api/photos/anchors")
def api_photo_anchors():
    """Year-month anchors for the date rail. Cached for `_ANCHORS_CACHE_TTL`."""
    global _anchors_cache
    now = time.time()
    if _anchors_cache is not None and now - _anchors_cache[0] < _ANCHORS_CACHE_TTL:
        return _anchors_cache[1]

    import datetime as _dt

    anchors = []
    for r in photo_anchors():
        ym = r["ym"]
        try:
            label = _dt.datetime.strptime(ym, "%Y-%m").strftime("%b %Y")
        except Exception:
            label = ym
        anchors.append({"ym": ym, "label": label, "first_ts": r["first_ts"]})
    payload = {"anchors": anchors}
    _anchors_cache = (now, payload)
    return payload


@app.get("/api/albums")
def api_albums():
    albums = []
    for r in all_albums():
        albums.append(
            {
                "uid": r["uid"],
                "name": r["name"] or r["uid"],
                "photo_count": r["photo_count"] or 0,
                "start_ts": r["start_ts"],
                "end_ts": r["end_ts"],
                "cover_url": _sign_if_needed(
                    f"/api/photos/{r['cover_uid']}/thumb" if r["cover_uid"] else None
                ),
            }
        )
    return {"albums": albums}


@app.get("/api/albums/{album_uid}/photos")
def api_album_photos(album_uid: str, limit: int = 200, offset: int = 0,
                      user: CurrentUser = Depends(require_user)):
    rows = album_photos(album_uid, limit=limit, offset=offset)
    return {"photos": _user_photos(user.id, rows)}


@app.get("/api/places")
def api_places(limit: int = 500):
    rows = place_stats(limit=limit)
    places = []
    for r in rows:
        city = r["place"].split(",")[0].strip()
        places.append({"place": r["place"], "city": city, "count": r["photo_count"]})
    return {"places": places}


@app.get("/api/map")
def api_map(limit: int = 1000):
    rows = map_markers(limit=limit)
    markers = []
    for r in rows:
        city = r["place"].split(",")[0].strip()
        markers.append(
            {
                "place": r["place"],
                "city": city,
                "count": r["photo_count"],
                "lat": r["lat"],
                "lng": r["lng"],
                "thumb_url": _sign_if_needed(
                    f"/api/photos/{r['cover_uid']}/thumb" if r["cover_uid"] else None
                ),
            }
        )
    return {"markers": markers}


@app.get("/api/photos/{uid}")
def api_photo(uid: str, user: CurrentUser = Depends(require_user)):
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    return _single_user_photo(user.id, row)


@app.get("/api/photos/{uid}/meta")
def api_photo_meta(uid: str, user: CurrentUser = Depends(require_user)):
    """Full metadata for the photo detail view.

    Merges the local index row (GPS, place, faces, people) with the live
    node metadata Proton exposes for the photo (size, creation/modification
    times, photo tags, live-photo relations) fetched on demand from the
    bridge. Album names resolve from the local albums table (synced every
    10 minutes) so opening a detail panel never triggers a full bridge
    album enumeration.
    """
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    meta = _single_user_photo(user.id, row)

    # Faces + people in this photo (local index).
    faces = faces_for_photo(uid)
    people = {}
    for f in faces:
        pid = f["person_id"]
        if pid is None:
            continue
        people.setdefault(pid, f["person_name"])
    meta["face_count"] = len(faces)
    meta["people"] = [{"person_id": k, "name": v} for k, v in people.items()]

    # Live metadata from Proton (on demand; tolerate bridge failures).
    try:
        nodes = get_bridge().nodes([uid])
        if nodes:
            n = nodes[0]
            keys = (
                "size", "creationTime", "modificationTime", "mainPhotoNodeUid",
                "relatedPhotoNodeUids", "mediaType",
            )
            for k in keys:
                if n.get(k) is not None:
                    meta[k] = n[k]
            # Proton's read-only photo tags come back under `proton_tags` so
            # they don't collide with the user's local tags column.
            pt = n.get("tags")
            if pt is not None:
                meta["proton_tags"] = pt
    except Exception as exc:
        log.warning("bridge node metadata failed for %s: %s", uid, exc)

    # Album names resolve from the local albums table (synced every 10
    # minutes) — never a full bridge enumeration on photo open.
    albums_raw = meta.get("albums")
    if isinstance(albums_raw, str):
        try:
            album_uids = json.loads(albums_raw)
        except Exception:
            album_uids = []
    else:
        album_uids = albums_raw or []
    name_by_uid = album_names(album_uids)
    meta["albums_detail"] = [
        {"uid": u, "name": name_by_uid.get(u, u)} for u in album_uids
    ]

    return meta


@app.get("/api/photos/{uid}/thumb")
def api_thumb(uid: str, request: Request,
               _: object = Depends(signed_or_token)):
    if not is_valid_uid(uid):
        log.warning("discarding media request: invalid uid %r (%s)", uid, uid_invalid_reason(uid))
        raise HTTPException(404, "no thumbnail")
    p = settings.thumb_dir / f"{uid}.webp"
    if not p.exists():
        raise HTTPException(404, "thumbnail file missing")
    return FileResponse(p, media_type="image/webp", headers=_IMMUTABLE_HEADERS)


def _sniff_image_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        if data[8:12] in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
            return "image/heic"
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return None


@app.get("/api/photos/{uid}/full")
async def api_full(uid: str, request: Request,
                    _: object = Depends(signed_or_token)):
    """Stream the full-resolution photo from Proton (on demand, read-only).

    Uses an async semaphore for admission control (no threadpool threads
    consumed) and ``asyncio.wait_for`` for the hard timeout so a stuck
    Proton downloader endpoint can't hold the request open indefinitely.
    If we timeout, the bridge response is closed from the async side and
    we return a fast 504 to the browser.
    """
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    range_header = request.headers.get("range")

    # Acquire the semaphore before starting the timer so admission-queue
    # wait doesn't count toward the 30s header budget.
    async with _full_semaphore:
        try:
            resp = await asyncio.wait_for(
                get_bridge().full_photo_async(
                    uid, range_header=range_header,
                    timeout_ms=int(_FULL_TIMEOUT_SEC * 1000),
                ),
                timeout=_FULL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            log.warning(
                "full photo timed out after %.1fs for %s; returning 504",
                _FULL_TIMEOUT_SEC, uid,
            )
            _record_full_res_failure()
            raise HTTPException(504, "full photo fetch timed out — try again later")
        except BridgeTransientError as exc:
            log.warning(
                "full photo bridge transient for %s: %s", uid, exc,
            )
            _record_full_res_failure()
            detail = exc.args[0] if exc.args else "bridge transient error"
            raise HTTPException(
                status_code=exc.status_code,
                detail=detail,
                headers={"Retry-After": str(int(exc.retry_after_sec or 1))},
            )
        except Exception as exc:
            log.warning("full photo fetch failed for %s: %s", uid, exc)
            _record_full_res_failure()
            raise HTTPException(502, "bridge fetch failed")

    if resp.status_code not in (200, 206):
        log.warning("full photo bridge error for %s: status %s", uid, resp.status_code)
        _record_full_res_failure()
        await resp.aclose()
        raise HTTPException(resp.status_code, "bridge error")

    content_type = resp.headers.get("content-type", "application/octet-stream")
    headers = {"Cache-Control": "no-store"}
    for h in ("content-length", "accept-ranges", "content-range"):
        v = resp.headers.get(h)
        if v:
            headers[h] = v

    # Sniff the first chunk for content-type if the bridge returned
    # application/octet-stream.
    first_chunk = b""
    if content_type == "application/octet-stream":
        log.warning("full photo %s returned octet-stream; sniffing magic bytes", uid)
        try:
            first_chunk = await resp.aread(1 << 16)
        except Exception:
            pass
        sniffed = _sniff_image_type(first_chunk)
        headers["Content-Type"] = sniffed or "image/jpeg"

    async def gen():
        try:
            if first_chunk:
                yield first_chunk
            async for chunk in resp.aiter_bytes(1 << 16):
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        headers=headers,
    )


# --- people ----------------------------------------------------------------

def _people_cache_get_locked(q: str | None, now: float) -> list | None:
    """Fresh cached list for `q`, else None. Promotes `q` to most-recently-used
    on a hit. Caller must hold `_people_cache_lock`."""
    entry = _people_cache.get(q)
    if entry is None:
        return None
    ts, full = entry
    if now - ts >= _PEOPLE_CACHE_TTL:
        return None
    _people_cache.move_to_end(q)
    return full


def _people_cache_put_locked(q: str | None, now: float, full: list) -> None:
    """Store `full` under `q`, refreshing its timestamp and evicting the
    least-recently-used entry once the cache exceeds `_PEOPLE_CACHE_MAX`.
    Caller must hold `_people_cache_lock`."""
    _people_cache[q] = (now, full)
    _people_cache.move_to_end(q)
    while len(_people_cache) > _PEOPLE_CACHE_MAX:
        _people_cache.popitem(last=False)


def _people_all_cached() -> list:
    """Full (unfiltered) people list as serialized dicts, cached briefly.

    Avoids re-running the expensive people query on every call — shares the
    same LRU slot (q=None) as `/api/people` so the suggest endpoint and the
    people grid don't each pay for it independently.
    """
    global _people_cache
    now = time.time()
    entry = _people_cache.get(None)
    if entry is not None and now - entry[0] < _PEOPLE_CACHE_TTL:
        return entry[1]
    with _people_cache_lock:
        full = _people_cache_get_locked(None, now)
        if full is not None:
            return full
        rows = all_people()
        full = [
            {
                "id": r["id"],
                "name": r["name"],
                "cover_uid": r["cover_uid"],
                "cover_face_id": r["cover_face_id"],
                "face_count": r["face_count"],
                "photo_count": r["photo_count"],
                "cover_url": _sign_if_needed(
                    f"/api/people/{r['id']}/cover" if r["cover_face_id"] else None
                ),
            }
            for r in rows
        ]
        _people_cache_put_locked(None, now, full)
        return full


@app.get("/api/people")
def api_people(limit: int = 200, offset: int = 0, q: str | None = None):
    """People ordered by photo_count DESC. Paginated, optionally filtered by a
    case-insensitive name prefix.

    Each `q` prefix is materialized once and cached in a small LRU, then each
    page just slices the list — so infinite scroll (new offset per page) and
    typeahead keystrokes no longer re-run the aggregation for every request.
    """
    q = (q or "").strip() or None
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    global _people_cache
    now = time.time()
    if q is None:
        full = _people_all_cached()
        page = full[offset : offset + limit]
        return {"people": page, "total": len(full), "limit": limit, "offset": offset}
    entry = _people_cache.get(q)
    if entry is not None and now - entry[0] < _PEOPLE_CACHE_TTL:
        page = entry[1][offset : offset + limit]
        return {"people": page, "total": len(entry[1]), "limit": limit, "offset": offset}

    with _people_cache_lock:
        now = time.time()
        full = _people_cache_get_locked(q, now)
        if full is not None:
            page = full[offset : offset + limit]
            return {"people": page, "total": len(full), "limit": limit, "offset": offset}
        rows = all_people(q=q)
        total = len(rows)
        full = [
            {
                "id": r["id"],
                "name": r["name"],
                "cover_uid": r["cover_uid"],
                "cover_face_id": r["cover_face_id"],
                "face_count": r["face_count"],
                "photo_count": r["photo_count"],
                "cover_url": _sign_if_needed(
                    f"/api/people/{r['id']}/cover" if r["cover_face_id"] else None
                ),
            }
            for r in rows
        ]
        _people_cache_put_locked(q, now, full)
        page = full[offset : offset + limit]
        return {"people": page, "total": total, "limit": limit, "offset": offset}


@app.get("/api/people/{person_id}/cover")
def api_person_cover(person_id: int,
                      _: object = Depends(signed_or_token)):
    person = get_person(person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    face_id = person["cover_face_id"]
    if face_id is None:
        if not person["cover_uid"] or not is_valid_uid(person["cover_uid"]):
            log.warning("discarding cover request: invalid uid %r (%s)",
                        person["cover_uid"], uid_invalid_reason(person["cover_uid"]))
            raise HTTPException(404, "no cover available")
        p = settings.thumb_dir / f"{person['cover_uid']}.webp"
        if not p.exists():
            raise HTTPException(404, "thumbnail file missing")
        return FileResponse(p, media_type="image/webp", headers=_IMMUTABLE_HEADERS)
    cache_path = _crop_cache_path(face_id)
    if not cache_path.exists():
        crop = _face_crop_bytes(face_id)
        if crop is None:
            raise HTTPException(404, "cover face crop unavailable")
    return FileResponse(cache_path, media_type="image/jpeg", headers=_IMMUTABLE_HEADERS)


@app.get("/api/people/{person_id}/faces")
def api_person_faces(person_id: int, limit: int = 500,
                      user: CurrentUser = Depends(require_user)):
    """Every face of a person, for the cover picker.

    Returns face ids plus a signed crop URL for each, so the front-end can
    render a grid of candidate face-crops without knowing bbox math.
    """
    person = get_person(person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    rows = faces_for_person(person_id, limit=max(1, min(limit, 1000)))
    faces = [
        {
            "id": r["id"],
            "photo_uid": r["photo_uid"],
            "confidence": r["confidence"],
            "crop_url": _sign_if_needed(f"/api/faces/{r['id']}/crop"),
            "is_cover": r["id"] == person["cover_face_id"],
        }
        for r in rows
    ]
    return {"faces": faces, "cover_face_id": person["cover_face_id"], "count": len(faces)}


@app.post("/api/people/{person_id}/cover")
def api_people_set_cover(person_id: int, body: dict,
                         user: CurrentUser = Depends(require_role("write"))):
    """Set a person's cover photo from one of their own faces."""
    face_id = body.get("face_id")
    if not isinstance(face_id, int):
        raise HTTPException(400, "face_id required")
    person = get_person(person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    row = _face_row(face_id)
    if row is None or row["person_id"] != person_id:
        raise HTTPException(400, "face does not belong to this person")
    set_person_cover_face(person_id, face_id)
    _invalidate_people_cache()
    return {"ok": True, "cover_face_id": face_id,
            "cover_url": _sign_if_needed(f"/api/people/{person_id}/cover")}


def _face_crop_bytes(face_id: int) -> bytes | None:
    """Crop a face from its photo's cached thumbnail using the normalized bbox.

    Caches the result on disk under `crops/{face_id}.jpg` so subsequent
    requests serve a plain file (and `api_person_cover` / `api_face_crop`
    can use `FileResponse` with immutable cache headers).
    """
    import json

    from PIL import Image

    cache_path = _crop_cache_path(face_id)
    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except OSError:
            pass

    row = _face_row(face_id)
    if row is None:
        return None
    if not is_valid_uid(row["photo_uid"]):
        log.warning("discarding face crop: invalid photo uid %r (%s)",
                    row["photo_uid"], uid_invalid_reason(row["photo_uid"]))
        return None
    thumb = settings.thumb_dir / f"{row['photo_uid']}.webp"
    if not thumb.exists():
        return None
    bbox = json.loads(row["bbox"])
    x, y, w, h = bbox
    try:
        img = Image.open(thumb).convert("RGB")
        iw, ih = img.size
        # bbox is normalized to the thumbnail dimensions
        left = int(x * iw)
        top = int(y * ih)
        right = int((x + w) * iw)
        bottom = int((y + h) * ih)
        # pad slightly for context
        pad = 0.25
        pw = int((right - left) * pad)
        ph = int((bottom - top) * pad)
        left = max(0, left - pw)
        top = max(0, top - ph)
        right = min(iw, right + pw)
        bottom = min(ih, bottom + ph)
        crop = img.crop((left, top, right, bottom))
        out = io.BytesIO()
        crop.save(out, format="JPEG", quality=90)
        data = out.getvalue()
    except Exception as exc:
        log.warning("face crop failed for face %s: %s", face_id, exc)
        return None

    with _crop_lock:
        tmp = cache_path.with_suffix(".tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, cache_path)
        except OSError as exc:
            log.warning("face crop cache write failed for %s: %s", face_id, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return data


def start_crop_prewarm_worker() -> None:
    """Background thread that pre-generates every people cover crop once, so
    the People page serves plain files instead of ~30 ms PIL encodes on a
    ~97% cache-miss grid. Runs in the parent process only (with uvicorn
    workers>1 the parent is the only place these daemon threads live).
    Resumable: skips crops that already exist, and rescans periodically so
    newly created people eventually get covered too."""
    import concurrent.futures

    def _loop() -> None:
        import time as _time

        from store import get_conn

        log = logging.getLogger("crop-prewarm")
        while True:
            try:
                with get_conn() as conn:
                    rows = conn.execute(
                        "SELECT cover_face_id FROM people "
                        "WHERE cover_face_id IS NOT NULL ORDER BY id"
                    ).fetchall()
                missing = [
                    r[0] for r in rows if not _crop_cache_path(r[0]).exists()
                ]
                if not missing:
                    _time.sleep(300.0)
                    continue
                log.info("crop prewarm: %d cover crops missing, generating", len(missing))
                done = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futs = [pool.submit(_face_crop_bytes, fid) for fid in missing]
                    for _ in concurrent.futures.as_completed(futs):
                        done += 1
                        if done % 500 == 0:
                            log.info(
                                "crop prewarm: %d/%d done", done, len(missing)
                            )
                log.info("crop prewarm: finished %d crops", done)
                _time.sleep(300.0)
            except Exception:
                log.exception("crop prewarm iteration failed")
                _time.sleep(300.0)

    t = threading.Thread(target=_loop, name="crop-prewarm", daemon=True)
    t.start()
    log.info("crop prewarm worker started")


def _face_row(face_id: int):
    from store import get_conn

    with get_conn() as conn:
        row = conn.execute(
            """SELECT f.id, f.photo_uid, f.person_id, f.bbox
               FROM faces f JOIN photos ph ON ph.uid = f.photo_uid
               WHERE f.id=?""",
            (face_id,),
        ).fetchone()
        return row


@app.get("/api/faces/unassigned")
def api_unassigned_faces(limit: int = 500):
    rows = unassigned_faces(limit=limit)
    faces = [
        {
            "id": r["id"],
            "photo_uid": r["photo_uid"],
            "confidence": r["confidence"],
            "thumb_url": _sign_if_needed(f"/api/photos/{r['photo_uid']}/thumb"),
            "crop_url": _sign_if_needed(f"/api/faces/{r['id']}/crop"),
        }
        for r in rows
    ]
    return {"faces": faces}


@app.get("/api/faces/{face_id}/crop")
def api_face_crop(face_id: int,
                   _: object = Depends(signed_or_token)):
    cache_path = _crop_cache_path(face_id)
    if not cache_path.exists():
        crop = _face_crop_bytes(face_id)
        if crop is None:
            raise HTTPException(404, "face crop unavailable")
    return FileResponse(cache_path, media_type="image/jpeg", headers=_IMMUTABLE_HEADERS)


@app.get("/api/faces/{face_id}/suggest")
def api_face_suggest(face_id: int, limit: int = 5):
    """Rank existing people by how likely they are to be this face.

    Compares the face's own embedding against every person's mean embedding
    (cosine similarity). Drives the "top similar names" quick-pick in the
    unassigned-face popover. Read-only.
    """
    emb = face_embedding(face_id)
    if emb is None:
        return {"suggestions": []}
    fe = np.frombuffer(emb, dtype=np.float32)
    pids, M = person_mean_matrix_from_cache()
    if M.shape[0] == 0:
        return {"suggestions": []}
    sims = M @ fe
    order = np.argsort(-sims)
    top_n = max(1, min(limit, 50))
    # Batch fetch ONLY the top-N people by PK (plus face/photo counts). Never
    # the 37 s all-people aggregation: typeahead and grid already warm this via
    # `_people_cache`, but a cold cache must not stall the popover.

    by_id = {r["id"]: r for r in people_by_ids([int(pids[i]) for i in order[:top_n]])}
    scored = []
    for i in order[:top_n]:
        pid = int(pids[i])
        p = by_id.get(pid)
        scored.append(
            {
                "person_id": pid,
                "name": (p["name"] if p else None) or f"person {pid}",
                "similarity": float(sims[i]),
                "photo_count": p["photo_count"] if p else 0,
                "face_count": p["face_count"] if p else 0,
                "cover_url": _sign_if_needed(
                    f"/api/people/{pid}/cover" if p and p["cover_face_id"] else None
                ),
            }
        )
    return {"suggestions": scored}


@app.get("/api/photos/{uid}/faces")
def api_photo_faces(uid: str):
    rows = faces_for_photo(uid)
    faces = []
    for r in rows:
        import json

        faces.append(
            {
                "id": r["id"],
                "person_id": r["person_id"],
                "person_name": r["person_name"],
                "bbox": json.loads(r["bbox"]),
                "confidence": r["confidence"],
            }
        )
    return {"faces": faces}


def _merge_propagate(person_id: int, threshold: float | None = None) -> int:
    """Auto-tag unassigned faces similar to a person's mean embedding.

    Returns how many faces were assigned. Used after merges / renames so a
    merged person also pulls in unassigned look-alikes.
    """
    emb = person_mean_embedding(person_id)
    if emb is None:
        return 0
    thr = threshold if threshold is not None else settings.face_sim_threshold
    assigned = 0
    for sim_row in similar_faces(emb.tobytes(), thr, limit=500):
        if sim_row[2] is None:  # person_id
            assign_face_person(sim_row[0], person_id)
            assigned += 1
    return assigned


@app.post("/api/faces/{face_id}/person")
def api_face_assign(face_id: int, body: dict,
                     user: CurrentUser = Depends(require_role("write"))):
    """Assign a face to an existing person (person_id) or create a new named person (name).
    When creating by name, merge into an existing person with the same name.
    Propagates the assignment to similar unassigned faces."""
    person_id = body.get("person_id")
    name = (body.get("name") or "").strip()
    if person_id is None and not name:
        raise HTTPException(400, "provide person_id or name")
    merged = False
    if person_id is not None:
        person = get_person(person_id)
        if person is None:
            raise HTTPException(404, "person not found")
    else:
        row = _face_row(face_id)
        cover_uid = row["photo_uid"] if row else None
        existing = find_person_by_name(name, exclude_id=None)
        if existing is not None:
            person_id = existing["id"]
            merged = True
            set_person_cover_face(person_id, face_id)
        else:
            person_id = create_person(name=name, cover_uid=cover_uid, cover_face_id=face_id)

    assign_face_person(face_id, person_id)
    set_person_cover_face(person_id, face_id)
    _drop_crop_cache(face_id)

    # similarity propagation: tag unassigned look-alikes
    emb = face_embedding(face_id)
    assigned = 0
    if emb is not None:
        for sim_row in similar_faces(emb, settings.face_sim_threshold, limit=500):
            if sim_row[2] is None:  # person_id
                assign_face_person(sim_row[0], person_id)
                assigned += 1
    _invalidate_dups_cache()
    _invalidate_people_cache()
    return {
        "ok": True,
        "person_id": person_id,
        "merged": merged,
        "assigned_similar": assigned,
    }


@app.post("/api/faces/{face_id}/unassign")
def api_face_unassign(face_id: int,
                       user: CurrentUser = Depends(require_role("write"))):
    unassign_face(face_id)
    _drop_crop_cache(face_id)
    _invalidate_people_cache()
    _invalidate_dups_cache()
    return {"ok": True}


@app.post("/api/people/{person_id}/name")
def api_people_rename(person_id: int, body: dict,
                       user: CurrentUser = Depends(require_role("write"))):
    """Rename a person. If another person already has that name, merge instead."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    existing = find_person_by_name(name, exclude_id=person_id)
    if existing is not None:
        _drop_person_crops(person_id)
        merge_person(person_id, existing["id"])
        _merge_propagate(existing["id"])
        _invalidate_dups_cache()
        _invalidate_people_cache()
        tgt = get_person(existing["id"])
        return {
            "ok": True,
            "merged": True,
            "target_id": existing["id"],
            "photo_count": tgt["photo_count"] if tgt else None,
            "face_count": tgt["face_count"] if tgt else None,
        }
    rename_person(person_id, name)
    _invalidate_dups_cache()
    _invalidate_people_cache()
    return {"ok": True, "merged": False}


@app.post("/api/people/{source_id}/merge")
def api_people_merge(source_id: int, body: dict,
                      user: CurrentUser = Depends(require_role("write"))):
    """Explicitly merge source person into target (by id)."""
    target_id = body.get("target_id")
    if not isinstance(target_id, int):
        raise HTTPException(400, "target_id required")
    if source_id == target_id:
        raise HTTPException(400, "cannot merge a person into itself")
    target = get_person(target_id)
    if target is None:
        raise HTTPException(404, "target person not found")
    _drop_person_crops(source_id)
    merge_person(source_id, target_id)
    assigned = _merge_propagate(target_id)
    _invalidate_dups_cache()
    _invalidate_people_cache()
    tgt = get_person(target_id)
    return {
        "ok": True,
        "target_id": target_id,
        "assigned_similar": assigned,
        "photo_count": tgt["photo_count"] if tgt else None,
        "face_count": tgt["face_count"] if tgt else None,
    }


@app.get("/api/people/{person_id}/similar")
def api_people_similar(person_id: int, threshold: float = 0.40, limit: int = 50, offset: int = 0):
    """People whose mean face embedding is similar to `person_id`'s (cosine).

    Vectorized: one (P,512) @ (512,) matmul over every other person's mean,
    reusing the shared cached face matrix. Returns only people at/above the
    given similarity `threshold`, sorted by score desc. Drives the per-person
    "similar people" merge assistant. `offset` pages past `limit` — the
    matmul is O(P) and cheap, so showing all candidates is just slicing.
    """
    if limit < 1:
        limit = 50
    offset = max(0, offset)
    pids, M = person_mean_matrix_from_cache()
    tgt = np.flatnonzero(pids == person_id)
    if tgt.size == 0 or pids.size < 2:
        return {"similar": [], "total": 0}
    fe = M[tgt[0]]
    sims = M @ fe
    hits = np.flatnonzero((sims >= threshold) & (pids != person_id))
    total = int(hits.size)
    if total == 0:
        return {"similar": [], "total": 0}
    order = hits[np.argsort(-sims[hits])][offset : offset + limit]
    top_pids = [int(pids[i]) for i in order]
    top_sims = [float(sims[i]) for i in order]
    by_id = {r["id"]: r for r in people_by_ids(top_pids)}
    similar = []
    for pid, sim in zip(top_pids, top_sims):
        p = by_id.get(pid)
        similar.append(
            {
                "person_id": pid,
                "name": (p["name"] if p else None) or f"person {pid}",
                "similarity": round(sim, 4),
                "photo_count": p["photo_count"] if p else 0,
                "face_count": p["face_count"] if p else 0,
                "cover_url": _sign_if_needed(
                    f"/api/people/{pid}/cover" if p and p["cover_face_id"] else None
                ),
            }
        )
    return {"similar": similar, "total": total}


@app.post("/api/people/{target_id}/merge_all")
def api_people_merge_all(target_id: int, body: dict,
                         user: CurrentUser = Depends(require_role("write"))):
    """Merge many people into `target_id` in one call.

    Body: {"source_ids": [..]}. Each source is moved into the target (its
    faces re-parented, its row deleted), then look-alike propagation runs once
    against the merged target so newly absorbed faces also pull in unassigned
    matches. The target itself must exist and is never merged into itself.
    """
    source_ids = body.get("source_ids")
    if not isinstance(source_ids, list) or not source_ids:
        raise HTTPException(400, "source_ids list required")
    target = get_person(target_id)
    if target is None:
        raise HTTPException(404, "target person not found")
    merged_ids: list[int] = []
    seen: set[int] = set()
    for sid in source_ids:
        if not isinstance(sid, int) or sid == target_id or sid in seen:
            continue
        seen.add(sid)
        if get_person(sid) is None:
            continue
        _drop_person_crops(sid)
        merge_person(sid, target_id)
        merged_ids.append(sid)
    assigned = _merge_propagate(target_id) if merged_ids else 0
    _invalidate_dups_cache()
    _invalidate_people_cache()
    tgt = get_person(target_id)
    return {
        "ok": True,
        "target_id": target_id,
        "merged_count": len(merged_ids),
        "assigned_similar": assigned,
        "photo_count": tgt["photo_count"] if tgt else None,
        "face_count": tgt["face_count"] if tgt else None,
    }


@app.post("/api/people/{target_id}/merge_all_similar")
def api_people_merge_all_similar(target_id: int, body: dict,
                                 user: CurrentUser = Depends(require_role("write"))):
    """Merge every person whose mean embedding is similar to the target's.

    Body: {"threshold": 0.40, "max_sources": 5000}. Unlike the per-person
    similar list (capped at 50 for the UI modal), this computes the FULL set
    of look-alike people with one vectorized matmul and merges them all in a
    single transaction + one propagation pass, so a 49k-people dedupe costs
    one request instead of thousands of 50-batches. `max_sources` bounds a
    single call for very large campaigns (the client can loop).
    """
    threshold = float(body.get("threshold", 0.40))
    max_sources = int(body.get("max_sources", 5000))
    if max_sources < 1:
        max_sources = 5000
    target = get_person(target_id)
    if target is None:
        raise HTTPException(404, "target person not found")
    pids, M = person_mean_matrix_from_cache()
    tgt_idx = np.flatnonzero(pids == target_id)
    if tgt_idx.size == 0 or pids.size < 2:
        return {"ok": True, "target_id": target_id, "merged_count": 0, "assigned_similar": 0,
                "photo_count": target["photo_count"], "face_count": target["face_count"]}
    fe = M[tgt_idx[0]]
    sims = M @ fe
    hits = np.flatnonzero((sims >= threshold) & (pids != target_id))
    order = hits[np.argsort(-sims[hits])]
    source_ids = [int(pids[i]) for i in order[:max_sources]]
    if not source_ids:
        return {"ok": True, "target_id": target_id, "merged_count": 0, "assigned_similar": 0,
                "photo_count": target["photo_count"], "face_count": target["face_count"]}
    _drop_people_crops(source_ids)
    merged_count = merge_people_bulk(source_ids, target_id)
    assigned = _merge_propagate(target_id) if merged_count else 0
    _invalidate_dups_cache()
    _invalidate_people_cache()
    tgt = get_person(target_id)
    return {
        "ok": True,
        "target_id": target_id,
        "merged_count": merged_count,
        "assigned_similar": assigned,
        "photo_count": tgt["photo_count"] if tgt else None,
        "face_count": tgt["face_count"] if tgt else None,
    }


@app.get("/api/people/duplicates")
def api_people_duplicates(threshold: float = 0.40, limit: int = 50):
    """Find people whose mean face embeddings are highly similar (likely dupes).

    Reuses the shared cached people list + person-mean embeddings, and walks
    the similarity matrix block-by-block with a bounded top-K heap so the
    (M x M) pair matrix is never materialized in full — at 49k people that
    would be ~9.6 GB and OOM. Results (and response shape) are identical to
    the naive single matmul. Cached for a few seconds so reloads are cheap.
    """
    global _dups_cache
    if limit < 1:
        limit = 50
    now = time.time()
    if _dups_cache is not None and now - _dups_cache[0] < _DUP_CACHE_TTL:
        return _dups_cache[1]
    with _dups_cache_lock:
        now = time.time()
        if _dups_cache is not None and now - _dups_cache[0] < _DUP_CACHE_TTL:
            return _dups_cache[1]
        resp = _dups_payload(threshold, limit)
        _dups_cache = (now, resp)
        return resp


def _dups_payload(threshold: float, limit: int) -> dict:
    """Walk the blockwise similarity matrix and build the duplicates payload."""
    people = _people_all_cached()  # avoids re-running the expensive GROUP-BY query
    if len(people) < 2:
        return {"duplicates": []}
    means = person_mean_embeddings_from_cache()  # reuses the shared face-matrix cache
    mats = []
    ids = []
    by_id = {}
    for p in people:
        emb = means.get(p["id"])
        if emb is not None:
            mats.append(emb)
            ids.append(p["id"])
            by_id[p["id"]] = p
    if len(mats) < 2:
        return {"duplicates": []}
    X = np.stack(mats).astype(np.float32)  # (M, 512)
    M = X.shape[0]
    import heapq

    # Blockwise similarity walk: each block is (B x M), so peak memory is
    # O(B*M) instead of O(M*M). We keep only the top-`limit` pairs in a
    # bounded min-heap, only counting each unordered pair once (global i < j).
    block = 1024
    heap: list[tuple[float, int, int]] = []
    for s in range(0, M, block):
        e = min(s + block, M)
        Sb = X[s:e] @ X.T
        rows, cols = np.nonzero(Sb >= threshold)
        keep = (s + rows) < cols  # global i < j, no np.triu copy
        rows, cols = rows[keep], cols[keep]
        g_rows = s + rows
        vals = Sb[rows, cols].tolist()
        for v, gi, gj in zip(vals, g_rows.tolist(), cols.tolist()):
            if len(heap) < limit:
                heapq.heappush(heap, (v, gi, gj))
            elif v > heap[0][0]:
                heapq.heapreplace(heap, (v, gi, gj))
    hits = sorted(heap, reverse=True)

    dups = []
    for sim, i, j in hits:
        a, b = by_id[ids[i]], by_id[ids[j]]
        dups.append(
            {
                "similarity": round(float(sim), 4),
                "a": {
                    "id": a["id"],
                    "name": a["name"],
                    "photo_count": a["photo_count"],
                    "face_count": a["face_count"],
                    "cover_url": a["cover_url"],
                },
                "b": {
                    "id": b["id"],
                    "name": b["name"],
                    "photo_count": b["photo_count"],
                    "face_count": b["face_count"],
                    "cover_url": b["cover_url"],
                },
            }
        )
    return {"duplicates": dups}


def _suggested_rows(threshold: float) -> list[dict]:
    """Per-target ranking of which people have look-alikes at/above `threshold`.

    Same blockwise pairwise scan as ``_dups_payload``, but aggregated per
    person instead of into a top-K pair heap: each person collects a
    candidate count and their top-3 similarity scores. Rows are ranked with
    **named** people first (merging anonymous clusters into an already
    identified person is the high-value outcome), then by candidate count,
    then by best score. The scan itself is the expensive part (~40 s at 30 k
    people); callers cache the resulting list per threshold.
    """
    people = _people_all_cached()
    if len(people) < 2:
        return []
    means = person_mean_embeddings_from_cache()
    mats: list[np.ndarray] = []
    ids: list[int] = []
    by_id: dict[int, dict] = {}
    for p in people:
        emb = means.get(p["id"])
        if emb is not None:
            mats.append(emb)
            ids.append(p["id"])
            by_id[p["id"]] = p
    if len(mats) < 2:
        return []
    X = np.stack(mats).astype(np.float32)  # (M, 512)
    M = X.shape[0]
    counts = np.zeros(M, dtype=np.int32)
    tops: list[list[float]] = [[] for _ in range(M)]

    block = 1024
    for s in range(0, M, block):
        e = min(s + block, M)
        Sb = X[s:e] @ X.T
        rows, cols = np.nonzero(Sb >= threshold)
        keep = (s + rows) < cols  # global i < j, no np.triu copy
        rows, cols = rows[keep], cols[keep]
        g_rows = s + rows
        vals = Sb[rows, cols].tolist()
        for v, gi, gj in zip(vals, g_rows.tolist(), cols.tolist()):
            counts[gi] += 1
            counts[gj] += 1
            for idx in (gi, gj):
                t = tops[idx]
                if len(t) < 3:
                    t.append(v)
                    t.sort(reverse=True)
                elif v > t[-1]:
                    t[-1] = v
                    t.sort(reverse=True)

    rows = []
    for gi, c in enumerate(counts):
        if not c:
            continue
        p = by_id[ids[gi]]
        rows.append(
            {
                "person_id": ids[gi],
                "name": p["name"],
                "cover_url": p["cover_url"],
                "photo_count": p["photo_count"],
                "face_count": p["face_count"],
                "candidate_count": int(c),
                "top_scores": [round(float(x), 4) for x in tops[gi]],
            }
        )
    rows.sort(
        key=lambda r: (
            0 if r["name"] else 1,  # named targets first
            -r["candidate_count"],
            -(r["top_scores"][0] if r["top_scores"] else 0.0),
            r["person_id"],
        )
    )
    return rows


@app.get("/api/people/suggested-merges")
def api_people_suggested_merges(threshold: float = 0.40, limit: int = 50, offset: int = 0):
    """Person-centric suggested merges: who has look-alikes? (named first)

    Discovery counterpart to `/duplicates`: instead of 1:1 pairs, returns
    one row per person that has at least one look-alike at/above `threshold`,
    ranked with named people before anonymous clusters. Each row carries a
    `candidate_count` and `top_scores` so the UI can offer "review look-alikes"
    for one target at a time. The computation shares the expensive blockwise
    pairwise scan; the ranked row list is cached per threshold for
    `_SUGGESTED_CACHE_TTL` seconds and `offset`/`limit` slice it.
    """
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    with _suggested_cache_lock:
        now = time.time()
        entry = _suggested_cache.get(threshold)
        if entry is not None and now - entry[0] < _SUGGESTED_CACHE_TTL:
            full = entry[1]
        else:
            full = _suggested_rows(threshold)
            _suggested_cache[threshold] = (time.time(), full)
    page = full[offset : offset + limit]
    named = sum(1 for r in full if r["name"])
    return {
        "people": page,
        "total": len(full),
        "named": named,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/people/{person_id}/photos")
def api_person_photos(person_id: int, limit: int = 200, offset: int = 0,
                       user: CurrentUser = Depends(require_user)):
    rows = photos_for_person(person_id, limit=limit, offset=offset)
    return {"photos": _user_photos(user.id, rows), "count": count_faces_for_person(person_id)}


@app.get("/api/people/{person_id}/map")
def api_person_map(person_id: int, limit: int = 500):
    """Clustered map markers for one person: places they've been photographed in.

    Same shape as ``/api/map`` but filtered to photos carrying faces of this
    person. Drives the per-person "map of where I've seen them" view.
    """
    rows = person_map_markers(person_id, limit=limit)
    markers = []
    for r in rows:
        city = r["place"].split(",")[0].strip()
        markers.append({
            "place": r["place"],
            "city": city,
            "count": r["photo_count"],
            "lat": r["lat"],
            "lng": r["lng"],
            "thumb_url": _sign_if_needed(
                f"/api/photos/{r['cover_uid']}/thumb" if r["cover_uid"] else None
            ),
        })
    return {"markers": markers}


# --- search ----------------------------------------------------------------

# Upper bound for `limit` on the search endpoints. The UI never asks for more
# than 200 results, and the result-assembly loops do a `get_photo` SQLite call
# per hit — an unclamped limit lets one request amplify into a worker-wide
# CPU/memory/bandwidth DoS.
SEARCH_MAX_LIMIT = 200


def _clamp_search_limit(limit: int) -> int:
    return max(1, min(limit, SEARCH_MAX_LIMIT))


# Over-sample factor for the face-search top-k window. Face search dedupes
# per photo (best face per photo wins), so a photo that contributed many
# faces could otherwise exhaust the window; `_face_similarity` scans the tail
# exactly when the window cannot fill `limit` photos, keeping results
# identical to a full argsort.
_FACE_DEDUPE_SLACK = 4


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the largest ``k`` scores, in descending order.

    ``np.argpartition`` selects the top-k in O(N) instead of a full
    O(N log N) ``np.argsort``; only the k survivors get sorted. ``k`` is
    clamped to the number of scores.
    """
    n = scores.shape[0]
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.intp)
    k = min(k, n)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


@app.get("/api/search")
def api_search(q: str, limit: int = 100, user: CurrentUser = Depends(require_user)):
    """Free-text semantic search via CLIP (objects, scenes, etc.)."""
    q = q.strip()
    if not q:
        raise HTTPException(400, "q required")
    try:
        vec = embed_text(q)
    except Exception as exc:
        log.warning("clip text embed failed: %s", exc)
        raise HTTPException(503, "CLIP model unavailable")
    return _semantic_search(vec, _clamp_search_limit(limit), user.id)


# Max bytes for `/api/search/face` uploads (P-03 from the 2026-09-01 pen
# test). Without a cap, an attacker can stream a 50 MB blob to the API
# and force Python to allocate ~50 MB per request — 20 parallel requests
# spike to 1 GB resident. 8 MB is plenty for a face photo and well
# within the limits InsightFace accepts.
FACE_SEARCH_MAX_UPLOAD_BYTES = int(
    os.environ.get("FACE_SEARCH_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024))
)

# Max decoded pixel count for an image. Without this, a 1.6 MB JPEG can
# decompress to 300 MB of pixel data (decompression bomb) and OOM the
# process. 50 megapixels ≈ 7000×7000 px, enough for any sensible face
# photo and well below the (no)default limit.
FACE_SEARCH_MAX_IMAGE_PIXELS = int(
    os.environ.get("FACE_SEARCH_MAX_IMAGE_PIXELS", str(50_000_000))
)


@app.post("/api/search/face")
def api_face_search(file: UploadFile = File(...), limit: int = 50,
                     user: CurrentUser = Depends(require_user)):
    """Upload a face photo, find matching people/photos.

    Hardening (P-03):
    - `FACE_SEARCH_MAX_UPLOAD_BYTES` cap on the raw upload. 413 if
      exceeded — we never even allocate the buffer.
    - `FACE_SEARCH_MAX_IMAGE_PIXELS` cap on decoded pixels (PIL).
      Defense against decompression bombs.

    This is a sync ``def`` so FastAPI runs it in the threadpool — the
    PIL decode, InsightFace inference, and vector search are all CPU-bound
    and would stall the event loop if ``async def``.
    """
    # Read with a hard cap so a 50 MB blob doesn't get fully buffered.
    # In a sync route, UploadFile.file is a SpooledTemporaryFile.
    data = file.file.read(FACE_SEARCH_MAX_UPLOAD_BYTES + 1)
    if len(data) > FACE_SEARCH_MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"file too large (max {FACE_SEARCH_MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )
    try:
        from PIL import Image

        # Pre-check the declared image dimensions before we decode.
        # PIL reads the header cheaply and tells us the width/height;
        # rejecting here blocks decompression bombs without paying for
        # the (potentially huge) decode.
        Image.MAX_IMAGE_PIXELS = FACE_SEARCH_MAX_IMAGE_PIXELS
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
            if w * h > FACE_SEARCH_MAX_IMAGE_PIXELS:
                raise HTTPException(
                    413,
                    f"image too large ({w}×{h} = {w * h:,} pixels, "
                    f"max {FACE_SEARCH_MAX_IMAGE_PIXELS:,})",
                )
            img.load()  # force full decode so any further limits take effect
            arr = np.asarray(img.convert("RGB"))
            bgr = arr[:, :, ::-1].copy()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"could not read image: {exc}")
    emb = embed_query_face(bgr)
    if emb is None:
        raise HTTPException(404, "no face found in image")
    return _face_similarity(emb, _clamp_search_limit(limit), user.id)


def _get_clip_matrix() -> tuple[list[str], np.ndarray]:
    """Cached (uids, X) matrix of every CLIP embedding.

    Prefers the mmap sidecar written by the indexer; falls back to the
    DB-based cache when sidecar files are absent. Freshness follows the TTL
    on the hot path: the SQLite COUNT that detects new clips is only re-read
    once the TTL expires, so a debounced text search never opens an extra DB
    connection per keystroke. At expiry the matrix is reused (re-stamped)
    when the count is unchanged.
    """
    global _clip_cache
    now = time.time()
    if _clip_cache is not None:
        ts, _, uids, X = _clip_cache
        if (now - ts) < _CLIP_CACHE_TTL:
            return uids, X
    with _clip_cache_lock:
        now = time.time()
        if _clip_cache is not None:
            ts, _, uids, X = _clip_cache
            if (now - ts) < _CLIP_CACHE_TTL:
                return uids, X
        # TTL expired (or cold cache): one COUNT decides build vs reuse.
        n_now = clip_count()
        if _clip_cache is not None:
            ts, n_cached, uids, X = _clip_cache
            if n_cached == n_now:
                _clip_cache = (now, n_cached, uids, X)
                return uids, X
        # Try mmap sidecar first
        sidecar = read_clip_sidecar()
        if sidecar is not None:
            uids, X = sidecar
            _clip_cache = (now, len(uids), uids, X)
            return uids, X
        # Fallback: build from DB
        rows = all_clips()
        if not rows:
            return [], np.empty((0, 512), dtype=np.float32)
        uids = [r["photo_uid"] for r in rows]
        X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        _clip_cache = (now, n_now, uids, X)
        return uids, X


def _semantic_search(vec: np.ndarray, limit: int, user_id: int) -> dict:
    limit = _clamp_search_limit(limit)
    uids, X = _get_clip_matrix()
    if X.size == 0:
        return {"results": [], "total": 0}
    sims = X @ vec  # all embeddings are L2-normalized
    idx = _topk_indices(sims, limit)
    photo_uids = [uids[i] for i in idx]
    photos = get_photos_batch(photo_uids)
    fav_set = favorite_uids(user_id, photo_uids)
    results = []
    for i in idx:
        uid = uids[i]
        photo = photos.get(uid)
        if photo is None:
            continue
        d = _row_to_dict(photo)
        d["favorited_by_me"] = uid in fav_set
        d["score"] = float(sims[i])
        results.append(d)
    return {"results": results, "total": len(results)}


def _face_similarity(emb: np.ndarray, limit: int, user_id: int) -> dict:
    limit = _clamp_search_limit(limit)
    data = _embedding_cache_data()
    mat = data["mat"]
    if mat.shape[0] == 0:
        return {"results": [], "total": 0}
    uids = data["photo_uids"]
    scores = mat @ emb
    # Rank all faces by similarity, then dedupe per photo keeping the best
    # face score. Only the top (limit * slack) faces are selected via
    # argpartition (O(N) vs the full O(N log N) argsort); the slack covers
    # photos that contributed many faces. If the window still cannot fill
    # `limit` photos, the remaining lower-ranked faces are scanned exactly,
    # so results stay identical to a full argsort.
    n = scores.shape[0]
    order = _topk_indices(scores, min(limit * _FACE_DEDUPE_SLACK, n))
    seen: set = set()
    top_uids: list = []
    top_scores: list = []
    for i in order:
        uid = uids[i]
        if uid in seen:
            continue
        seen.add(uid)
        top_uids.append(uid)
        top_scores.append(float(scores[i]))
        if len(top_uids) >= limit:
            break
    if len(order) < n and len(top_uids) < limit:
        # The top-k window ran out of distinct photos before `limit` was
        # reached (one photo contributed too many faces). Keep walking the
        # remaining faces in score order so the result stays exact.
        rest = np.setdiff1d(np.arange(n), order)
        rest = rest[np.argsort(-scores[rest])]
        for i in rest:
            if len(top_uids) >= limit:
                break
            uid = uids[i]
            if uid in seen:
                continue
            seen.add(uid)
            top_uids.append(uid)
            top_scores.append(float(scores[i]))
    # Batch-fetch all photos in one query instead of N+1 get_photo calls.
    photos = get_photos_batch(top_uids)
    fav_set = favorite_uids(user_id, top_uids)
    results = []
    for uid, score in zip(top_uids, top_scores):
        photo = photos.get(uid)
        if photo is None:
            continue
        d = _row_to_dict(photo)
        d["favorited_by_me"] = uid in fav_set
        d["score"] = score
        results.append(d)
    return {"results": results, "total": len(results)}


# --- admin: user management + server ops -----------------------------------
import admin  # noqa: E402  (imported late to avoid a circular import)


def _user_row_public(row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
        "disabled": bool(row["disabled"]),
    }


@app.get("/api/admin/users")
def api_admin_list_users(_: CurrentUser = Depends(require_role("admin"))):
    if demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    return {"users": [_user_row_public(r) for r in list_users()]}


@app.post("/api/admin/users")
def api_admin_create_user(body: dict,
                           _: CurrentUser = Depends(require_role("admin"))):
    if demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    username = (body.get("username") or "").strip()
    display_name = (body.get("display_name") or username).strip() or username
    password = body.get("password") or ""
    role = (body.get("role") or "read").strip().lower()
    if len(username) < 2:
        raise HTTPException(400, "username must be at least 2 characters")
    if len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    if role not in ROLE_RANK:
        raise HTTPException(400, f"role must be one of {sorted(ROLE_RANK)}")
    if get_user_by_username(username) is not None:
        raise HTTPException(409, "username already exists")
    user_id = create_user(username=username, password_hash=hash_password(password),
                           role=role, display_name=display_name)
    row = get_user_by_id(user_id)
    return {"user": _user_row_public(row)}


@app.patch("/api/admin/users/{user_id}")
def api_admin_update_user(user_id: int, body: dict,
                           actor: CurrentUser = Depends(require_role("admin"))):
    if demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    if get_user_by_id(user_id) is None:
        raise HTTPException(404, "user not found")
    display_name = body.get("display_name")
    role = body.get("role")
    disabled = body.get("disabled")
    password = body.get("password")
    password_hash = hash_password(password) if password else None
    if role is not None and role not in ROLE_RANK:
        raise HTTPException(400, f"role must be one of {sorted(ROLE_RANK)}")
    if password is not None and len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    try:
        update_user(user_id, display_name=display_name, role=role,
                    disabled=disabled, password_hash=password_hash)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    # P-04: when the password changes, revoke all existing tokens for
    # this user so the new credentials take effect immediately. We
    # skip the revoke when the actor is editing themselves — otherwise
    # the admin gets logged out mid-edit and the response would be
    # useless (no way to use the new password without re-logging in).
    revoked = 0
    if password and not (user_id == actor.id):
        revoked = revoke_all_tokens(user_id)
    row = get_user_by_id(user_id)
    return {"user": _user_row_public(row), "tokens_revoked": revoked}


@app.delete("/api/admin/users/{user_id}")
def api_admin_delete_user(user_id: int,
                           actor: CurrentUser = Depends(require_role("admin"))):
    """Remove a user. The last remaining admin cannot delete themselves."""
    if demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    row = get_user_by_id(user_id)
    if row is None:
        raise HTTPException(404, "user not found")
    if row["id"] == actor.id:
        # Refuse if this is the last admin (would lock everyone out).
        admins = [u for u in list_users() if u["role"] == "admin" and not u["disabled"] and u["id"] != actor.id]
        if not admins:
            raise HTTPException(400, "cannot delete the last admin")
    delete_user(user_id)  # ON DELETE CASCADE drops their tokens + favorites
    return {"ok": True}


@app.post("/api/admin/users/{user_id}/logout")
def api_admin_revoke_user_tokens(user_id: int,
                                   _: CurrentUser = Depends(require_role("admin"))):
    """Sign a user out of every device."""
    if demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    n = revoke_all_tokens(user_id)
    return {"ok": True, "revoked": n}


# --- admin: server ops (backup / disk / checks / schedule) -----------------

@app.get("/api/admin/overview")
def api_admin_overview(_: CurrentUser = Depends(require_role("admin"))):
    return admin.overview()


@app.post("/api/admin/backup")
def api_admin_backup(_: CurrentUser = Depends(require_role("admin"))):
    """Trigger a manual snapshot now."""
    if demo_disable_backups():
        raise HTTPException(404, "not found")
    try:
        res = admin.snapshot_backup()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    return {"name": res["name"], "size": res["size"], "ts": res["ts"]}


@app.get("/api/admin/backups")
def api_admin_list_backups(_: CurrentUser = Depends(require_role("admin"))):
    if demo_disable_backups():
        raise HTTPException(404, "not found")
    return admin.list_backups()


@app.delete("/api/admin/backups/{name}")
def api_admin_delete_backup(name: str, _: CurrentUser = Depends(require_role("admin"))):
    if demo_disable_backups():
        raise HTTPException(404, "not found")
    try:
        return admin.delete_backup(name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/admin/backups/prune")
def api_admin_prune_backups(body: dict = Body(default={}),
                            _: CurrentUser = Depends(require_role("admin"))):
    if demo_disable_backups():
        raise HTTPException(404, "not found")
    try:
        res = admin.prune_backups(body.get("keep"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return {"ok": res.get("ok", True),
            "removed": res.get("removed", []),
            "removed_count": len(res.get("removed", [])),
            "kept": res.get("kept", 0)}


@app.get("/api/admin/schedule")
def api_admin_get_schedule(_: CurrentUser = Depends(require_role("admin"))):
    return admin.get_schedule()


@app.put("/api/admin/schedule")
def api_admin_set_schedule(body: dict = Body(...),
                           _: CurrentUser = Depends(require_role("admin"))):
    return admin.set_schedule(body)


@app.get("/api/admin/sync")
def api_admin_get_sync(_: CurrentUser = Depends(require_role("admin"))):
    """Admin view of the indexer's sync state + live sync configuration."""
    status = _indexer_proxy_json("GET", "/status")
    cfg = _indexer_proxy_json("GET", "/sync-config")
    return {
        "last_sync": status.get("last_sync"),
        "last_sync_error": status.get("last_sync_error"),
        "threads": status.get("threads", {}),
        "tip_interval": settings.sync_interval,
        "config": cfg,
    }


@app.post("/api/admin/sync/trigger")
def api_admin_trigger_sync(_: CurrentUser = Depends(require_role("admin"))):
    """Ask the indexer to run a full scan on its next sync-loop iteration."""
    return _indexer_proxy_json("POST", "/trigger-sync")


@app.put("/api/admin/sync")
def api_admin_set_sync(body: dict = Body(...),
                       _: CurrentUser = Depends(require_role("admin"))):
    """Persist the live sync configuration on the indexer."""
    return _indexer_proxy_json("PUT", "/sync-config", body)


@app.post("/api/admin/checks")
def api_admin_checks(_: CurrentUser = Depends(require_role("admin"))):
    return admin.run_checks(recent_full_res_failures=_recent_full_res_failures())


@app.post("/api/admin/people/gc-empty")
def api_admin_gc_empty_people(_: CurrentUser = Depends(require_role("admin"))):
    """Delete anonymous people rows with no faces / no photos (ghost rows).

    Merges sweep orphaned placeholders created by earlier face deletions that
    predate the automatic GC. Returns how many rows were removed; re-running
    is a no-op (idempotent).
    """
    deleted = delete_empty_people()
    _invalidate_people_cache()
    return {"deleted": deleted, "ok": True}


# --- admin: bridge SDK cache management -----------------------------------
#
# These let an admin recover from a stale Proton SDK cache without SSH.
# The bridge unlinks its on-disk cache files and exits with code 1; compose
# `restart: unless-stopped` then respawns the bridge with a fresh cache.
# See docs/reference/troubleshooting.md for the full story.

@app.get("/api/admin/bridge/cache")
def api_admin_bridge_cache_status(_: CurrentUser = Depends(require_role("admin"))):
    try:
        return get_bridge().cache_status()
    except Exception as exc:
        raise HTTPException(502, f"bridge cache lookup failed: {exc}")


@app.post("/api/admin/bridge/cache/clear")
def api_admin_bridge_cache_clear(_: CurrentUser = Depends(require_role("admin"))):
    """Tell the bridge to clear its SDK cache and restart itself.

    Returns immediately with the list of files removed (or the error from
    the bridge). The bridge exits ~500 ms after responding, so a follow-up
    GET /api/admin/bridge/cache will fail until compose has restarted the
    container (~5-10 s). That's the expected signal of a successful clear.
    """
    try:
        return get_bridge().clear_cache()
    except Exception as exc:
        raise HTTPException(502, f"bridge cache clear failed: {exc}")


# --- static UI -------------------------------------------------------------

app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")

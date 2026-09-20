"""FastAPI application: search API + static web UI.

This module is the thin aggregator for the refactored API (issue #108).
It owns:

- The ``FastAPI`` app instance, its lifespan, and middleware.
- The static-file mount for the SPA.
- Re-exports of every symbol that tests and conftest reach through
  ``api.<name>`` (route handlers, helpers, TTL caches, locks, the
  fullres semaphore, the indexer proxy client, etc.).

The actual route bodies live in sibling modules:

- ``api_state`` — module-level state (TTL caches, locks, fullres
  semaphore/backoff, indexer proxy client, invalidation helpers).
- ``api_common`` — cross-router pure helpers (refresh cookie, bearer
  extraction, signed URLs, row serialization, crop cache, image
  sniffing, people LRU, photo dups cache, stats/bridge health caches,
  duplicates payload, suggested rows, merge propagation, search
  helpers, admin user-row serializer, face crop bytes, face row
  lookup, binary endpoint helpers, indexer proxy helpers, anchors
  payload).
- ``api_routes_auth`` — auth + sign + 2FA handlers.
- ``api_routes_photos`` — photos list/archived/memories/duplicates/
  tags/anchors/albums/places/map/photo/meta.
- ``api_routes_binary`` — thumb/full/cover/crop.
- ``api_routes_people`` — people list/detail/cover/faces/set_cover/
  unassigned/face_crop/face_suggest/photo_faces/face_assign/
  face_unassign/rename/merge/similar/merge_all/merge_all_similar/
  duplicates/suggested_merges/person_photos/person_map.
- ``api_routes_search`` — search/face_search.
- ``api_routes_admin`` — users CRUD/2fa disable/overview/backup/
  backups/prune/schedule/sync/trigger/checks/gc_empty_people/
  bridge_cache.
- ``api_routes_status`` — health/stats/bridge_health/status.

The split is mechanical: every auth/security invariant (F-01…F-14) is
preserved and every test-coupling preserved. A handful of small,
intentional behavior deltas landed alongside the split rather than in a
separate PR: a new ``GET /api/bridge_health`` endpoint, a lower default
``limit`` for ``/api/search`` (100→60), and pagination (``offset``, and a
lower default ``limit``) on ``/api/faces/unassigned`` and
``/api/people/{id}/faces``.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

import clip
import faces

# --- Cross-router pure helpers --------------------------------------------
# Re-exported so tests can call `api._people_all_cached()`,
# `api._duplicate_groups_cached(200)`, `api._get_clip_matrix()`,
# `api._face_similarity(...)`, `api._topk_indices(...)`,
# `api._cached_stats()`, `api._cached_bridge_health()`,
# `api._indexer_proxy_json("GET", "/sync-config")`, etc.
from api_common import (  # noqa: F401  (re-exported for tests)
    _REFRESH_COOKIE,
    LIST_MAX_LIMIT,
    SEARCH_MAX_LIMIT,
    THRESHOLD_MIN,
    _anchors_payload,
    _bearer_is_valid,
    _cached_bridge_health,
    _cached_stats,
    _clamp_limit,
    _clamp_offset,
    _clamp_search_limit,
    _clamp_threshold,
    _clear_refresh_cookie,
    _crop_cache_path,
    _drop_crop_cache,
    _drop_people_crops,
    _drop_person_crops,
    _duplicate_groups_cached,
    _dups_payload,
    _extract_bearer,
    _face_crop_bytes,
    _face_row,
    _face_similarity,
    _fetch_remote_indexer_state,
    _get_clip_matrix,
    _indexer_is_local,
    _indexer_proxy_json,
    _merge_propagate,
    _merged_indexer_state,
    _people_all_cached,
    _people_cache_get_locked,
    _people_cache_put_locked,
    _row_to_dict,
    _semantic_search,
    _serve_face_crop,
    _serve_full,
    _serve_person_cover,
    _serve_thumb,
    _set_refresh_cookie,
    _sign_if_needed,
    _single_user_photo,
    _sniff_image_type,
    _suggested_rows,
    _topk_indices,
    _user_photos,
    _user_row_public,
    start_crop_prewarm_worker,
)
from api_routes_admin import (  # noqa: F401  (re-exported for tests)
    api_admin_backup,
    api_admin_backups,
    api_admin_bridge_cache,
    api_admin_bridge_cache_clear,
    api_admin_checks,
    api_admin_create_user,
    api_admin_db_compact,
    api_admin_delete_backup,
    api_admin_delete_user,
    api_admin_disable_2fa,
    api_admin_force_logout,
    api_admin_gc_empty_people,
    api_admin_get_schedule,
    api_admin_get_sync,
    api_admin_overview,
    api_admin_patch_user,
    api_admin_prune_backups,
    api_admin_prune_small_people,
    api_admin_set_schedule,
    api_admin_set_sync,
    api_admin_sync_trigger,
    api_admin_users,
)
from api_routes_admin import router as _admin_router

# --- Route handlers (re-exported so tests can call them directly) ---------
# Each router module defines handlers as module-level functions and
# registers them on its own APIRouter. We include_router each module
# below and re-export the handler functions so tests can call
# `api.api_people(...)`, `api.api_login(...)`, etc.
from api_routes_auth import (  # noqa: F401  (re-exported for tests)
    api_2fa_confirm,
    api_2fa_disable,
    api_2fa_setup,
    api_2fa_verify,
    api_change_password,
    api_limits,
    api_login,
    api_logout,
    api_me,
    api_refresh,
    api_sign,
)

# --- Router modules (include_router each one) -----------------------------
from api_routes_auth import router as _auth_router
from api_routes_binary import (  # noqa: F401  (re-exported for tests)
    api_face_crop,
    api_full,
    api_person_cover,
    api_thumb,
)
from api_routes_binary import router as _binary_router
from api_routes_people import (  # noqa: F401  (re-exported for tests)
    api_assign_face,
    api_face_suggest,
    api_merge_all,
    api_merge_all_similar,
    api_merge_people,
    api_people,
    api_people_duplicates,
    api_people_faces,
    api_people_similar,
    api_people_suggested_merges,
    api_person_map,
    api_person_photos,
    api_photo_faces,
    api_rename_person,
    api_set_person_cover,
    api_unassign_face,
    api_unassigned_faces,
)
from api_routes_people import router as _people_router
from api_routes_photos import (  # noqa: F401  (re-exported for tests)
    api_album_photos,
    api_albums,
    api_archived_photos,
    api_duplicates,
    api_get_tags,
    api_map,
    api_memories,
    api_patch_photo,
    api_photo,
    api_photo_anchors,
    api_photo_meta,
    api_photos,
    api_places,
    api_set_tags,
    api_tags,
)
from api_routes_photos import router as _photos_router
from api_routes_search import (  # noqa: F401  (re-exported for tests)
    api_search,
    api_search_face,
)
from api_routes_search import router as _search_router
from api_routes_status import (  # noqa: F401  (re-exported for tests)
    api_bridge_health,
    api_health,
    api_stats,
    api_status,
)
from api_routes_status import router as _status_router

# --- Module-level state (TTL caches, locks, fullres semaphore, etc.) ------
# Re-exported so tests can reset them via `api._dups_cache = None` etc.
# (see tests/conftest.py::_reset_module_state).
from api_state import (  # noqa: F401  (re-exported for tests)
    _ANCHORS_CACHE_TTL,
    _BRIDGE_HEALTH_CACHE_TTL,
    _CLIP_CACHE_TTL,
    _DUP_CACHE_TTL,
    _FULL_RES_FAILURE_LOG_LOCK,
    _FULL_RES_FAILURE_WINDOW_SEC,
    _FULL_SEMAPHORE_MAX,
    _FULL_TIMEOUT_SEC,
    _IMMUTABLE_HEADERS,
    _INDEXER_PROXY_CACHE_TTL,
    _INDEXER_PROXY_LOG_THROTTLE,
    _INDEXER_PROXY_TIMEOUT,
    _PEOPLE_CACHE_MAX,
    _PEOPLE_CACHE_TTL,
    _PHOTO_DUPS_CACHE_TTL,
    _STATS_CACHE_TTL,
    _SUGGESTED_CACHE_TTL,
    FACE_SEARCH_MAX_IMAGE_PIXELS,
    FACE_SEARCH_MAX_UPLOAD_BYTES,
    _anchors_cache,
    _bridge_health_cache,
    _bridge_health_cache_lock,
    _clip_cache,
    _clip_cache_lock,
    _crop_lock,
    _dups_cache,
    _dups_cache_lock,
    _full_res_failure_ts,
    _full_semaphore,
    _get_indexer_proxy_client,
    _indexer_proxy_cache,
    _indexer_proxy_client,
    _indexer_proxy_client_lock,
    _indexer_proxy_last_warn,
    _invalidate_clip_cache,
    _invalidate_dups_cache,
    _invalidate_people_cache,
    _invalidate_photo_dups_cache,
    _people_cache,
    _people_cache_lock,
    _photo_dups_cache,
    _photo_dups_cache_lock,
    _recent_full_res_failures,
    _record_full_res_failure,
    _stats_cache,
    _stats_cache_lock,
    _suggested_cache,
    _suggested_cache_lock,
)

# Re-export the auth TTL helpers used by the refresh-cookie helpers.
from auth import (  # noqa: F401  (re-exported for tests)
    access_ttl as auth_access_ttl,
)

# Re-export auth helpers that tests monkeypatch via `api.<name>`.
from auth import (  # noqa: F401  (re-exported for tests)
    demo_disable_admin_area,
    demo_disable_admin_user_management,
    hash_password,
    require_signing_secret,
    require_user,
)
from auth import (  # noqa: F401  (re-exported for tests)
    refresh_ttl as auth_refresh_ttl,
)
from body_limit import BodyLimitMiddleware

# Re-export bridge_client symbols that tests monkeypatch via `api.<name>`.
from bridge_client import get_bridge  # noqa: F401  (re-exported for tests)

# Re-export the ML modules so tests can monkeypatch `api.clip.warm_up` /
# `api.faces.warm_up` and the route bodies resolve through `api.<name>`.
from clip import embed_text  # noqa: F401  (re-exported for tests)
from compression import CompressionMiddleware
from config import settings
from faces import embed_query_face  # noqa: F401  (re-exported for tests)
from security_headers import SecurityHeadersMiddleware

# Re-export sidecar symbols that tests monkeypatch via `api.<name>`.
from sidecar import read_clip_sidecar  # noqa: F401  (re-exported for tests)

# Re-export store symbols that tests monkeypatch via `api.<name>`.
from store import (  # noqa: F401  (re-exported for tests)
    _embedding_cache_data,
    all_people,
    clip_count,
    duplicate_groups,
    face_counts_for_photos,
    favorite_uids,
    get_photos_batch,
    person_mean_embeddings_from_cache,
    stats,
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
    _cap_threadpool()
    warm_models()
    yield


def _cap_threadpool() -> None:
    """Bound the threadpool sync routes run on (anyio default: 40).

    Each thread keeps its own SQLite connection + page cache; 40 of them per
    worker was the largest single term in the app's idle RSS.
    """
    try:
        import anyio

        anyio.to_thread.current_default_thread_limiter().total_tokens = settings.api_threadpool_size
    except Exception as exc:  # noqa: BLE001 - never block boot on this
        log.warning("could not cap the threadpool: %s", exc)


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
app.add_middleware(SecurityHeadersMiddleware)
# Outermost: reject oversized bodies before anything else touches them. The
# face-search upload has its own byte cap (F-09) and is multipart, so it is
# exempt here.
app.add_middleware(
    BodyLimitMiddleware,
    max_bytes=settings.max_json_body_bytes,
    exempt_prefixes=("/api/search/face",),
)

_STATIC = Path(__file__).parent / "static"

# --- Include all routers ---------------------------------------------------
app.include_router(_auth_router)
app.include_router(_photos_router)
app.include_router(_binary_router)
app.include_router(_people_router)
app.include_router(_search_router)
app.include_router(_admin_router)
app.include_router(_status_router)

# --- Static UI -------------------------------------------------------------
app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")

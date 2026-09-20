"""Module-level state for the FastAPI app: TTL caches, locks, fullres backoff,
indexer proxy client, and the small set of constants shared across routers.

This module is intentionally side-effect-free at import time: every cache
starts empty, every lock is a fresh ``threading.Lock``, and the fullres
semaphore is sized from the ``FULL_SEMAPHORE_MAX`` env var. The aggregator
``api.py`` re-exports the names that tests and conftest reset between
tests, so the singleton-reset pattern in ``tests/conftest.py`` keeps
working unchanged.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import OrderedDict

import httpx
import numpy as np

# --- TTL constants ---------------------------------------------------------

_DUP_CACHE_TTL = 600.0
_SUGGESTED_CACHE_TTL = 600.0
_PHOTO_DUPS_CACHE_TTL = 30.0
_ANCHORS_CACHE_TTL = 60.0
_PEOPLE_CACHE_TTL = 10.0
_PEOPLE_CACHE_MAX = 32
_CLIP_CACHE_TTL = 60.0

# --- Face search limits (F-09) --------------------------------------------
# Decompression-bomb guard: cap the raw upload size and the decoded pixel
# count so a single /api/search/face request can't amplify into a worker-
# wide CPU/memory DoS. Defaults: 8 MB upload, 50 M pixels (≈ 7000×7000).
FACE_SEARCH_MAX_UPLOAD_BYTES = int(
    os.environ.get("FACE_SEARCH_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024))
)
FACE_SEARCH_MAX_IMAGE_PIXELS = int(
    os.environ.get("FACE_SEARCH_MAX_IMAGE_PIXELS", str(50_000_000))
)
_STATS_CACHE_TTL = 5.0
_BRIDGE_HEALTH_CACHE_TTL = 30.0
_INDEXER_PROXY_CACHE_TTL = 30.0
_INDEXER_PROXY_TIMEOUT = 3.0
_INDEXER_PROXY_LOG_THROTTLE = 30.0

# --- Fullres admission control + failure log -------------------------------

_FULL_TIMEOUT_SEC = 30.0
_FULL_SEMAPHORE_MAX = int(os.environ.get("FULL_SEMAPHORE_MAX", "8"))
_full_semaphore = asyncio.Semaphore(_FULL_SEMAPHORE_MAX)

_FULL_RES_FAILURE_WINDOW_SEC = 15 * 60
_full_res_failure_ts: list[float] = []
_FULL_RES_FAILURE_LOG_LOCK = threading.Lock()


def _record_full_res_failure() -> None:
    cutoff = time.time() - _FULL_RES_FAILURE_WINDOW_SEC
    with _FULL_RES_FAILURE_LOG_LOCK:
        _full_res_failure_ts.append(time.time())
        if _full_res_failure_ts and _full_res_failure_ts[0] < cutoff:
            _full_res_failure_ts[:] = [t for t in _full_res_failure_ts if t >= cutoff]


def _recent_full_res_failures() -> int:
    cutoff = time.time() - _FULL_RES_FAILURE_WINDOW_SEC
    with _FULL_RES_FAILURE_LOG_LOCK:
        while _full_res_failure_ts and _full_res_failure_ts[0] < cutoff:
            _full_res_failure_ts.pop(0)
        return len(_full_res_failure_ts)


# --- TTL caches (all start empty) ------------------------------------------

_dups_cache: tuple[float, dict] | None = None
_dups_cache_lock = threading.Lock()

_suggested_cache: dict[float, tuple[float, list]] = {}
_suggested_cache_lock = threading.Lock()

_photo_dups_cache: tuple[float, dict[int, list]] | None = None
_photo_dups_cache_lock = threading.Lock()

_anchors_cache: tuple[float, dict] | None = None

_people_cache: OrderedDict[str | None, tuple[float, list]] = OrderedDict()
_people_cache_lock = threading.Lock()

_stats_cache: tuple[float, dict] | None = None
_stats_cache_lock = threading.Lock()

_clip_cache: tuple[float, int, list[str], np.ndarray] | None = None
_clip_cache_lock = threading.Lock()

_bridge_health_cache: tuple[float, tuple[bool, bool]] | None = None
_bridge_health_cache_lock = threading.Lock()

_indexer_proxy_cache: tuple[float, dict] | None = None
_indexer_proxy_last_warn: float = 0.0

# --- Indexer proxy client (lazy singleton) ---------------------------------

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


# --- Crop cache lock + immutable headers -----------------------------------

_crop_lock = threading.Lock()
_IMMUTABLE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}


# --- Cache invalidation helpers --------------------------------------------

def _invalidate_dups_cache() -> None:
    # Mutate through api's live namespace, not this module's own `global`:
    # every reader/writer of these caches (api_common, api_routes_people,
    # api_routes_photos) goes through `api.<name>`, which is a separate
    # binding from this module's own once the aggregator re-exports it.
    # Rebinding the bare name here would silently no-op from their
    # perspective (see issue #108 code review).
    import api
    api._dups_cache = None
    api._suggested_cache.clear()


def _invalidate_photo_dups_cache() -> None:
    import api
    api._photo_dups_cache = None


def _invalidate_people_cache() -> None:
    import api
    api._people_cache.clear()


def _invalidate_clip_cache() -> None:
    import api
    api._clip_cache = None

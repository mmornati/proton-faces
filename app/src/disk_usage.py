"""Shared directory-size helpers with a long-TTL cache.

The thumb dir can hold tens of thousands of files, so a full recursive
`stat()` walk is expensive. Both `/api/status` and the admin overview
need the same byte counts; keeping the walk + cache in one module means
the two callers share a single cache instead of each paying for their
own walk. The indexer process computes the authoritative values once per
status call and ships them in the proxied payload; this module is the
fallback for when the indexer is unreachable (and the source for the
indexer's own computation).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

_DIRSIZE_CACHE_TTL = 3600.0
_dirsize_cache: dict[str, tuple[float, int]] = {}


_dirsize_lock = threading.Lock()
_dirsize_refreshing: set[str] = set()


def dir_size_bytes(path: Path) -> int:
    """Best-effort recursive directory size in bytes."""
    if not path.exists():
        return 0
    total = 0
    try:
        for entry in path.iterdir():
            try:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += dir_size_bytes(entry)
            except OSError:
                continue
    except OSError:
        return total
    return total


def cached_dir_size(path: Path) -> int:
    """Disk-walk with a 1 h TTL — the thumb dir has tens of thousands of
    files and a full `stat()` walk is expensive when polled frequently."""
    key = str(path)
    now = time.time()
    hit = _dirsize_cache.get(key)
    if hit is not None and now - hit[0] < _DIRSIZE_CACHE_TTL:
        return hit[1]
    if hit is None:
        # First call: nothing to serve yet, so pay for the walk once.
        n = dir_size_bytes(path)
        _dirsize_cache[key] = (time.time(), n)
        return n
    # Expired: answer with the last value and refresh off the request path.
    # Walking 80k thumbnails inline made the indexer's /status miss the
    # app's 3 s proxy timeout once an hour, and every poll during the walk
    # started another one.
    with _dirsize_lock:
        if key not in _dirsize_refreshing:
            _dirsize_refreshing.add(key)
            threading.Thread(target=_refresh_dir_size, args=(path, key), name="dirsize", daemon=True).start()
    return hit[1]


def _refresh_dir_size(path: Path, key: str) -> None:
    try:
        _dirsize_cache[key] = (time.time(), dir_size_bytes(path))
    finally:
        with _dirsize_lock:
            _dirsize_refreshing.discard(key)

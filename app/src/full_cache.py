"""On-disk cache of full-resolution images fetched through the bridge.

Every /full request used to re-download and re-decrypt the original from
Proton (0.5 s typical, tens of seconds when the bridge queue is busy). Images
that went through once are kept under ``data/fullcache/<uid><ext>`` and served
as plain files afterwards. Videos are never cached (size, Range-driven
playback). The directory is bounded by FULL_CACHE_MAX_MB: least recently
served files go first. FULL_CACHE_MAX_MB=0 disables the cache.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path

from config import settings

log = logging.getLogger("full_cache")

_EXT_BY_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/avif": ".avif",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
}
_TYPE_BY_EXT = {ext: ct for ct, ext in _EXT_BY_TYPE.items()}

# One original larger than this is streamed but not kept.
MAX_ENTRY_BYTES = 64 * 1024 * 1024
_EVICT_INTERVAL_SEC = 60.0
_EVICT_TARGET_RATIO = 0.9

_evict_lock = threading.Lock()
_last_evict_ts = 0.0


def _cap_bytes() -> int:
    return max(0, settings.full_cache_max_mb) * 1024 * 1024


def enabled() -> bool:
    return _cap_bytes() > 0


def cacheable_type(content_type: str) -> bool:
    return content_type in _EXT_BY_TYPE


def lookup(uid: str) -> tuple[Path, str] | None:
    """Cached file and its content type, or None. A hit refreshes the LRU clock."""
    if not enabled():
        return None
    for ext, ct in _TYPE_BY_EXT.items():
        p = settings.full_cache_dir / f"{uid}{ext}"
        try:
            os.utime(p)
        except FileNotFoundError:
            continue
        except OSError:
            return None
        return p, ct
    return None


class Writer:
    """Tee for a streamed response: collects chunks into a temp file and
    publishes it atomically only once the whole body arrived."""

    def __init__(self, uid: str, content_type: str) -> None:
        self.final = settings.full_cache_dir / f"{uid}{_EXT_BY_TYPE[content_type]}"
        self.tmp = settings.full_cache_dir / f".{uid}.{uuid.uuid4().hex}.part"
        self.size = 0
        self._fh = None
        self._failed = False
        try:
            self._fh = open(self.tmp, "wb")
        except OSError as exc:
            log.warning("full cache: cannot open %s: %s", self.tmp, exc)
            self._failed = True

    def write(self, chunk: bytes) -> None:
        if self._failed:
            return
        self.size += len(chunk)
        if self.size > MAX_ENTRY_BYTES:
            self.abort()
            return
        try:
            self._fh.write(chunk)
        except OSError as exc:
            log.warning("full cache: write failed for %s: %s", self.final.name, exc)
            self.abort()

    def commit(self, expected_len: int | None) -> bool:
        if self._failed:
            return False
        if self.size == 0 or (expected_len is not None and expected_len != self.size):
            self.abort()
            return False
        try:
            self._fh.close()
            os.replace(self.tmp, self.final)
        except OSError as exc:
            log.warning("full cache: publish failed for %s: %s", self.final.name, exc)
            self.abort()
            return False
        self._fh = None
        evict_if_needed()
        return True

    def abort(self) -> None:
        self._failed = True
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        try:
            self.tmp.unlink(missing_ok=True)
        except OSError:
            pass


def writer(uid: str, content_type: str) -> Writer | None:
    if not enabled() or not cacheable_type(content_type):
        return None
    w = Writer(uid, content_type)
    return None if w._failed else w


def evict_if_needed(force: bool = False) -> int:
    """Delete least recently served files until the cache is under 90 % of its
    cap. Throttled to one directory scan per minute per process. Returns the
    number of files removed."""
    global _last_evict_ts
    now = time.time()
    if not force and now - _last_evict_ts < _EVICT_INTERVAL_SEC:
        return 0
    if not _evict_lock.acquire(blocking=False):
        return 0
    try:
        _last_evict_ts = now
        cap = _cap_bytes()
        entries = []
        total = 0
        with os.scandir(settings.full_cache_dir) as it:
            for e in it:
                try:
                    st = e.stat()
                except OSError:
                    continue
                if e.name.endswith(".part"):
                    # Orphans of a crashed process; live temps are younger.
                    if now - st.st_mtime > 3600:
                        Path(e.path).unlink(missing_ok=True)
                    continue
                entries.append((st.st_mtime, st.st_size, e.path))
                total += st.st_size
        if total <= cap:
            return 0
        target = int(cap * _EVICT_TARGET_RATIO)
        removed = 0
        for _mtime, size, path in sorted(entries):
            if total <= target:
                break
            try:
                os.unlink(path)
            except OSError:
                continue
            total -= size
            removed += 1
        log.info("full cache: evicted %d files, %d MB left", removed, total // (1024 * 1024))
        return removed
    except OSError as exc:
        log.warning("full cache: eviction scan failed: %s", exc)
        return 0
    finally:
        _evict_lock.release()


def remove(uids: list[str]) -> None:
    """Drop cached originals of photos deleted from Proton."""
    for uid in uids:
        for ext in _TYPE_BY_EXT:
            try:
                (settings.full_cache_dir / f"{uid}{ext}").unlink(missing_ok=True)
            except OSError:
                pass

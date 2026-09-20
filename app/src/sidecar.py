"""Shared mmap sidecar for face embedding and CLIP matrices.

The indexer writes flat numpy arrays + metadata JSON files atomically
(tmp+rename). API workers open them with ``np.load(…, mmap_mode='r')`` so
the OS page cache shares physical pages across all workers.

Fallback: when sidecar files are absent (first run before indexer upgrade),
callers fall through to the DB-based cache.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from config import settings

log = logging.getLogger(__name__)

# --- paths -----------------------------------------------------------------

_SIDECAR_DIR: Path | None = None
_sidcar_dir_lock = threading.Lock()


def _sidecar_dir() -> Path:
    global _SIDECAR_DIR
    if _SIDECAR_DIR is None:
        with _sidcar_dir_lock:
            if _SIDECAR_DIR is None:
                _SIDECAR_DIR = Path(
                    os.environ.get("SIDECAR_DIR", str(settings.data_dir / "index"))
                ).resolve()
                _SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    return _SIDECAR_DIR


def _set_sidecar_dir(path: str | Path) -> None:
    """Override sidecar directory (for tests)."""
    global _SIDECAR_DIR
    _SIDECAR_DIR = Path(path).resolve()
    _SIDECAR_DIR.mkdir(parents=True, exist_ok=True)


# --- face embedding sidecar ------------------------------------------------

_FACE_META_PATH = "embeddings_meta.json"
_FACE_MAT_PATH = "embeddings.npy"
_FACE_UIDS_PATH = "embeddings_uids.npy"

# Module-level mmap handles (one per process — each worker has its own).
_face_mmap: dict | None = None
_face_mmap_ts = 0.0
_face_mmap_lock = threading.Lock()


def read_face_sidecar() -> dict | None:
    """Return cached face matrix + metadata from mmap, or None if stale/missing.

    Returns dict with keys: ids (int64 ndarray), photo_uids (list[str]),
    person_ids (list[int|None]), mat (float32 ndarray, mmap'd).
    """
    global _face_mmap, _face_mmap_ts
    now = time.time()
    if _face_mmap is not None and now - _face_mmap_ts < 120.0:
        return _face_mmap
    with _face_mmap_lock:
        now = time.time()
        if _face_mmap is not None and now - _face_mmap_ts < 120.0:
            return _face_mmap
        meta = _read_meta(_FACE_META_PATH)
        if meta is None:
            _face_mmap = None
            _face_mmap_ts = 0.0
            return None
        mat_path = _sidecar_dir() / _FACE_MAT_PATH
        uids_path = _sidecar_dir() / _FACE_UIDS_PATH
        if not mat_path.exists() or not uids_path.exists():
            _face_mmap = None
            _face_mmap_ts = 0.0
            return None
        try:
            mat = np.load(str(mat_path), mmap_mode="r")
            uids_data = np.load(str(uids_path), mmap_mode="r")
            # uids_data is a fixed-width bytes array (N, max_len); decode each row
            photo_uids = [
                row.tobytes().decode("utf-8").rstrip("\x00")
                for row in uids_data
            ]
            ids = np.asarray(meta["ids"], dtype=np.int64)
            person_ids = meta["person_ids"]
            n = int(mat.shape[0])
            if len(ids) != n or len(person_ids) != n or len(photo_uids) != n:
                # A reader that lands between two sidecar generations (or a
                # torn write) would otherwise index face ids / person ids by
                # a row offset from a different matrix — silently assigning
                # faces to the wrong people. Serve nothing instead.
                log.warning(
                    "face sidecar mismatch: meta=%d uids=%d mat=%d — ignoring",
                    len(ids), len(photo_uids), n,
                )
                _face_mmap = None
                _face_mmap_ts = 0.0
                return None
            _face_mmap = {
                "ids": ids,
                "photo_uids": photo_uids,
                "person_ids": person_ids,
                "mat": mat,
            }
            _face_mmap_ts = now
            return _face_mmap
        except Exception:
            log.warning("failed to mmap face sidecar", exc_info=True)
            _face_mmap = None
            _face_mmap_ts = 0.0
            return None


def write_face_sidecar(
    ids: list[int],
    photo_uids: list[str],
    person_ids: list[int | None],
    mat: np.ndarray,
) -> None:
    """Write the face sidecar as a set: every temp file first, then the renames.

    Each rename is atomic on its own, but a reader on another process could
    land between two of them. All three temps are fully written (and
    fsync'd) before any rename, and the metadata — the file readers open
    first — is renamed LAST, so a reader that sees the new meta also sees
    the new matrix. ``read_face_sidecar`` additionally refuses mismatched
    lengths as a second line of defence.
    """
    if len(ids) != mat.shape[0] or len(photo_uids) != mat.shape[0] or len(person_ids) != mat.shape[0]:
        raise ValueError("face sidecar inputs disagree on row count")
    d = _sidecar_dir()
    meta = {
        "count": len(ids),
        "generated_at": time.time(),
        "ids": ids,
        "person_ids": person_ids,
    }
    # Store uids as fixed-width bytes for mmap-friendly loading
    max_len = max((len(u) for u in photo_uids), default=0) + 1  # +1 for null terminator
    uids_arr = np.zeros((len(photo_uids), max_len), dtype=np.uint8)
    for i, u in enumerate(photo_uids):
        encoded = u.encode("utf-8")
        uids_arr[i, : len(encoded)] = list(encoded)
    tmp_mat = _stage_npy(d / _FACE_MAT_PATH, mat)
    tmp_uids = _stage_npy(d / _FACE_UIDS_PATH, uids_arr)
    tmp_meta = _stage_bytes(d / _FACE_META_PATH, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    os.replace(tmp_mat, d / _FACE_MAT_PATH)
    os.replace(tmp_uids, d / _FACE_UIDS_PATH)
    os.replace(tmp_meta, d / _FACE_META_PATH)
    _fsync_dir(d)


# --- CLIP sidecar ----------------------------------------------------------

_CLIP_META_PATH = "clip_meta.json"
_CLIP_MAT_PATH = "clip_matrix.npy"

_clip_mmap: tuple[float, int, list[str], np.ndarray] | None = None
_clip_mmap_lock = threading.Lock()


def read_clip_sidecar() -> tuple[list[str], np.ndarray] | None:
    """Return (uids, X) from mmap'd CLIP matrix, or None if stale/missing."""
    global _clip_mmap
    now = time.time()
    if _clip_mmap is not None:
        ts, n_cached, uids, X = _clip_mmap
        if (now - ts) < 60.0:
            return uids, X
    with _clip_mmap_lock:
        now = time.time()
        if _clip_mmap is not None:
            ts, n_cached, uids, X = _clip_mmap
            if (now - ts) < 60.0:
                return uids, X
        meta = _read_meta(_CLIP_META_PATH)
        if meta is None:
            _clip_mmap = None
            return None
        mat_path = _sidecar_dir() / _CLIP_MAT_PATH
        if not mat_path.exists():
            _clip_mmap = None
            return None
        try:
            X = np.load(str(mat_path), mmap_mode="r")
            uids = meta["uids"]
            n = meta["count"]
            if len(uids) != int(X.shape[0]):
                log.warning("clip sidecar mismatch: meta=%d mat=%d — ignoring", len(uids), X.shape[0])
                _clip_mmap = None
                return None
            _clip_mmap = (now, n, uids, X)
            return uids, X
        except Exception:
            log.warning("failed to mmap CLIP sidecar", exc_info=True)
            _clip_mmap = None
            return None


def write_clip_sidecar(uids: list[str], X: np.ndarray) -> None:
    """Write the CLIP sidecar as a set (see write_face_sidecar): matrix first, meta last."""
    if len(uids) != X.shape[0]:
        raise ValueError("clip sidecar inputs disagree on row count")
    d = _sidecar_dir()
    meta = {
        "count": len(uids),
        "generated_at": time.time(),
        "uids": uids,
    }
    tmp_mat = _stage_npy(d / _CLIP_MAT_PATH, X)
    tmp_meta = _stage_bytes(d / _CLIP_META_PATH, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    os.replace(tmp_mat, d / _CLIP_MAT_PATH)
    os.replace(tmp_meta, d / _CLIP_META_PATH)
    _fsync_dir(d)


# --- internal helpers ------------------------------------------------------


def _read_meta(name: str) -> dict | None:
    path = _sidecar_dir() / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_bytes())
    except Exception:
        log.warning("failed to read sidecar meta %s", name, exc_info=True)
        return None


def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    """Persist the renames themselves; a power loss otherwise can leave the
    old directory entry pointing at a zero-length inode."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _stage_bytes(path: Path, data: bytes) -> Path:
    """Write `data` to a temp sibling of `path`, fsync it, return the temp path."""
    tmp = path.with_suffix(".tmp" + path.suffix)
    tmp.write_bytes(data)
    _fsync_file(tmp)
    return tmp


def _stage_npy(path: Path, arr: np.ndarray) -> Path:
    tmp = path.with_suffix(".tmp.npy")
    np.save(str(tmp), arr)
    _fsync_file(tmp)
    return tmp


def invalidate_face_cache() -> None:
    """Drop the cached face sidecar mmap so the next read re-maps from disk.

    Called by `store.invalidate_embedding_cache` after a merge or face
    reassignment changes the face->person mapping. The face matrix itself is
    immutable, but its `person_ids` column is baked into the metadata the
    indexer rewrites on its own debounced schedule; dropping the mmap here
    lets a subsequent read pick up the indexer's rewrite as soon as it lands.
    Keeps `_SIDECAR_DIR` (and the CLIP mmap) untouched.
    """
    global _face_mmap, _face_mmap_ts
    with _face_mmap_lock:
        _face_mmap = None
        _face_mmap_ts = 0.0


# --- test helpers ----------------------------------------------------------


def reset_state() -> None:
    """Clear module-level caches (called from test teardown)."""
    global _face_mmap, _face_mmap_ts, _clip_mmap, _SIDECAR_DIR
    _face_mmap = None
    _face_mmap_ts = 0.0
    _clip_mmap = None
    _SIDECAR_DIR = None

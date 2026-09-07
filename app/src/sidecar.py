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
    """Write face embedding sidecar files atomically (tmp+rename)."""
    d = _sidecar_dir()
    meta = {
        "count": len(ids),
        "generated_at": time.time(),
        "ids": ids,
        "person_ids": person_ids,
    }
    _write_atomic(d / _FACE_META_PATH, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    _write_atomic_npy(d / _FACE_MAT_PATH, mat)
    # Store uids as fixed-width bytes for mmap-friendly loading
    max_len = max((len(u) for u in photo_uids), default=0) + 1  # +1 for null terminator
    uids_arr = np.zeros((len(photo_uids), max_len), dtype=np.uint8)
    for i, u in enumerate(photo_uids):
        encoded = u.encode("utf-8")
        uids_arr[i, : len(encoded)] = list(encoded)
    _write_atomic_npy(d / _FACE_UIDS_PATH, uids_arr)


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
            _clip_mmap = (now, n, uids, X)
            return uids, X
        except Exception:
            log.warning("failed to mmap CLIP sidecar", exc_info=True)
            _clip_mmap = None
            return None


def write_clip_sidecar(uids: list[str], X: np.ndarray) -> None:
    """Write CLIP matrix sidecar files atomically (tmp+rename)."""
    d = _sidecar_dir()
    meta = {
        "count": len(uids),
        "generated_at": time.time(),
        "uids": uids,
    }
    _write_atomic(d / _CLIP_META_PATH, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    _write_atomic_npy(d / _CLIP_MAT_PATH, X)


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


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(".tmp" + path.suffix)
    tmp.write_bytes(data)
    tmp.rename(path)


def _write_atomic_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(str(tmp), arr)
    tmp.rename(path)


# --- test helpers ----------------------------------------------------------


def reset_state() -> None:
    """Clear module-level caches (called from test teardown)."""
    global _face_mmap, _face_mmap_ts, _clip_mmap, _SIDECAR_DIR
    _face_mmap = None
    _face_mmap_ts = 0.0
    _clip_mmap = None
    _SIDECAR_DIR = None

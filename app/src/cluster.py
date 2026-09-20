"""Incremental people clustering over ArcFace face embeddings.

Faces that have no person_id are clustered with HDBSCAN (euclidean distance
on L2-normalized embeddings). For unit-norm vectors euclidean distance is a
monotone transform of cosine distance (d² = 2 − 2·cos), so the hierarchy is
identical to the previous cosine formulation while unlocking BLAS-optimized
euclidean paths. Each resulting cluster becomes a person row; subsequent runs
only look at faces still lacking a person, so named people are never disturbed.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
from sklearn.cluster import HDBSCAN

from config import settings
from store import (
    assign_faces_person_bulk,
    create_person,
    faces_without_person,
    person_mean_embeddings,
)

log = logging.getLogger("cluster")


def _decode(row) -> np.ndarray:
    return np.frombuffer(row["embedding"], dtype=np.float32)


# Cache of person mean embeddings for the worker-time matching path. Only
# person-assigned faces are loaded (small vs the API's all-face matrix), and
# the means are tiny. A full reload fetches ~180 k face embeddings (~360 MB)
# from SQLite, so it must never stall the recognition workers: once loaded we
# serve the stale cache immediately while a single background thread refreshes
# (stale-while-revalidate, same pattern as the API embedding cache).
_PERSON_MEANS_TTL = 300.0
_person_means: dict[int, np.ndarray] | None = None
# (pids, mat) stacked once when the cache is built so match_person can do a
# single matrix-vector product instead of a per-person Python loop. `pids[i]`
# holds the person_id whose mean embedding is row `i` of `mat`. Published as
# ONE tuple, together with the dict, under `_person_means_lock`: two separate
# global writes let a worker argmax against the old matrix and index the new
# pids — a face silently attached to the wrong person after every merge.
_person_means_stack: tuple[np.ndarray, np.ndarray] | None = None
_person_means_ts = 0.0
_person_means_lock = threading.Lock()
_person_means_refreshing = False


def _build_person_means() -> tuple[dict[int, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """Fetch {person_id: L2-normalized mean embedding} plus its stacked
    (pids, mat) form. Pure: touches no globals, so it can run WITHOUT
    `_person_means_lock` while readers keep using the previous generation;
    the caller publishes the result under the lock in one assignment."""
    means = person_mean_embeddings()
    if not means:
        return means, None
    pids = np.array(list(means.keys()), dtype=np.int64)
    mat = np.stack(list(means.values())).astype(np.float32)
    return means, (pids, mat)


def _publish_person_means(means: dict[int, np.ndarray],
                          stack: tuple[np.ndarray, np.ndarray] | None) -> None:
    """Swap in a new generation. Caller must hold `_person_means_lock`."""
    global _person_means, _person_means_stack, _person_means_ts
    _person_means = means
    _person_means_stack = stack
    _person_means_ts = time.time()


def _background_refresh_person_means() -> None:
    global _person_means_refreshing
    try:
        means, stack = _build_person_means()
    except Exception:
        log.exception("background person-means refresh failed")
        with _person_means_lock:
            _person_means_refreshing = False
        return
    with _person_means_lock:
        _publish_person_means(means, stack)
        _person_means_refreshing = False


def _person_means_cached():
    """Lazily load {person_id: L2-normalized mean embedding} with a TTL.

    Builds the stacked (P, 512) matrix once so match_person can vectorize.
    Returns the dict for compatibility. Expired caches are served stale while
    a single background thread refreshes, so workers never stall.
    """
    global _person_means_refreshing
    now = time.time()
    if _person_means is not None and now - _person_means_ts < _PERSON_MEANS_TTL:
        return _person_means
    if _person_means is None:
        # First load: synchronous, we have nothing to serve stale.
        with _person_means_lock:
            if _person_means is None:
                means, stack = _build_person_means()
                _publish_person_means(means, stack)
        return _person_means
    # Expired but we have a stale cache: serve it and refresh in the background.
    with _person_means_lock:
        if _person_means_refreshing:
            return _person_means
        _person_means_refreshing = True
    threading.Thread(target=_background_refresh_person_means, daemon=True).start()
    return _person_means


def match_person(embedding: bytes, threshold: float) -> int | None:
    """Return the person_id whose mean embedding best matches `embedding`, or
    None when no existing person scores at or above `threshold`.

    Used at ingest time so a newly-detected face is assigned to an existing
    person immediately instead of waiting to be re-clustered (and possibly
    forming a duplicate person). Both vectors are L2-normalized, so the dot
    product is the cosine similarity.
    """
    _person_means_cached()
    stack = _person_means_stack  # one read: pids and mat from the same generation
    if stack is None:
        return None
    pids, mat = stack
    emb = np.frombuffer(embedding, dtype=np.float32)
    sims = mat @ emb
    i = int(np.argmax(sims))
    best_sim = float(sims[i])
    if best_sim < threshold:
        return None
    return int(pids[i])


def cluster_once(max_faces: int = 5000) -> int:
    """Cluster faces that have no person yet. Returns number of people created.

    `max_faces` caps how many unassigned faces one run clusters; 0 or a
    negative value means no cap (use with care on large libraries).
    """
    rows = faces_without_person(limit=max_faces if max_faces > 0 else None)
    if len(rows) < settings.min_cluster_size:
        return 0

    X = np.stack([_decode(r) for r in rows]).astype(np.float32)
    # Cheap insurance before the euclidean run: ArcFace already emits
    # L2-normalized embeddings, but re-normalizing keeps the euclidean/cosine
    # equivalence exact even if a row was written by an older path.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms
    labels = HDBSCAN(
        min_cluster_size=settings.min_cluster_size,
        # min_samples > 1 suppresses singleton/pair clusters that HDBSCAN
        # would otherwise emit as noise-or-cluster when every point is a
        # cluster core (the min_samples=1 default). Defaults to 2.
        # Existing people rows are never re-clustered (we only cluster
        # faces_without_person), so this only affects new clusters.
        min_samples=settings.min_samples,
        # euclidean on unit-norm vectors is monotone-equivalent to cosine and
        # unlocks kd/ball-tree + BLAS paths instead of brute-force pairwise
        # cosine (the pre-change bottleneck on 5 k×512 runs).
        metric="euclidean",
    ).fit_predict(X)

    n_created = 0
    for label in np.unique(labels):
        if label == -1:
            continue
        idxs = np.where(labels == label)[0]
        if len(idxs) < settings.min_cluster_size:
            continue
        cover_row = rows[idxs[0]]
        person_id = create_person(
            name=None,
            cover_uid=cover_row["photo_uid"],
            cover_face_id=cover_row["id"],
        )
        assign_faces_person_bulk([rows[i]["id"] for i in idxs], person_id)
        n_created += 1
        log.debug("cluster -> person %s with %d faces", person_id, len(idxs))

    if n_created:
        log.info("clustering created %d people from %d faces", n_created, len(rows))
    return n_created

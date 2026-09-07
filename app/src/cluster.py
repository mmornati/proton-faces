"""Incremental people clustering over ArcFace face embeddings.

Faces that have no person_id are clustered with HDBSCAN (cosine distance).
Each resulting cluster becomes a person row; subsequent runs only look at
faces still lacking a person, so named people are never disturbed.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
from sklearn.cluster import HDBSCAN

from config import settings
from store import (
    assign_face_person,
    create_person,
    faces_without_person,
    person_mean_embeddings,
    set_person_cover_face,
    update_person_cover,
)

log = logging.getLogger("cluster")


def _decode(row) -> np.ndarray:
    return np.frombuffer(row["embedding"], dtype=np.float32)


# Cache of person mean embeddings for the worker-time matching path. Only
# person-assigned faces are loaded (small vs the API's all-face matrix), and
# the means are tiny, so a short TTL keeps new people visible within a couple
# of minutes without hammering SQLite per photo.
_PERSON_MEANS_TTL = 300.0
_person_means: dict[int, np.ndarray] | None = None
# Parallel arrays stacked once when the cache is built so match_person can do
# a single matrix-vector product instead of a per-person Python loop. `pids[i]`
# holds the person_id whose mean embedding is row `i` of `mat`.
_person_means_pids: np.ndarray | None = None
_person_means_mat: np.ndarray | None = None
_person_means_ts = 0.0
_person_means_lock = threading.Lock()


def _person_means_cached():
    """Lazily load {person_id: L2-normalized mean embedding} with a TTL.

    Builds the stacked (P, 512) matrix once so match_person can vectorize.
    Returns the dict for compatibility.
    """
    global _person_means, _person_means_pids, _person_means_mat, _person_means_ts
    now = time.time()
    if _person_means is not None and now - _person_means_ts < _PERSON_MEANS_TTL:
        return _person_means
    with _person_means_lock:
        now = time.time()
        if _person_means is not None and now - _person_means_ts < _PERSON_MEANS_TTL:
            return _person_means
        _person_means = person_mean_embeddings()
        if _person_means:
            _person_means_pids = np.array(list(_person_means.keys()), dtype=np.int64)
            _person_means_mat = np.stack(list(_person_means.values())).astype(np.float32)
        else:
            _person_means_pids = None
            _person_means_mat = None
        _person_means_ts = now
        return _person_means


def match_person(embedding: bytes, threshold: float) -> int | None:
    """Return the person_id whose mean embedding best matches `embedding`, or
    None when no existing person scores at or above `threshold`.

    Used at ingest time so a newly-detected face is assigned to an existing
    person immediately instead of waiting to be re-clustered (and possibly
    forming a duplicate person). Both vectors are L2-normalized, so the dot
    product is the cosine similarity.
    """
    means = _person_means_cached()
    if not means:
        return None
    emb = np.frombuffer(embedding, dtype=np.float32)
    sims = _person_means_mat @ emb
    i = int(np.argmax(sims))
    best_sim = float(sims[i])
    if best_sim < threshold:
        return None
    return int(_person_means_pids[i])


def cluster_once(max_faces: int = 5000) -> int:
    """Cluster faces that have no person yet. Returns number of people created."""
    rows = faces_without_person(limit=max_faces)
    if len(rows) < settings.min_cluster_size:
        return 0

    X = np.stack([_decode(r) for r in rows]).astype(np.float32)
    labels = HDBSCAN(
        min_cluster_size=settings.min_cluster_size,
        # min_samples > 1 suppresses singleton/pair clusters that HDBSCAN
        # would otherwise emit as noise-or-cluster when every point is a
        # cluster core (the min_samples=1 default). Defaults to 2.
        # Existing people rows are never re-clustered (we only cluster
        # faces_without_person), so this only affects new clusters.
        min_samples=settings.min_samples,
        metric="cosine",
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
        update_person_cover(person_id, cover_row["photo_uid"])
        set_person_cover_face(person_id, cover_row["id"])
        for i in idxs:
            assign_face_person(rows[i]["id"], person_id)
        n_created += 1
        log.debug("cluster -> person %s with %d faces", person_id, len(idxs))

    if n_created:
        log.info("clustering created %d people from %d faces", n_created, len(rows))
    return n_created

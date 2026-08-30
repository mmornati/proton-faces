"""Incremental people clustering over ArcFace face embeddings.

Faces that have no person_id are clustered with HDBSCAN (cosine distance).
Each resulting cluster becomes a person row; subsequent runs only look at
faces still lacking a person, so named people are never disturbed.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import HDBSCAN

from config import settings
from store import (
    assign_face_person,
    create_person,
    faces_without_person,
    update_person_cover,
)

log = logging.getLogger("cluster")


def _decode(row) -> np.ndarray:
    return np.frombuffer(row["embedding"], dtype=np.float32)


def cluster_once(max_faces: int = 5000) -> int:
    """Cluster faces that have no person yet. Returns number of people created."""
    rows = faces_without_person(limit=max_faces)
    if len(rows) < settings.min_cluster_size:
        return 0

    X = np.stack([_decode(r) for r in rows]).astype(np.float32)
    labels = HDBSCAN(
        min_cluster_size=settings.min_cluster_size,
        min_samples=1,
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
        person_id = create_person(name=None, cover_uid=cover_row["photo_uid"])
        update_person_cover(person_id, cover_row["photo_uid"])
        for i in idxs:
            assign_face_person(rows[i]["id"], person_id)
        n_created += 1
        log.debug("cluster -> person %s with %d faces", person_id, len(idxs))

    if n_created:
        log.info("clustering created %d people from %d faces", n_created, len(rows))
    return n_created
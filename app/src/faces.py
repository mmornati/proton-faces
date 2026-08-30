"""Face detection + embedding via InsightFace (RetinaFace + ArcFace).

Runs on CPU. The buffalo_l model bundle provides detection, landmark
alignment and 512-d face embeddings.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from config import settings

log = logging.getLogger("faces")

_lock = threading.Lock()
_app = None


def _load():
    global _app
    if _app is not None:
        return _app
    with _lock:
        if _app is not None:
            return _app
        import insightface  # heavy import, do it lazily
        from insightface.app import FaceAnalysis

        model_dir = Path(settings.models_dir) / "insightface"
        model_dir.mkdir(parents=True, exist_ok=True)

        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
            root=str(model_dir),
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _app = app
        log.info("InsightFace buffalo_l loaded")
        return app


def detect_faces(image: np.ndarray) -> list[dict]:
    """Detect faces in a BGR numpy image (H,W,3).

    Returns a list of dicts:
      {bbox: [x1,y1,x2,y2] (pixels), confidence: float, embedding: np.ndarray (512,)}
    """
    app = _load()
    try:
        faces = app.get(image)
    except Exception as exc:  # pragma: no cover
        log.warning("face detection failed: %s", exc)
        return []
    out = []
    for f in faces:
        out.append(
            {
                "bbox": [float(f.bbox[0]), float(f.bbox[1]), float(f.bbox[2]), float(f.bbox[3])],
                "confidence": float(f.det_score),
                "embedding": f.normed_embedding.astype(np.float32),
            }
        )
    return out


def embed_query_face(image: np.ndarray) -> np.ndarray | None:
    """Embed a single query face image for 'who is this?' search.

    Returns the 512-d normalized embedding or None if no face found.
    """
    faces = detect_faces(image)
    if not faces:
        return None
    # Take the largest face as the query.
    largest = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
    return largest["embedding"]
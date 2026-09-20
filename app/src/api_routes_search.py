"""Search route handlers: text search and face similarity search.

Handlers are defined as module-level functions so they can be re-exported
from ``api.py`` (tests call them as ``api.api_search(...)`` etc.).
"""
from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

import api  # noqa: E402  (intentional: see api_common docstring)
from auth import CurrentUser, require_user

log = logging.getLogger("api")

router = APIRouter()


@router.get("/api/search")
def api_search(q: str, limit: int = 60, offset: int = 0,
                user: CurrentUser = Depends(require_user)):
    """Semantic text search over photo captions."""
    q = (q or "").strip()
    if not q:
        raise HTTPException(400, "q required")
    try:
        vec = api.embed_text(q)
    except Exception as exc:
        log.warning("clip text embed failed: %s", exc)
        raise HTTPException(503, "CLIP model unavailable")
    return api._semantic_search(vec, limit, user.id)


@router.post("/api/search/face")
async def api_search_face(file: UploadFile = File(...),
                           limit: int = 50,
                           user: CurrentUser = Depends(require_user)):
    """Find photos containing a face similar to the uploaded image."""
    raw = await file.read()
    if len(raw) > api.FACE_SEARCH_MAX_UPLOAD_BYTES:
        raise HTTPException(413, "upload too large")
    try:
        import numpy as np
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = api.FACE_SEARCH_MAX_IMAGE_PIXELS
        with Image.open(io.BytesIO(raw)) as img:
            w, h = img.size
            if w * h > api.FACE_SEARCH_MAX_IMAGE_PIXELS:
                raise HTTPException(413, "image too large")
            img.load()
            arr = np.asarray(img.convert("RGB"))
            bgr = arr[:, :, ::-1].copy()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"could not read image: {exc}")
    emb = api.embed_query_face(bgr)
    if emb is None:
        raise HTTPException(404, "no face found in image")
    return api._face_similarity(emb, limit, user.id)

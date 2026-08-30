"""FastAPI application: search API + static web UI."""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bridge_client import get_bridge
from clip import embed_text
from config import settings
from faces import embed_query_face
from store import (
    all_clips,
    all_face_rows,
    all_people,
    count_faces_for_person,
    done_photos,
    get_photo,
    photos_for_person,
    rename_person,
    search_photos_by_place,
    stats,
)
import indexer

log = logging.getLogger("api")

app = FastAPI(title="proton-faces", version="0.1.0")

_STATIC = Path(__file__).parent / "static"


# --- helpers ---------------------------------------------------------------

def _row_to_dict(row) -> dict:
    d = dict(row)
    d.pop("embedding", None)
    if d.get("thumb_path"):
        d["thumb_url"] = f"/api/photos/{d['uid']}/thumb"
    else:
        d["thumb_url"] = None
    return d


# --- status ----------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    try:
        b = get_bridge().health()
        bridge_ok = bool(b.get("ok"))
        bridge_logged_in = bool(b.get("loggedIn"))
    except Exception as exc:
        bridge_ok = False
        bridge_logged_in = False
        log.warning("bridge health failed: %s", exc)
    return {
        "ok": True,
        "bridge": {"reachable": bridge_ok, "loggedIn": bridge_logged_in},
        "stats": stats(),
    }


@app.get("/api/stats")
def api_stats() -> dict:
    return stats()


# --- photos ----------------------------------------------------------------

@app.get("/api/photos")
def api_photos(limit: int = 200, offset: int = 0, place: str | None = None):
    if place:
        rows = search_photos_by_place(place, limit=limit)
    else:
        rows = done_photos(limit=limit, offset=offset)
    return {"photos": [_row_to_dict(r) for r in rows]}


@app.get("/api/photos/{uid}")
def api_photo(uid: str):
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    return _row_to_dict(row)


@app.get("/api/photos/{uid}/thumb")
def api_thumb(uid: str):
    row = get_photo(uid)
    if row is None or not row["thumb_path"]:
        raise HTTPException(404, "no thumbnail")
    p = settings.thumb_dir / row["thumb_path"]
    if not p.exists():
        raise HTTPException(404, "thumbnail file missing")
    return FileResponse(p, media_type="image/webp")


@app.get("/api/photos/{uid}/full")
def api_full(uid: str):
    """Stream the full-resolution photo from Proton (on demand, read-only)."""
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    try:
        resp = get_bridge().full_photo(uid)
    except Exception as exc:
        log.warning("full photo fetch failed for %s: %s", uid, exc)
        raise HTTPException(502, "bridge fetch failed")
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, "bridge error")
    content_type = resp.headers.get("content-type", "application/octet-stream")
    headers = {"Cache-Control": "no-store"}
    clen = resp.headers.get("content-length")
    if clen:
        headers["Content-Length"] = clen

    def gen():
        try:
            for chunk in resp.iter_bytes(1 << 16):
                yield chunk
        finally:
            resp.close()

    return StreamingResponse(
        gen(),
        media_type=content_type,
        headers=headers,
    )


# --- people ----------------------------------------------------------------

@app.get("/api/people")
def api_people():
    people = []
    for row in all_people():
        people.append(
            {
                "id": row["id"],
                "name": row["name"],
                "cover_uid": row["cover_uid"],
                "face_count": row["face_count"],
                "photo_count": row["photo_count"],
                "cover_url": (
                    f"/api/photos/{row['cover_uid']}/thumb" if row["cover_uid"] else None
                ),
            }
        )
    return {"people": people}


@app.post("/api/people/{person_id}/name")
def api_people_rename(person_id: int, body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    rename_person(person_id, name)
    return {"ok": True}


@app.get("/api/people/{person_id}/photos")
def api_person_photos(person_id: int, limit: int = 1000):
    rows = photos_for_person(person_id, limit=limit)
    return {"photos": [_row_to_dict(r) for r in rows], "count": count_faces_for_person(person_id)}


# --- search ----------------------------------------------------------------

@app.get("/api/search")
def api_search(q: str, limit: int = 100):
    """Free-text semantic search via CLIP (objects, scenes, etc.)."""
    q = q.strip()
    if not q:
        raise HTTPException(400, "q required")
    try:
        vec = embed_text(q)
    except Exception as exc:
        log.warning("clip text embed failed: %s", exc)
        raise HTTPException(503, "CLIP model unavailable")
    return _semantic_search(vec, limit)


@app.post("/api/search/face")
async def api_face_search(file: UploadFile = File(...), limit: int = 50):
    """Upload a face photo, find matching people/photos."""
    data = await file.read()
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        arr = np.asarray(img.convert("RGB"))
        bgr = arr[:, :, ::-1].copy()
    except Exception as exc:
        raise HTTPException(400, f"could not read image: {exc}")
    emb = embed_query_face(bgr)
    if emb is None:
        raise HTTPException(404, "no face found in image")
    return _face_similarity(emb, limit)


def _semantic_search(vec: np.ndarray, limit: int) -> dict:
    rows = all_clips()
    if not rows:
        return {"results": [], "total": 0}
    uids = [r["photo_uid"] for r in rows]
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = X @ vec  # all embeddings are L2-normalized
    idx = np.argsort(-sims)[:limit]
    results = []
    for i in idx:
        photo = get_photo(uids[i])
        if photo is None:
            continue
        d = _row_to_dict(photo)
        d["score"] = float(sims[i])
        results.append(d)
    return {"results": results, "total": len(results)}


def _face_similarity(emb: np.ndarray, limit: int) -> dict:
    rows = all_face_rows()
    if not rows:
        return {"results": [], "total": 0}
    photo_to_best = {}
    sim_to_photo = {}
    for r in rows:
        fe = np.frombuffer(r["embedding"], dtype=np.float32)
        s = float(fe @ emb)
        uid = r["photo_uid"]
        if s > sim_to_photo.get(uid, -1.0):
            sim_to_photo[uid] = s
            photo_to_best[uid] = s
    ranked = sorted(sim_to_photo.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    results = []
    for uid, score in ranked:
        photo = get_photo(uid)
        if photo is None:
            continue
        d = _row_to_dict(photo)
        d["score"] = score
        results.append(d)
    return {"results": results, "total": len(results)}


# --- static UI -------------------------------------------------------------

app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
"""FastAPI application: search API + static web UI."""
from __future__ import annotations

import io
import json
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
    assign_face_person,
    count_faces_for_person,
    create_person,
    done_photos,
    face_embedding,
    faces_for_photo,
    find_person_by_name,
    get_photo,
    get_person,
    map_markers,
    merge_person,
    person_mean_embedding,
    photos_for_person,
    place_stats,
    rename_person,
    search_photos_by_place,
    set_person_cover_face,
    similar_faces,
    stats,
    unassign_face,
    unassigned_faces,
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


@app.get("/api/places")
def api_places(limit: int = 500):
    rows = place_stats(limit=limit)
    places = []
    for r in rows:
        city = r["place"].split(",")[0].strip()
        places.append({"place": r["place"], "city": city, "count": r["photo_count"]})
    return {"places": places}


@app.get("/api/map")
def api_map(limit: int = 1000):
    rows = map_markers(limit=limit)
    markers = []
    for r in rows:
        city = r["place"].split(",")[0].strip()
        markers.append(
            {
                "place": r["place"],
                "city": city,
                "count": r["photo_count"],
                "lat": r["lat"],
                "lng": r["lng"],
                "thumb_url": f"/api/photos/{r['cover_uid']}/thumb" if r["cover_uid"] else None,
            }
        )
    return {"markers": markers}


@app.get("/api/photos/{uid}")
def api_photo(uid: str):
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    return _row_to_dict(row)


@app.get("/api/photos/{uid}/meta")
def api_photo_meta(uid: str):
    """Full metadata for the photo detail view.

    Merges the local index row (GPS, place, faces, people) with the live
    metadata Proton exposes for the node (size, creation/modification times,
    photo tags, live-photo relations, album names) fetched on demand from the
    bridge.
    """
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    meta = _row_to_dict(row)

    # Faces + people in this photo (local index).
    faces = faces_for_photo(uid)
    people = {}
    for f in faces:
        pid = f["person_id"]
        if pid is None:
            continue
        people.setdefault(pid, f["person_name"])
    meta["face_count"] = len(faces)
    meta["people"] = [{"person_id": k, "name": v} for k, v in people.items()]

    # Live metadata from Proton (on demand; tolerate bridge failures).
    try:
        nodes = get_bridge().nodes([uid])
        if nodes:
            n = nodes[0]
            for k in ("size", "creationTime", "modificationTime", "tags", "mainPhotoNodeUid", "relatedPhotoNodeUids", "mediaType"):
                if n.get(k) is not None:
                    meta[k] = n[k]
    except Exception as exc:
        log.warning("bridge node metadata failed for %s: %s", uid, exc)

    # Album names.
    try:
        alb = get_bridge().albums()
        name_by_uid = {a["uid"]: a["name"] for a in alb.get("albums", [])}
        albums_raw = meta.get("albums")
        if isinstance(albums_raw, str):
            try:
                album_uids = json.loads(albums_raw)
            except Exception:
                album_uids = []
        else:
            album_uids = albums_raw or []
        meta["albums_detail"] = [
            {"uid": u, "name": name_by_uid.get(u, u)} for u in album_uids
        ]
    except Exception as exc:
        log.warning("bridge albums fetch failed: %s", exc)
        meta["albums_detail"] = [{"uid": u, "name": u} for u in (meta.get("albums") or [])]

    return meta


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
                "cover_face_id": row["cover_face_id"],
                "face_count": row["face_count"],
                "photo_count": row["photo_count"],
                "cover_url": (
                    f"/api/people/{row['id']}/cover" if row["cover_face_id"] else None
                ),
            }
        )
    return {"people": people}


@app.get("/api/people/{person_id}/cover")
def api_person_cover(person_id: int):
    person = get_person(person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    face_id = person["cover_face_id"]
    if face_id is None:
        photo = get_photo(person["cover_uid"]) if person["cover_uid"] else None
        if photo is None or not photo["thumb_path"]:
            raise HTTPException(404, "no cover available")
        p = settings.thumb_dir / photo["thumb_path"]
        if not p.exists():
            raise HTTPException(404, "thumbnail file missing")
        return FileResponse(p, media_type="image/webp")
    crop = _face_crop_bytes(face_id)
    if crop is None:
        raise HTTPException(404, "cover face crop unavailable")
    return Response(content=crop, media_type="image/jpeg")


def _face_crop_bytes(face_id: int) -> bytes | None:
    """Crop a face from its photo's cached thumbnail using the normalized bbox."""
    import json

    from PIL import Image

    row = _face_row(face_id)
    if row is None:
        return None
    thumb = settings.thumb_dir / row["thumb_path"]
    if not thumb.exists():
        return None
    bbox = json.loads(row["bbox"])
    x, y, w, h = bbox
    try:
        img = Image.open(thumb).convert("RGB")
        iw, ih = img.size
        # bbox is normalized to the thumbnail dimensions
        left = int(x * iw)
        top = int(y * ih)
        right = int((x + w) * iw)
        bottom = int((y + h) * ih)
        # pad slightly for context
        pad = 0.25
        pw = int((right - left) * pad)
        ph = int((bottom - top) * pad)
        left = max(0, left - pw)
        top = max(0, top - ph)
        right = min(iw, right + pw)
        bottom = min(ih, bottom + ph)
        crop = img.crop((left, top, right, bottom))
        out = io.BytesIO()
        crop.save(out, format="JPEG", quality=90)
        return out.getvalue()
    except Exception as exc:
        log.warning("face crop failed for face %s: %s", face_id, exc)
        return None


def _face_row(face_id: int):
    import sqlite3

    from config import settings as _s

    conn = sqlite3.connect(_s.db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT f.id, f.photo_uid, f.person_id, f.bbox,
                      ph.thumb_path
               FROM faces f JOIN photos ph ON ph.uid = f.photo_uid
               WHERE f.id=?""",
            (face_id,),
        ).fetchone()
        return row
    finally:
        conn.close()


@app.get("/api/faces/unassigned")
def api_unassigned_faces(limit: int = 500):
    rows = unassigned_faces(limit=limit)
    faces = []
    for r in rows:
        import json

        bbox = json.loads(r["bbox"])
        faces.append(
            {
                "id": r["id"],
                "photo_uid": r["photo_uid"],
                "bbox": bbox,
                "confidence": r["confidence"],
                "thumb_url": f"/api/photos/{r['photo_uid']}/thumb",
                "crop_url": f"/api/faces/{r['id']}/crop",
            }
        )
    return {"faces": faces}


@app.get("/api/faces/{face_id}/crop")
def api_face_crop(face_id: int):
    crop = _face_crop_bytes(face_id)
    if crop is None:
        raise HTTPException(404, "face crop unavailable")
    return Response(content=crop, media_type="image/jpeg")


@app.get("/api/photos/{uid}/faces")
def api_photo_faces(uid: str):
    rows = faces_for_photo(uid)
    faces = []
    for r in rows:
        import json

        faces.append(
            {
                "id": r["id"],
                "person_id": r["person_id"],
                "person_name": r["person_name"],
                "bbox": json.loads(r["bbox"]),
                "confidence": r["confidence"],
            }
        )
    return {"faces": faces}


def _merge_propagate(person_id: int, threshold: float | None = None) -> int:
    """Auto-tag unassigned faces similar to a person's mean embedding.

    Returns how many faces were assigned. Used after merges / renames so a
    merged person also pulls in unassigned look-alikes.
    """
    emb = person_mean_embedding(person_id)
    if emb is None:
        return 0
    thr = threshold if threshold is not None else settings.face_sim_threshold
    assigned = 0
    for sim_row in similar_faces(emb.tobytes(), thr, limit=500):
        if sim_row[2] is None:  # person_id
            assign_face_person(sim_row[0], person_id)
            assigned += 1
    return assigned


@app.post("/api/faces/{face_id}/person")
def api_face_assign(face_id: int, body: dict):
    """Assign a face to an existing person (person_id) or create a new named person (name).
    When creating by name, merge into an existing person with the same name.
    Propagates the assignment to similar unassigned faces."""
    person_id = body.get("person_id")
    name = (body.get("name") or "").strip()
    if person_id is None and not name:
        raise HTTPException(400, "provide person_id or name")
    merged = False
    if person_id is not None:
        person = get_person(person_id)
        if person is None:
            raise HTTPException(404, "person not found")
    else:
        row = _face_row(face_id)
        cover_uid = row["photo_uid"] if row else None
        existing = find_person_by_name(name, exclude_id=None)
        if existing is not None:
            person_id = existing["id"]
            merged = True
            set_person_cover_face(person_id, face_id)
        else:
            person_id = create_person(name=name, cover_uid=cover_uid, cover_face_id=face_id)

    assign_face_person(face_id, person_id)
    set_person_cover_face(person_id, face_id)

    # similarity propagation: tag unassigned look-alikes
    emb = face_embedding(face_id)
    assigned = 0
    if emb is not None:
        for sim_row in similar_faces(emb, settings.face_sim_threshold, limit=500):
            if sim_row[2] is None:  # person_id
                assign_face_person(sim_row[0], person_id)
                assigned += 1
    return {
        "ok": True,
        "person_id": person_id,
        "merged": merged,
        "assigned_similar": assigned,
    }


@app.post("/api/faces/{face_id}/unassign")
def api_face_unassign(face_id: int):
    unassign_face(face_id)
    return {"ok": True}


@app.post("/api/people/{person_id}/name")
def api_people_rename(person_id: int, body: dict):
    """Rename a person. If another person already has that name, merge instead."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    existing = find_person_by_name(name, exclude_id=person_id)
    if existing is not None:
        merge_person(person_id, existing["id"])
        _merge_propagate(existing["id"])
        return {"ok": True, "merged": True, "target_id": existing["id"]}
    rename_person(person_id, name)
    return {"ok": True, "merged": False}


@app.post("/api/people/{source_id}/merge")
def api_people_merge(source_id: int, body: dict):
    """Explicitly merge source person into target (by id)."""
    target_id = body.get("target_id")
    if not isinstance(target_id, int):
        raise HTTPException(400, "target_id required")
    if source_id == target_id:
        raise HTTPException(400, "cannot merge a person into itself")
    target = get_person(target_id)
    if target is None:
        raise HTTPException(404, "target person not found")
    merge_person(source_id, target_id)
    assigned = _merge_propagate(target_id)
    return {"ok": True, "target_id": target_id, "assigned_similar": assigned}


@app.get("/api/people/duplicates")
def api_people_duplicates(threshold: float = 0.40, limit: int = 50):
    """Find people whose mean face embeddings are highly similar (likely dupes).

    Vectorized with a single (N x D) @ (D x N) matrix multiply so it stays
    fast even with thousands of people.
    """
    people = all_people()
    n = len(people)
    if n < 2:
        return {"duplicates": []}
    mats = []
    idx = []  # positions in `people` that have an embedding
    for i, p in enumerate(people):
        emb = person_mean_embedding(p["id"])
        if emb is not None:
            mats.append(emb)
            idx.append(i)
    if len(mats) < 2:
        return {"duplicates": []}
    X = np.stack(mats).astype(np.float32)          # (M, 512)
    S = (X @ X.T).astype(np.float32)               # (M, M) cosine sims
    iu = np.triu_indices(S.shape[0], k=1)
    sims = S[iu]
    mask = sims >= threshold
    if not mask.any():
        return {"duplicates": []}
    hits = np.argsort(-sims[mask])[:limit]
    dups = []
    for k in hits:
        i = idx[iu[0][mask][k]]
        j = idx[iu[1][mask][k]]
        a, b = people[i], people[j]
        dups.append(
            {
                "similarity": round(float(sims[mask][k]), 4),
                "a": {"id": a["id"], "name": a["name"], "photo_count": a["photo_count"], "face_count": a["face_count"]},
                "b": {"id": b["id"], "name": b["name"], "photo_count": b["photo_count"], "face_count": b["face_count"]},
            }
        )
    return {"duplicates": dups}


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
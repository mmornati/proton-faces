"""FastAPI application: search API + static web UI."""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import Body
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bridge_client import get_bridge
from clip import embed_text
from config import settings
from faces import embed_query_face
from indexer import get_indexer_state
from store import (
    all_albums,
    album_photos,
    all_clips,
    all_face_rows,
    all_people,
    all_tags,
    archived_photos,
    assign_face_person,
    clip_count,
    count_faces_for_person,
    count_people,
    create_person,
    done_photos,
    face_embedding,
    faces_for_photo,
    find_person_by_name,
    get_photo,
    get_person,
    get_tags,
    map_markers,
    merge_person,
    person_mean_embedding,
    person_mean_embeddings,
    person_map_markers,
    photo_anchors,
    photos_by_tag,
    duplicate_groups,
    memories_for_today,
    photos_for_person,
    place_stats,
    rename_person,
    search_photos_by_place,
    set_archived,
    set_favorited,
    set_hidden,
    set_person_cover_face,
    set_tags,
    similar_faces,
    stats,
    unassign_face,
    unassigned_faces,
)
import indexer

log = logging.getLogger("api")

app = FastAPI(title="proton-faces", version="0.1.0")

_STATIC = Path(__file__).parent / "static"

# TTL cache for the (expensive) duplicates computation.
_DUP_CACHE_TTL = 30.0
_dups_cache: tuple[float, dict] | None = None

# TTL caches (cheap, frequently re-requested on navigation).
_ANCHORS_CACHE_TTL = 60.0
_anchors_cache: tuple[float, dict] | None = None

_PEOPLE_CACHE_TTL = 5.0
_people_cache: tuple[float, dict] | None = None  # keyed by (q, limit, offset)

# In-memory CLIP matrix cache: avoids rebuilding an 88 MB numpy stack on
# every text-search request. Rebuilds only when the clip row count changes
# or after TTL.
_CLIP_CACHE_TTL = 60.0

# Cache for stats() and disk-size walks so the periodic /api/status poll
# doesn't re-pay for 4× COUNT(*) + GROUP BY + a 44 k stat() walk every 15 s.
_STATS_CACHE_TTL = 5.0
_stats_cache: tuple[float, dict] | None = None
_DIRSIZE_CACHE_TTL = 30.0
_dirsize_cache: dict[str, tuple[float, int]] = {}
_clip_cache: tuple[float, int, list[str], np.ndarray] | None = None

# Disk + lock for face crops (computed lazily, then served as plain files).
_crop_lock = threading.Lock()

_IMMUTABLE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}


def _invalidate_dups_cache() -> None:
    global _dups_cache
    _dups_cache = None


def _invalidate_people_cache() -> None:
    global _people_cache
    _people_cache = None


def _invalidate_clip_cache() -> None:
    global _clip_cache
    _clip_cache = None


def _crop_cache_path(face_id: int) -> Path:
    return settings.crops_dir / f"{face_id}.jpg"


def _drop_crop_cache(face_id: int) -> None:
    """Remove a single cached face crop (e.g. after the face was assigned)."""
    with _crop_lock:
        try:
            _crop_cache_path(face_id).unlink(missing_ok=True)
        except OSError:
            pass


def _drop_person_crops(person_id: int) -> None:
    """Remove cached face crops for every face belonging to a person."""
    import sqlite3

    from config import settings as _s

    conn = sqlite3.connect(_s.db_path)
    try:
        face_ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM faces WHERE person_id=?", (person_id,)
            ).fetchall()
        ]
    finally:
        conn.close()
    if not face_ids:
        return
    with _crop_lock:
        for fid in face_ids:
            try:
                _crop_cache_path(fid).unlink(missing_ok=True)
            except OSError:
                pass


# --- helpers ---------------------------------------------------------------

def _row_to_dict(row) -> dict:
    d = dict(row)
    d.pop("embedding", None)
    if d.get("thumb_path"):
        d["thumb_url"] = f"/api/photos/{d['uid']}/thumb"
    else:
        d["thumb_url"] = None
    mt = d.get("media_type") or ""
    if mt.startswith("video/"):
        d["kind"] = "video"
    elif mt.startswith("image/"):
        d["kind"] = "image"
    else:
        d["kind"] = "other"
    # Surface local-only metadata flags + tags as plain JSON-friendly values.
    d["favorited"] = bool(d.get("favorited"))
    d["archived"] = bool(d.get("archived"))
    d["hidden"] = bool(d.get("hidden"))
    raw_tags = d.get("tags")
    if raw_tags:
        try:
            d["tags"] = list(json.loads(raw_tags))
        except Exception:
            d["tags"] = []
    else:
        d["tags"] = []
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
    }


@app.get("/api/stats")
def api_stats() -> dict:
    return stats()


def _dir_size_bytes(path: Path) -> int:
    """Cheap directory size in bytes (sum of immediate children). Best-effort."""
    if not path.exists():
        return 0
    total = 0
    try:
        for entry in path.iterdir():
            try:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += _dir_size_bytes(entry)
            except OSError:
                continue
    except OSError:
        return total
    return total


def _cached_dir_size(path: Path) -> int:
    """Disk-walk with a 30 s TTL — the thumb dir has tens of thousands of
    files and a full `stat()` walk is expensive when polled every 15 s."""
    key = str(path)
    now = time.time()
    hit = _dirsize_cache.get(key)
    if hit is not None and now - hit[0] < _DIRSIZE_CACHE_TTL:
        return hit[1]
    n = _dir_size_bytes(path)
    _dirsize_cache[key] = (now, n)
    return n


def _cached_stats() -> dict:
    """Cached `stats()` so the periodic `/api/status` poll doesn't pay for
    4 COUNT(*) + 1 GROUP BY on every request. 5 s TTL is well under the
    user-perceived staleness of the bottom status bar."""
    global _stats_cache
    now = time.time()
    if _stats_cache is not None and now - _stats_cache[0] < _STATS_CACHE_TTL:
        return _stats_cache[1]
    payload = stats()
    _stats_cache = (now, payload)
    return payload


@app.get("/api/status")
def api_status() -> dict:
    """Aggregated status snapshot for the bottom status bar / details overlay.

    Combines bridge health, indexer stats, runtime state (thread liveness,
    last-sync timestamps, pending queue), and data-dir disk usage. Designed
    to be cheap to poll every few seconds.
    """
    try:
        b = get_bridge().health()
        bridge_ok = bool(b.get("ok"))
        bridge_logged_in = bool(b.get("loggedIn"))
    except Exception as exc:
        bridge_ok = False
        bridge_logged_in = False
        log.warning("bridge health failed: %s", exc)
    s = _cached_stats()
    rt = get_indexer_state()
    thumbs_bytes = _cached_dir_size(settings.thumb_dir)
    db_bytes = settings.db_path.stat().st_size if settings.db_path.exists() else 0
    return {
        "now": time.time(),
        "bridge": {"reachable": bridge_ok, "loggedIn": bridge_logged_in},
        "stats": s,
        "indexer": rt,
        "disk": {
            "thumb_dir_bytes": thumbs_bytes,
            "db_bytes": db_bytes,
        },
        "config": {
            "sync_interval": settings.sync_interval,
            "cluster_interval": settings.cluster_interval,
            "gps_interval": settings.gps_interval,
            "workers": settings.workers,
            "face_sim_threshold": settings.face_sim_threshold,
            "min_cluster_size": settings.min_cluster_size,
            "photos_dir": settings.photos_dir or None,
        },
    }


# --- photos ----------------------------------------------------------------

@app.get("/api/photos")
def api_photos(limit: int = 200, offset: int = 0, place: str | None = None,
               before: int | None = None, only_favorites: bool = False,
               include_archived: bool = True, tag: str | None = None):
    if tag:
        rows = photos_by_tag(tag, limit=limit, offset=offset)
    elif place:
        rows = search_photos_by_place(place, limit=limit, offset=offset)
    else:
        rows = done_photos(limit=limit, offset=offset, before=before,
                            only_favorites=only_favorites, include_archived=include_archived)
    return {"photos": [_row_to_dict(r) for r in rows]}


@app.get("/api/photos/archived")
def api_archived_photos(limit: int = 200, offset: int = 0):
    return {"photos": [_row_to_dict(r) for r in archived_photos(limit=limit, offset=offset)]}


@app.get("/api/memories")
def api_memories(month: int | None = None, day: int | None = None, limit: int = 60):
    """Photos captured on (month, day) in previous years — "on this day".

    Defaults to today's calendar date in UTC so the UI can just call the
    endpoint without arguments. Each result includes ``age_days`` so the UI
    can render "5 years ago today".
    """
    import datetime as _dt
    now = _dt.datetime.utcnow()
    m = month if month is not None else now.month
    d = day if day is not None else now.day
    rows = memories_for_today(m, d, limit=limit)
    out = []
    for r in rows:
        photo = _row_to_dict(r)
        age_days = int(r["age_days"]) if r["age_days"] is not None else None
        photo["age_days"] = age_days
        photo["age_years"] = int(age_days // 365) if age_days is not None else None
        out.append(photo)
    return {"month": m, "day": d, "photos": out}


@app.get("/api/duplicates")
def api_duplicates(limit: int = 200):
    """Groups of photos that share a Proton content-hash (sha1).

    Each group is rendered side-by-side in the Duplicates tab; users can
    hide individual copies (``hidden=1``) so they don't re-appear.
    """
    groups = duplicate_groups(limit=limit)
    out = []
    for members in groups:
        out.append({
            "sha1": members[0]["sha1"],
            "count": len(members),
            "photos": [_row_to_dict(r) for r in members],
        })
    return {"groups": out}


@app.get("/api/tags")
def api_tags():
    return {"tags": [{"name": r[0], "count": r[1]} for r in all_tags()]}


@app.patch("/api/photos/{uid}")
def api_patch_photo(uid: str, body: dict = Body(...)):
    """Set local-only metadata flags on a photo: favorited, archived, hidden.

    Body keys are all optional; only the provided ones are updated. Returns
    the updated photo row.
    """
    if get_photo(uid) is None:
        raise HTTPException(404, "photo not found")
    if "favorited" in body:
        set_favorited(uid, bool(body["favorited"]))
    if "archived" in body:
        set_archived(uid, bool(body["archived"]))
    if "hidden" in body:
        set_hidden(uid, bool(body["hidden"]))
    row = get_photo(uid)
    return _row_to_dict(row)


@app.put("/api/photos/{uid}/tags")
def api_set_tags(uid: str, body: dict = Body(...)):
    """Replace the freeform tag set for a photo. ``tags`` is a list of strings."""
    if get_photo(uid) is None:
        raise HTTPException(404, "photo not found")
    tags = body.get("tags") or []
    if not isinstance(tags, list):
        raise HTTPException(400, "tags must be a list of strings")
    clean = set_tags(uid, tags)
    return {"uid": uid, "tags": clean}


@app.get("/api/photos/{uid}/tags")
def api_get_tags(uid: str):
    if get_photo(uid) is None:
        raise HTTPException(404, "photo not found")
    return {"uid": uid, "tags": get_tags(uid)}


@app.get("/api/photos/anchors")
def api_photo_anchors():
    """Year-month anchors for the date rail. Cached for `_ANCHORS_CACHE_TTL`."""
    global _anchors_cache
    now = time.time()
    if _anchors_cache is not None and now - _anchors_cache[0] < _ANCHORS_CACHE_TTL:
        return _anchors_cache[1]

    import datetime as _dt

    anchors = []
    for r in photo_anchors():
        ym = r["ym"]
        try:
            label = _dt.datetime.strptime(ym, "%Y-%m").strftime("%b %Y")
        except Exception:
            label = ym
        anchors.append({"ym": ym, "label": label, "first_ts": r["first_ts"]})
    payload = {"anchors": anchors}
    _anchors_cache = (now, payload)
    return payload


@app.get("/api/albums")
def api_albums():
    albums = []
    for r in all_albums():
        albums.append(
            {
                "uid": r["uid"],
                "name": r["name"] or r["uid"],
                "photo_count": r["photo_count"] or 0,
                "start_ts": r["start_ts"],
                "end_ts": r["end_ts"],
                "cover_url": (
                    f"/api/photos/{r['cover_uid']}/thumb" if r["cover_uid"] else None
                ),
            }
        )
    return {"albums": albums}


@app.get("/api/albums/{album_uid}/photos")
def api_album_photos(album_uid: str, limit: int = 200, offset: int = 0):
    rows = album_photos(album_uid, limit=limit, offset=offset)
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
            for k in ("size", "creationTime", "modificationTime", "mainPhotoNodeUid", "relatedPhotoNodeUids", "mediaType"):
                if n.get(k) is not None:
                    meta[k] = n[k]
            # Proton's read-only photo tags come back under `proton_tags` so
            # they don't collide with the user's local tags column.
            pt = n.get("tags")
            if pt is not None:
                meta["proton_tags"] = pt
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
    return FileResponse(p, media_type="image/webp", headers=_IMMUTABLE_HEADERS)


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
def api_people(limit: int = 200, offset: int = 0, q: str | None = None):
    """People ordered by photo_count DESC. Paginated, optionally filtered by name."""
    q = (q or "").strip() or None
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    global _people_cache
    now = time.time()
    key = (q, limit, offset)
    if (
        _people_cache is not None
        and _people_cache[0] == key
        and now - _people_cache[1] < _PEOPLE_CACHE_TTL
    ):
        return _people_cache[2]

    rows = all_people(q=q, limit=limit, offset=offset)
    total = count_people(q=q)
    people = [
        {
            "id": r["id"],
            "name": r["name"],
            "cover_uid": r["cover_uid"],
            "cover_face_id": r["cover_face_id"],
            "face_count": r["face_count"],
            "photo_count": r["photo_count"],
            "cover_url": (
                f"/api/people/{r['id']}/cover" if r["cover_face_id"] else None
            ),
        }
        for r in rows
    ]
    payload = {"people": people, "total": total, "limit": limit, "offset": offset}
    _people_cache = (key, now, payload)
    return payload


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
        return FileResponse(p, media_type="image/webp", headers=_IMMUTABLE_HEADERS)
    cache_path = _crop_cache_path(face_id)
    if not cache_path.exists():
        crop = _face_crop_bytes(face_id)
        if crop is None:
            raise HTTPException(404, "cover face crop unavailable")
    return FileResponse(cache_path, media_type="image/jpeg", headers=_IMMUTABLE_HEADERS)


def _face_crop_bytes(face_id: int) -> bytes | None:
    """Crop a face from its photo's cached thumbnail using the normalized bbox.

    Caches the result on disk under `crops/{face_id}.jpg` so subsequent
    requests serve a plain file (and `api_person_cover` / `api_face_crop`
    can use `FileResponse` with immutable cache headers).
    """
    import json

    from PIL import Image

    cache_path = _crop_cache_path(face_id)
    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except OSError:
            pass

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
        data = out.getvalue()
    except Exception as exc:
        log.warning("face crop failed for face %s: %s", face_id, exc)
        return None

    with _crop_lock:
        tmp = cache_path.with_suffix(".tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, cache_path)
        except OSError as exc:
            log.warning("face crop cache write failed for %s: %s", face_id, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return data


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
    cache_path = _crop_cache_path(face_id)
    if not cache_path.exists():
        crop = _face_crop_bytes(face_id)
        if crop is None:
            raise HTTPException(404, "face crop unavailable")
    return FileResponse(cache_path, media_type="image/jpeg", headers=_IMMUTABLE_HEADERS)


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
    _drop_crop_cache(face_id)

    # similarity propagation: tag unassigned look-alikes
    emb = face_embedding(face_id)
    assigned = 0
    if emb is not None:
        for sim_row in similar_faces(emb, settings.face_sim_threshold, limit=500):
            if sim_row[2] is None:  # person_id
                assign_face_person(sim_row[0], person_id)
                assigned += 1
    _invalidate_dups_cache()
    _invalidate_people_cache()
    return {
        "ok": True,
        "person_id": person_id,
        "merged": merged,
        "assigned_similar": assigned,
    }


@app.post("/api/faces/{face_id}/unassign")
def api_face_unassign(face_id: int):
    unassign_face(face_id)
    _drop_crop_cache(face_id)
    _invalidate_people_cache()
    _invalidate_dups_cache()
    return {"ok": True}


@app.post("/api/people/{person_id}/name")
def api_people_rename(person_id: int, body: dict):
    """Rename a person. If another person already has that name, merge instead."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    existing = find_person_by_name(name, exclude_id=person_id)
    if existing is not None:
        _drop_person_crops(person_id)
        merge_person(person_id, existing["id"])
        _merge_propagate(existing["id"])
        _invalidate_dups_cache()
        _invalidate_people_cache()
        tgt = get_person(existing["id"])
        return {
            "ok": True,
            "merged": True,
            "target_id": existing["id"],
            "photo_count": tgt["photo_count"] if tgt else None,
            "face_count": tgt["face_count"] if tgt else None,
        }
    rename_person(person_id, name)
    _invalidate_dups_cache()
    _invalidate_people_cache()
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
    _drop_person_crops(source_id)
    merge_person(source_id, target_id)
    assigned = _merge_propagate(target_id)
    _invalidate_dups_cache()
    _invalidate_people_cache()
    tgt = get_person(target_id)
    return {
        "ok": True,
        "target_id": target_id,
        "assigned_similar": assigned,
        "photo_count": tgt["photo_count"] if tgt else None,
        "face_count": tgt["face_count"] if tgt else None,
    }


@app.get("/api/people/duplicates")
def api_people_duplicates(threshold: float = 0.40, limit: int = 50):
    """Find people whose mean face embeddings are highly similar (likely dupes).

    Vectorized with a single (N x D) @ (D x N) matrix multiply and cached for
    a few seconds so repeated reloads are cheap.
    """
    global _dups_cache
    now = time.time()
    if _dups_cache is not None and now - _dups_cache[0] < _DUP_CACHE_TTL:
        return _dups_cache[1]

    people = all_people()
    n = len(people)
    if n < 2:
        return {"duplicates": []}
    means = person_mean_embeddings()  # single query, one entry per person w/ faces
    idx = []
    mats = []
    for i, p in enumerate(people):
        emb = means.get(p["id"])
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
        resp = {"duplicates": []}
        _dups_cache = (now, resp)
        return resp
    hits = np.argsort(-sims[mask])[:limit]
    dups = []
    for k in hits:
        i = idx[iu[0][mask][k]]
        j = idx[iu[1][mask][k]]
        a, b = people[i], people[j]
        dups.append(
            {
                "similarity": round(float(sims[mask][k]), 4),
                "a": {
                    "id": a["id"],
                    "name": a["name"],
                    "photo_count": a["photo_count"],
                    "face_count": a["face_count"],
                    "cover_url": (
                        f"/api/people/{a['id']}/cover" if a["cover_face_id"] else None
                    ),
                },
                "b": {
                    "id": b["id"],
                    "name": b["name"],
                    "photo_count": b["photo_count"],
                    "face_count": b["face_count"],
                    "cover_url": (
                        f"/api/people/{b['id']}/cover" if b["cover_face_id"] else None
                    ),
                },
            }
        )
    resp = {"duplicates": dups}
    _dups_cache = (now, resp)
    return resp


@app.get("/api/people/{person_id}/photos")
def api_person_photos(person_id: int, limit: int = 200, offset: int = 0):
    rows = photos_for_person(person_id, limit=limit, offset=offset)
    return {"photos": [_row_to_dict(r) for r in rows], "count": count_faces_for_person(person_id)}


@app.get("/api/people/{person_id}/map")
def api_person_map(person_id: int, limit: int = 500):
    """Clustered map markers for one person: places they've been photographed in.

    Same shape as ``/api/map`` but filtered to photos carrying faces of this
    person. Drives the per-person "map of where I've seen them" view.
    """
    rows = person_map_markers(person_id, limit=limit)
    markers = []
    for r in rows:
        city = r["place"].split(",")[0].strip()
        markers.append({
            "place": r["place"],
            "city": city,
            "count": r["photo_count"],
            "lat": r["lat"],
            "lng": r["lng"],
            "thumb_url": f"/api/photos/{r['cover_uid']}/thumb" if r["cover_uid"] else None,
        })
    return {"markers": markers}


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


def _get_clip_matrix() -> tuple[list[str], np.ndarray]:
    """Cached (uids, X) matrix of every CLIP embedding. Rebuilt only when the
    clip row count changes or after `_CLIP_CACHE_TTL`."""
    global _clip_cache
    now = time.time()
    n_now = clip_count()
    if _clip_cache is not None:
        ts, n_cached, uids, X = _clip_cache
        if n_cached == n_now and (now - ts) < _CLIP_CACHE_TTL:
            return uids, X
    rows = all_clips()
    if not rows:
        return [], np.empty((0, 512), dtype=np.float32)
    uids = [r["photo_uid"] for r in rows]
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    _clip_cache = (now, n_now, uids, X)
    return uids, X


def _semantic_search(vec: np.ndarray, limit: int) -> dict:
    uids, X = _get_clip_matrix()
    if X.size == 0:
        return {"results": [], "total": 0}
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
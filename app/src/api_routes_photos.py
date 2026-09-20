"""Photo route handlers: list, batch, single, meta, favorites, tags, fullres.

Handlers are defined as module-level functions so they can be re-exported
from ``api.py`` (tests call them as ``api.api_photos(...)`` etc.).
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Body, Depends, HTTPException

import api  # noqa: E402  (intentional: see api_common docstring)
from auth import CurrentUser, require_role, require_user
from store import (
    all_albums,
    all_tags,
    archived_photos,
    count_places,
    done_photos,
    favorite_photo,
    get_photo,
    map_markers,
    photos_by_tag,
    place_stats,
    search_photos_by_place,
    set_archived,
    set_hidden,
    set_tags,
    unfavorite_photo,
)

log = logging.getLogger("api")

router = APIRouter()

# Places/map payloads are small per row (no per-row signing), so they get a
# higher ceiling than grid pages.
MAP_MAX_LIMIT = 5000


@router.get("/api/photos")
def api_photos(limit: int = 200, offset: int = 0, place: str | None = None,
               before: int | None = None, only_favorites: bool = False,
               include_archived: bool = True, tag: str | None = None,
               user: CurrentUser = Depends(require_user)):
    """List photos with optional filters."""
    limit = api._clamp_limit(limit)
    offset = api._clamp_offset(offset)
    if tag:
        rows = photos_by_tag(tag, limit=limit, offset=offset)
    elif place:
        rows = search_photos_by_place(place, limit=limit, offset=offset)
    else:
        rows = done_photos(limit=limit, offset=offset, before=before,
                            only_favorites=only_favorites,
                            include_archived=include_archived,
                            user_id=user.id)
    return {"photos": api._user_photos(user.id, rows)}


@router.get("/api/photos/archived")
def api_archived_photos(limit: int = 200, offset: int = 0,
                         user: CurrentUser = Depends(require_user)):
    """List archived photos."""
    limit = api._clamp_limit(limit)
    offset = api._clamp_offset(offset)
    return {"photos": api._user_photos(user.id, archived_photos(limit=limit, offset=offset))}


@router.get("/api/memories")
def api_memories(month: int | None = None, day: int | None = None, limit: int = 60,
                  user: CurrentUser = Depends(require_user)):
    """Photos captured on (month, day) in previous years — 'on this day'."""
    import datetime as _dt

    from store import memories_for_today
    limit = api._clamp_limit(limit, default=60)
    now = _dt.datetime.utcnow()
    m = month if month is not None else now.month
    d = day if day is not None else now.day
    rows = memories_for_today(m, d, limit=limit)
    photos = api._user_photos(user.id, rows)
    for photo, r in zip(photos, rows):
        age_days = int(r["age_days"]) if r["age_days"] is not None else None
        photo["age_days"] = age_days
        photo["age_years"] = int(age_days // 365) if age_days is not None else None
    return {"month": m, "day": d, "photos": photos}


@router.get("/api/duplicates")
def api_duplicates(limit: int = 200, user: CurrentUser = Depends(require_user)):
    """Groups of photos that share a Proton content-hash (sha1)."""
    limit = api._clamp_limit(limit)
    groups = api._duplicate_groups_cached(limit)
    all_uids = [r["uid"] for members in groups for r in members]
    fav_set = api.favorite_uids(user.id, all_uids) if all_uids else set()
    face_counts = api.face_counts_for_photos(all_uids) if all_uids else {}
    out = []
    for members in groups:
        out.append({
            "sha1": members[0]["sha1"],
            "count": len(members),
            "photos": api._user_photos(user.id, members, fav_set=fav_set, face_count_set=face_counts),
        })
    return {"groups": out}


@router.get("/api/tags")
def api_tags():
    """List all tags with photo counts."""
    return {"tags": [{"name": r[0], "count": r[1]} for r in all_tags()]}


@router.patch("/api/photos/{uid}")
def api_patch_photo(uid: str, body: dict = Body(...),
                     user: CurrentUser = Depends(require_role("write"))):
    """Set local-only metadata flags on a photo."""
    if get_photo(uid) is None:
        raise HTTPException(404, "photo not found")
    if "favorited" in body:
        if bool(body["favorited"]):
            favorite_photo(user.id, uid)
        else:
            unfavorite_photo(user.id, uid)
    if "archived" in body:
        set_archived(uid, bool(body["archived"]))
    if "hidden" in body:
        set_hidden(uid, bool(body["hidden"]))
        api._invalidate_photo_dups_cache()
    row = get_photo(uid)
    return api._single_user_photo(user.id, row)


@router.put("/api/photos/{uid}/tags")
def api_set_tags(uid: str, body: dict = Body(...),
                  user: CurrentUser = Depends(require_role("write"))):
    """Replace the freeform tag set for a photo."""
    if get_photo(uid) is None:
        raise HTTPException(404, "photo not found")
    tags = body.get("tags") or []
    if not isinstance(tags, list):
        raise HTTPException(400, "tags must be a list of strings")
    clean = set_tags(uid, tags)
    return {"uid": uid, "tags": clean}


@router.get("/api/photos/{uid}/tags")
def api_get_tags(uid: str):
    """Get the freeform tag set for a photo."""
    if get_photo(uid) is None:
        raise HTTPException(404, "photo not found")
    from store import get_tags
    return {"uid": uid, "tags": get_tags(uid)}


@router.get("/api/photos/anchors")
def api_photo_anchors():
    """Year-month anchors for the date rail."""
    return api._anchors_payload()


@router.get("/api/albums")
def api_albums(q: str | None = None):
    """List albums."""
    q = (q or "").strip() or None
    albums = []
    for r in all_albums(q=q):
        albums.append(
            {
                "uid": r["uid"],
                "name": r["name"] or r["uid"],
                "photo_count": r["photo_count"] or 0,
                "start_ts": r["start_ts"],
                "end_ts": r["end_ts"],
                "cover_url": api._sign_if_needed(
                    f"/api/photos/{r['cover_uid']}/thumb" if r["cover_uid"] else None
                ),
            }
        )
    total = len(all_albums()) if q else len(albums)
    return {"albums": albums, "total": total}


@router.get("/api/albums/{album_uid}/photos")
def api_album_photos(album_uid: str, limit: int = 200, offset: int = 0,
                      user: CurrentUser = Depends(require_user)):
    """List photos in an album."""
    from store import album_photos
    limit = api._clamp_limit(limit)
    offset = api._clamp_offset(offset)
    rows = album_photos(album_uid, limit=limit, offset=offset)
    return {"photos": api._user_photos(user.id, rows)}


@router.get("/api/places")
def api_places(limit: int = 500, q: str | None = None):
    """List places with photo counts."""
    limit = api._clamp_limit(limit, default=500, max_limit=MAP_MAX_LIMIT)
    q = (q or "").strip() or None
    rows = place_stats(limit=limit, q=q)
    places = []
    for r in rows:
        city = r["place"].split(",")[0].strip()
        places.append({"place": r["place"], "city": city, "count": r["photo_count"]})
    total = count_places() if q else len(places)
    return {"places": places, "total": total}


@router.get("/api/map")
def api_map(limit: int = 1000, q: str | None = None):
    """Map markers for all places."""
    limit = api._clamp_limit(limit, default=1000, max_limit=MAP_MAX_LIMIT)
    rows = map_markers(limit=limit, q=q)
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
                "thumb_url": api._sign_if_needed(
                    f"/api/photos/{r['cover_uid']}/thumb" if r["cover_uid"] else None
                ),
            }
        )
    return {"markers": markers}


@router.get("/api/photos/{uid}")
def api_photo(uid: str, user: CurrentUser = Depends(require_user)):
    """Get a single photo."""
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    return api._single_user_photo(user.id, row)


@router.get("/api/photos/{uid}/meta")
def api_photo_meta(uid: str, user: CurrentUser = Depends(require_user)):
    """Full metadata for the photo detail view."""
    from bridge_client import get_bridge
    from store import album_names, faces_for_photo
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    meta = api._single_user_photo(user.id, row)
    faces = faces_for_photo(uid)
    people = {}
    for f in faces:
        pid = f["person_id"]
        if pid is None:
            continue
        people.setdefault(pid, f["person_name"])
    meta["face_count"] = len(faces)
    meta["people"] = [{"person_id": k, "name": v} for k, v in people.items()]
    try:
        nodes = get_bridge().nodes([uid], timeout_sec=5.0)
        if nodes:
            n = nodes[0]
            keys = (
                "size", "creationTime", "modificationTime", "mainPhotoNodeUid",
                "relatedPhotoNodeUids", "mediaType",
            )
            for k in keys:
                if n.get(k) is not None:
                    meta[k] = n[k]
            pt = n.get("tags")
            if pt is not None:
                meta["proton_tags"] = pt
    except Exception as exc:
        log.warning("bridge node metadata failed for %s: %s", uid, exc)
    albums_raw = meta.get("albums")
    if isinstance(albums_raw, str):
        try:
            album_uids = json.loads(albums_raw)
        except Exception:
            album_uids = []
    else:
        album_uids = albums_raw or []
    name_by_uid = album_names(album_uids)
    meta["albums_detail"] = [
        {"uid": u, "name": name_by_uid.get(u, u)} for u in album_uids
    ]
    return meta

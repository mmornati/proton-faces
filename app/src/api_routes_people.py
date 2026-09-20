"""People + faces route handlers.

Handlers are defined as module-level functions so they can be re-exported
from ``api.py`` (tests call them as ``api.api_people(...)`` etc.).
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException

import api  # noqa: E402  (intentional: see api_common docstring)
from auth import CurrentUser, require_role, require_user
from store import (
    assign_face_person,
    count_faces_for_person,
    create_person,
    face_embedding,
    find_person_by_name,
    get_person,
    live_person_ids,
    merge_people_bulk,
    merge_person,
    people_by_ids,
    person_map_markers,
    person_mean_matrix_from_cache,
    photos_for_person,
    rename_person,
    set_person_cover_face_if_unset,
    similar_faces,
    unassign_face,
    unassigned_faces,
)

log = logging.getLogger("api")

router = APIRouter()

# Hard ceilings for the bulk people-merge and map endpoints (see
# api_common.LIST_MAX_LIMIT for the grid-page ceiling).
MERGE_ALL_MAX_SOURCES = 500
MAP_MAX_LIMIT = 5000


@router.get("/api/people")
def api_people(limit: int = 200, offset: int = 0, q: str | None = None,
               include_small: bool = False):
    """List people clusters (paginated), named people first.

    Anonymous clusters below MIN_CLUSTER_SIZE photos are hidden unless
    ``include_small=1``: the setting only shapes new clusters, so a library
    indexed under an older value carries thousands of 1–2 photo rows that
    bury the people worth naming. Named people always show. ``hidden_small``
    in the response tells the UI how many were filtered out.
    """
    limit = api._clamp_limit(limit)
    offset = api._clamp_offset(offset)
    full = api._people_all_cached(q=q)
    if q:
        ql = q.lower()
        full = [p for p in full if (p.get("name") or "").lower().find(ql) >= 0]
    hidden_small = 0
    if not include_small:
        min_photos = api.settings.min_cluster_size
        kept = [p for p in full if p.get("name") or (p.get("photo_count") or 0) >= min_photos]
        hidden_small = len(full) - len(kept)
        full = kept
    # Stable sort: named people first, each group keeps its photo_count order.
    full = sorted(full, key=lambda p: 0 if p.get("name") else 1)
    total = len(full)
    page = full[offset : offset + limit]
    return {"people": page, "total": total, "limit": limit, "offset": offset,
            "hidden_small": hidden_small}


@router.get("/api/people/{person_id}/faces")
def api_people_faces(person_id: int, limit: int = 200, offset: int = 0,
                      user: CurrentUser = Depends(require_user)):
    """Every face of a person, for the cover picker.

    Returns face ids plus a signed crop URL for each, so the front-end can
    render a grid of candidate face-crops without knowing bbox math.
    """
    from store import faces_for_person
    limit = api._clamp_limit(limit)
    offset = api._clamp_offset(offset)
    person = get_person(person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    # faces_for_person has no offset support at the store layer, so fetch
    # enough rows to cover the page and slice locally (see issue #108 code
    # review — slicing a `limit`-sized fetch always returned empty pages
    # past the first).
    rows = faces_for_person(person_id, limit=limit + offset)
    faces = [
        {
            "id": r["id"],
            "photo_uid": r["photo_uid"],
            "confidence": r["confidence"],
            "crop_url": api._sign_if_needed(f"/api/faces/{r['id']}/crop"),
            "is_cover": r["id"] == person["cover_face_id"],
        }
        for r in rows[offset:offset+limit]
    ]
    return {"faces": faces, "cover_face_id": person["cover_face_id"], "count": len(faces)}


@router.post("/api/people/{person_id}/cover")
def api_set_person_cover(person_id: int, body: dict = Body(...),
                          user: CurrentUser = Depends(require_role("write"))):
    """Set the cover face for a person cluster."""
    face_id = body.get("face_id")
    if not isinstance(face_id, int):
        raise HTTPException(400, "face_id (int) required")
    face = api._face_row(face_id)
    if face is None or face["person_id"] != person_id:
        raise HTTPException(400, "face does not belong to this person")
    from store import set_person_cover_face
    set_person_cover_face(person_id, face_id)
    api._invalidate_people_cache()
    return {"ok": True, "person_id": person_id, "cover_face_id": face_id}


@router.get("/api/faces/unassigned")
def api_unassigned_faces(limit: int = 200, offset: int = 0):
    """List faces that have not been assigned to a person."""
    limit = api._clamp_limit(limit)
    offset = api._clamp_offset(offset)
    # unassigned_faces has no offset support at the store layer; see the
    # comment in api_people_faces above for why we over-fetch instead of
    # slicing a `limit`-sized result.
    rows = unassigned_faces(limit=limit + offset)
    faces = [
        {
            "id": r["id"],
            "photo_uid": r["photo_uid"],
            "confidence": r["confidence"],
            "thumb_url": api._sign_if_needed(f"/api/photos/{r['photo_uid']}/thumb"),
            "crop_url": api._sign_if_needed(f"/api/faces/{r['id']}/crop"),
        }
        for r in rows[offset:offset+limit]
    ]
    return {"faces": faces}


@router.get("/api/faces/{face_id}/suggest")
def api_face_suggest(face_id: int, limit: int = 5):
    """Rank existing people by how likely they are to be this face."""
    import numpy as np
    limit = api._clamp_limit(limit, default=5, max_limit=50)

    from store import face_embedding, live_person_ids, people_by_ids, person_mean_matrix_from_cache
    emb = face_embedding(face_id)
    if emb is None:
        return {"suggestions": []}
    fe = np.frombuffer(emb, dtype=np.float32)
    pids, M = person_mean_matrix_from_cache()
    if M.shape[0] == 0:
        return {"suggestions": []}
    sims = M @ fe
    order = np.argsort(-sims)
    top_n = max(1, min(limit, 50))
    live = set(live_person_ids([int(pids[i]) for i in order[:top_n]]))
    order = order[[int(pids[i]) in live for i in order[:top_n]]]
    by_id = {r["id"]: r for r in people_by_ids([int(pids[i]) for i in order[:top_n]])}
    scored = []
    for i in order[:top_n]:
        pid = int(pids[i])
        p = by_id.get(pid)
        scored.append({
            "person_id": pid,
            "name": (p["name"] if p else None) or f"person {pid}",
            "similarity": float(sims[i]),
        })
    return {"suggestions": scored}


@router.get("/api/photos/{uid}/faces")
def api_photo_faces(uid: str):
    """List faces detected in a photo."""
    from store import faces_for_photo
    rows = faces_for_photo(uid)
    faces = [
        {
            "id": r["id"],
            "person_id": r["person_id"],
            "person_name": r["person_name"],
            "bbox": json.loads(r["bbox"]),
            "confidence": r["confidence"],
        }
        for r in rows
    ]
    return {"faces": faces}


@router.post("/api/faces/{face_id}/person")
def api_assign_face(face_id: int, body: dict = Body(...),
                     user: CurrentUser = Depends(require_role("write"))):
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
        row = api._face_row(face_id)
        cover_uid = row["photo_uid"] if row else None
        existing = find_person_by_name(name, exclude_id=None)
        if existing is not None:
            person_id = existing["id"]
            merged = True
            set_person_cover_face_if_unset(person_id, face_id)
        else:
            person_id = create_person(name=name, cover_uid=cover_uid, cover_face_id=face_id)

    assign_face_person(face_id, person_id)
    set_person_cover_face_if_unset(person_id, face_id)
    api._drop_crop_cache(face_id)

    # similarity propagation: tag unassigned look-alikes
    emb = face_embedding(face_id)
    assigned = 0
    if emb is not None:
        for sim_row in similar_faces(emb, api.settings.face_sim_threshold, limit=500):
            if sim_row[2] is None:  # person_id
                assign_face_person(sim_row[0], person_id)
                assigned += 1
    api._invalidate_dups_cache()
    api._invalidate_people_cache()
    return {
        "ok": True,
        "person_id": person_id,
        "merged": merged,
        "assigned_similar": assigned,
    }


@router.post("/api/faces/{face_id}/unassign")
def api_unassign_face(face_id: int,
                       user: CurrentUser = Depends(require_role("write"))):
    """Remove a face from its person cluster."""
    unassign_face(face_id)
    api._drop_crop_cache(face_id)
    api._invalidate_people_cache()
    api._invalidate_dups_cache()
    return {"ok": True}


@router.post("/api/people/{person_id}/name")
def api_rename_person(person_id: int, body: dict = Body(...),
                       user: CurrentUser = Depends(require_role("write"))):
    """Rename a person. If another person already has that name, merge instead."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    existing = find_person_by_name(name, exclude_id=person_id)
    if existing is not None:
        api._drop_person_crops(person_id)
        merge_person(person_id, existing["id"])
        api._merge_propagate(existing["id"])
        api._invalidate_dups_cache()
        api._invalidate_people_cache()
        tgt = get_person(existing["id"])
        return {
            "ok": True,
            "merged": True,
            "target_id": existing["id"],
            "photo_count": tgt["photo_count"] if tgt else None,
            "face_count": tgt["face_count"] if tgt else None,
        }
    rename_person(person_id, name)
    api._invalidate_dups_cache()
    api._invalidate_people_cache()
    return {"ok": True, "merged": False}


@router.post("/api/people/{source_id}/merge")
def api_merge_people(source_id: int, body: dict = Body(...),
                      user: CurrentUser = Depends(require_role("write"))):
    """Explicitly merge source person into target (by id)."""
    target_id = body.get("target_id")
    if not isinstance(target_id, int):
        raise HTTPException(400, "target_id required")
    if source_id == target_id:
        raise HTTPException(400, "cannot merge a person into itself")
    target = get_person(target_id)
    if target is None:
        raise HTTPException(404, "target person not found")
    api._drop_person_crops(source_id)
    merge_person(source_id, target_id)
    assigned = api._merge_propagate(target_id)
    api._invalidate_dups_cache()
    api._invalidate_people_cache()
    tgt = get_person(target_id)
    return {
        "ok": True,
        "target_id": target_id,
        "assigned_similar": assigned,
        "photo_count": tgt["photo_count"] if tgt else None,
        "face_count": tgt["face_count"] if tgt else None,
    }


@router.get("/api/people/{person_id}/similar")
def api_people_similar(person_id: int, threshold: float = 0.40, limit: int = 50, offset: int = 0):
    """People whose mean face embedding is similar to `person_id`'s (cosine)."""
    threshold = api._clamp_threshold(threshold)
    limit = api._clamp_limit(limit, default=50)
    offset = api._clamp_offset(offset)
    pids, M = person_mean_matrix_from_cache()
    tgt = np.flatnonzero(pids == person_id)
    if tgt.size == 0 or pids.size < 2:
        return {"similar": [], "total": 0}
    fe = M[tgt[0]]
    sims = M @ fe
    hits = np.flatnonzero((sims >= threshold) & (pids != person_id))
    if hits.size == 0:
        return {"similar": [], "total": 0}
    order = hits[np.argsort(-sims[hits])]
    live = set(live_person_ids([int(pids[i]) for i in order]))
    order = order[[int(pids[i]) in live for i in order]]
    total = int(order.size)
    if total == 0:
        return {"similar": [], "total": 0}
    order = order[offset : offset + limit]
    top_pids = [int(pids[i]) for i in order]
    top_sims = [float(sims[i]) for i in order]
    by_id = {r["id"]: r for r in people_by_ids(top_pids)}
    similar = []
    for pid, sim in zip(top_pids, top_sims):
        p = by_id.get(pid)
        similar.append(
            {
                "person_id": pid,
                "name": (p["name"] if p else None) or f"person {pid}",
                "similarity": round(sim, 4),
                "photo_count": p["photo_count"] if p else 0,
                "face_count": p["face_count"] if p else 0,
                "cover_url": api._sign_if_needed(
                    f"/api/people/{pid}/cover" if p and p["cover_face_id"] else None
                ),
            }
        )
    return {"similar": similar, "total": total}


@router.post("/api/people/{target_id}/merge_all")
def api_merge_all(target_id: int, body: dict = Body(default={}),
                   user: CurrentUser = Depends(require_role("write"))):
    """Merge many people into `target_id` in one call."""
    source_ids = body.get("source_ids")
    if not isinstance(source_ids, list) or not source_ids:
        raise HTTPException(400, "source_ids list required")
    target = get_person(target_id)
    if target is None:
        raise HTTPException(404, "target person not found")
    seen: set[int] = set()
    ids: list[int] = []
    for sid in source_ids:
        if not isinstance(sid, int) or sid == target_id or sid in seen:
            continue
        seen.add(sid)
        ids.append(sid)
    live = live_person_ids(ids)
    if live:
        api._drop_people_crops(live)
        merged_count = merge_people_bulk(live, target_id)
    else:
        merged_count = 0
    assigned = api._merge_propagate(target_id) if merged_count else 0
    api._invalidate_dups_cache()
    api._invalidate_people_cache()
    tgt = get_person(target_id)
    return {
        "ok": True,
        "target_id": target_id,
        "merged_count": merged_count,
        "assigned_similar": assigned,
        "photo_count": tgt["photo_count"] if tgt else None,
        "face_count": tgt["face_count"] if tgt else None,
    }


@router.post("/api/people/{target_id}/merge_all_similar")
def api_merge_all_similar(target_id: int, body: dict = Body(default={}),
                           user: CurrentUser = Depends(require_role("write"))):
    """Merge every person whose mean embedding is similar to the target's.

    Bounded on purpose: ``threshold`` is clamped to ``[THRESHOLD_MIN, 1.0]``
    and ``max_sources`` to ``MERGE_ALL_MAX_SOURCES`` so a single request can
    never fold the whole people table into one cluster (there is no undo).
    ``dry_run: true`` returns the candidate count without merging so the UI
    can confirm first.
    """
    threshold = api._clamp_threshold(body.get("threshold", 0.40))
    try:
        max_sources = int(body.get("max_sources", MERGE_ALL_MAX_SOURCES))
    except (TypeError, ValueError):
        max_sources = MERGE_ALL_MAX_SOURCES
    max_sources = max(1, min(max_sources, MERGE_ALL_MAX_SOURCES))
    dry_run = bool(body.get("dry_run", False))
    target = get_person(target_id)
    if target is None:
        raise HTTPException(404, "target person not found")
    pids, M = person_mean_matrix_from_cache()
    tgt_idx = np.flatnonzero(pids == target_id)
    if tgt_idx.size == 0 or pids.size < 2:
        return {"ok": True, "target_id": target_id, "merged_count": 0, "assigned_similar": 0,
                "photo_count": target["photo_count"], "face_count": target["face_count"]}
    fe = M[tgt_idx[0]]
    sims = M @ fe
    hits = np.flatnonzero((sims >= threshold) & (pids != target_id))
    order = hits[np.argsort(-sims[hits])]
    live = set(live_person_ids([int(pids[i]) for i in order]))
    order = order[[int(pids[i]) in live for i in order]]
    source_ids = [int(pids[i]) for i in order[:max_sources]]
    if dry_run:
        return {"ok": True, "target_id": target_id, "dry_run": True,
                "candidate_count": len(source_ids), "threshold": threshold,
                "max_sources": max_sources, "truncated": len(order) > max_sources}
    if not source_ids:
        return {"ok": True, "target_id": target_id, "merged_count": 0, "assigned_similar": 0,
                "photo_count": target["photo_count"], "face_count": target["face_count"]}
    api._drop_people_crops(source_ids)
    merged_count = merge_people_bulk(source_ids, target_id)
    assigned = api._merge_propagate(target_id) if merged_count else 0
    api._invalidate_dups_cache()
    api._invalidate_people_cache()
    tgt = get_person(target_id)
    return {
        "ok": True,
        "target_id": target_id,
        "merged_count": merged_count,
        "assigned_similar": assigned,
        "photo_count": tgt["photo_count"] if tgt else None,
        "face_count": tgt["face_count"] if tgt else None,
    }


@router.get("/api/people/duplicates")
def api_people_duplicates(threshold: float = 0.40, limit: int = 50):
    """Find people whose mean face embeddings are highly similar (likely dupes)."""
    threshold = api._clamp_threshold(threshold)
    limit = api._clamp_limit(limit, default=50)
    now = time.time()
    if api._dups_cache is not None and now - api._dups_cache[0] < api._DUP_CACHE_TTL:
        return api._dups_cache[1]
    with api._dups_cache_lock:
        now = time.time()
        if api._dups_cache is not None and now - api._dups_cache[0] < api._DUP_CACHE_TTL:
            return api._dups_cache[1]
        resp = api._dups_payload(threshold, limit)
        api._dups_cache = (now, resp)
        return resp


@router.get("/api/people/suggested-merges")
def api_people_suggested_merges(threshold: float = 0.40, limit: int = 50, offset: int = 0):
    """Person-centric suggested merges: who has look-alikes? (named first)"""
    threshold = api._clamp_threshold(threshold)
    limit = api._clamp_limit(limit, default=50)
    offset = api._clamp_offset(offset)
    with api._suggested_cache_lock:
        now = time.time()
        entry = api._suggested_cache.get(threshold)
        if entry is not None and now - entry[0] < api._SUGGESTED_CACHE_TTL:
            full = entry[1]
        else:
            full = api._suggested_rows(threshold)
            api._suggested_cache[threshold] = (time.time(), full)
    page = full[offset : offset + limit]
    named = sum(1 for r in full if r["name"])
    return {
        "people": page,
        "total": len(full),
        "named": named,
        "limit": limit,
        "offset": offset,
    }


@router.get("/api/people/{person_id}/photos")
def api_person_photos(person_id: int, limit: int = 200, offset: int = 0,
                       user: CurrentUser = Depends(require_user)):
    """List photos that contain this person."""
    limit = api._clamp_limit(limit)
    offset = api._clamp_offset(offset)
    rows = photos_for_person(person_id, limit=limit, offset=offset)
    return {"photos": api._user_photos(user.id, rows), "count": count_faces_for_person(person_id)}


@router.get("/api/people/{person_id}/map")
def api_person_map(person_id: int, limit: int = 500):
    """Clustered map markers for one person: places they've been photographed in."""
    limit = api._clamp_limit(limit, default=500, max_limit=MAP_MAX_LIMIT)
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
            "thumb_url": api._sign_if_needed(
                f"/api/photos/{r['cover_uid']}/thumb" if r["cover_uid"] else None
            ),
        })
    return {"markers": markers}

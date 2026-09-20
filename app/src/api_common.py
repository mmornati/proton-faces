"""Cross-router pure helpers used by multiple route modules.

Helpers here resolve monkeypatchable symbols via ``api.<name>`` at call
time so tests can ``monkeypatch.setattr(api, "favorite_uids", ...)`` and
have the patched value flow through. The ``import api`` at the bottom of
this module is intentional: it makes the lookup go through the
aggregator's attribute namespace, which is what tests patch.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from fastapi import HTTPException, Request, Response

import api  # noqa: E402  (intentional: see module docstring)
import store
from auth import allow_public_thumbs, make_signed_token
from config import settings
from store import (
    all_clips,
    count_faces_for_photo,
    is_favorite,
)

log = logging.getLogger("api")

# --- Refresh cookie name (FP-1) --------------------------------------------

_REFRESH_COOKIE = "pf_refresh"


def _set_refresh_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(
        _REFRESH_COOKIE,
        token,
        max_age=api.auth_refresh_ttl(),
        httponly=True,
        samesite="strict",
        secure=settings.auth_cookie_secure,
        path="/",
    )


def _clear_refresh_cookie(resp: Response) -> None:
    resp.delete_cookie(_REFRESH_COOKIE, path="/", samesite="strict")


# --- Bearer extraction (used by /api/status to gate the config block) -----

def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(None, 1)[1].strip()
    return token or None


def _bearer_is_valid(request: Request) -> bool:
    """True only when the request carries a live *access* token (F-03).

    Presence of an ``Authorization`` header is not enough: the config block
    on ``/api/status`` must stay hidden from anyone who merely sends
    ``Bearer x``. Resolves the token against ``auth_tokens`` exactly like
    ``require_user`` does, minus the 401.
    """
    token = _extract_bearer(request)
    if not token:
        return False
    row = store.lookup_token(token)
    if row is None or row["disabled"] or row["kind"] != "access":
        return False
    return row["expires_at"] >= time.time()


# --- Request-parameter clamps ---------------------------------------------
# Every list endpoint accepts a client-controlled `limit`/`offset`; the
# store helpers behind them build `IN (...)` lists or sign one URL per row,
# so an unbounded limit is a memory/CPU amplifier for any authenticated
# caller. LIST_MAX_LIMIT is the hard ceiling for grid-style pages; callers
# that legitimately need more (map markers) pass their own `max_limit`.

LIST_MAX_LIMIT = 500
THRESHOLD_MIN = 0.2


def _clamp_limit(limit: int, default: int = 200, max_limit: int = LIST_MAX_LIMIT) -> int:
    if limit is None or limit < 1:
        return min(default, max_limit)
    return min(limit, max_limit)


def _clamp_offset(offset: int) -> int:
    return max(0, offset or 0)


def _clamp_threshold(value, default: float = 0.40) -> float:
    """Cosine-similarity threshold in [THRESHOLD_MIN, 1.0].

    A threshold at or below zero matches every pair: the people-merge and
    suggested-merge paths would then collapse the whole people table (or
    walk an O(P^2) Python loop) from one request. Non-numeric input falls
    back to `default` rather than surfacing a 500.
    """
    try:
        t = float(value)
    except (TypeError, ValueError):
        t = default
    if t != t:  # NaN
        t = default
    return max(THRESHOLD_MIN, min(1.0, t))


# --- Signed URL helper -----------------------------------------------------

def _sign_if_needed(url: str | None, ttl_seconds: int = 300) -> str | None:
    if not url or allow_public_thumbs():
        return url
    sig, exp = make_signed_token(url, ttl_seconds=ttl_seconds)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sig={sig}&exp={exp}"


# --- Row serialization -----------------------------------------------------

def _row_to_dict(row) -> dict:
    d = dict(row)
    d.pop("embedding", None)
    if d.get("thumb_path"):
        d["thumb_url"] = _sign_if_needed(f"/api/photos/{d['uid']}/thumb")
    else:
        d["thumb_url"] = None
    mt = d.get("media_type") or ""
    if mt.startswith("video/"):
        d["kind"] = "video"
    elif mt.startswith("image/"):
        d["kind"] = "image"
    else:
        d["kind"] = "other"
    d["favorited"] = bool(d.get("favorited"))
    d["favorited_by_me"] = False
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


def _user_photos(
    user_id: int, rows, fav_set: set[str] | None = None, face_count_set: dict[str, int] | None = None
) -> list[dict]:
    if fav_set is None:
        uids = [r["uid"] for r in rows]
        fav_set = api.favorite_uids(user_id, uids) if uids else set()
    if face_count_set is None:
        uids = [r["uid"] for r in rows]
        face_count_set = api.face_counts_for_photos(uids) if uids else {}
    out = []
    for r in rows:
        d = _row_to_dict(r)
        d["favorited_by_me"] = r["uid"] in fav_set
        d["face_count"] = face_count_set.get(r["uid"], 0)
        out.append(d)
    return out


def _single_user_photo(user_id: int, row) -> dict:
    d = _row_to_dict(row)
    d["favorited_by_me"] = is_favorite(user_id, row["uid"])
    d["face_count"] = count_faces_for_photo(row["uid"])
    return d


# --- Crop cache helpers ----------------------------------------------------

def _crop_cache_path(face_id: int) -> Path:
    return settings.crops_dir / f"{face_id}.jpg"


def _drop_crop_cache(face_id: int) -> None:
    with api._crop_lock:
        try:
            _crop_cache_path(face_id).unlink(missing_ok=True)
        except OSError:
            pass


def _drop_person_crops(person_id: int) -> None:
    from store import get_conn

    with get_conn() as conn:
        face_ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM faces WHERE person_id=?", (person_id,)
            ).fetchall()
        ]
    if not face_ids:
        return
    with api._crop_lock:
        for fid in face_ids:
            try:
                _crop_cache_path(fid).unlink(missing_ok=True)
            except OSError:
                pass


def _drop_people_crops(person_ids: list[int]) -> None:
    from store import face_ids_for_people

    face_ids = face_ids_for_people(person_ids)
    if not face_ids:
        return
    with api._crop_lock:
        for fid in face_ids:
            try:
                _crop_cache_path(fid).unlink(missing_ok=True)
            except OSError:
                pass


# --- Image type sniffing (used by /full when bridge returns octet-stream) -

def _sniff_image_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        if data[8:12] in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
            return "image/heic"
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return None


# --- People LRU cache helpers ---------------------------------------------

def _people_cache_get_locked(q: str | None, now: float) -> list | None:
    entry = api._people_cache.get(q)
    if entry is None:
        return None
    ts, full = entry
    if now - ts >= api._PEOPLE_CACHE_TTL:
        return None
    api._people_cache.move_to_end(q)
    return full


def _people_cache_put_locked(q: str | None, now: float, full: list) -> None:
    api._people_cache[q] = (now, full)
    api._people_cache.move_to_end(q)
    while len(api._people_cache) > api._PEOPLE_CACHE_MAX:
        api._people_cache.popitem(last=False)


def _people_all_cached(q: str | None = None) -> list:
    now = time.time()
    with api._people_cache_lock:
        full = _people_cache_get_locked(q, now)
        if full is not None:
            return full
        rows = api.all_people(q=q)
        full = [
            {
                "id": r["id"],
                "name": r["name"],
                "cover_uid": r["cover_uid"],
                "cover_face_id": r["cover_face_id"],
                "face_count": r["face_count"],
                "photo_count": r["photo_count"],
                "cover_url": _sign_if_needed(
                    f"/api/people/{r['id']}/cover" if r["cover_face_id"] else None
                ),
            }
            for r in rows
        ]
        _people_cache_put_locked(q, now, full)
        return full


# --- Photo duplicates cache ------------------------------------------------

def _duplicate_groups_cached(limit: int) -> list[list]:
    now = time.time()
    cache = api._photo_dups_cache
    if cache is not None and now - cache[0] < api._PHOTO_DUPS_CACHE_TTL and limit in cache[1]:
        return cache[1][limit]
    with api._photo_dups_cache_lock:
        now = time.time()
        cache = api._photo_dups_cache
        if cache is not None and now - cache[0] < api._PHOTO_DUPS_CACHE_TTL and limit in cache[1]:
            return cache[1][limit]
        if cache is None or now - cache[0] >= api._PHOTO_DUPS_CACHE_TTL:
            cache = (now, {})
            api._photo_dups_cache = cache
        groups = api.duplicate_groups(limit=limit)
        cache[1][limit] = groups
        return groups


# --- Stats + bridge health caches -----------------------------------------

def _cached_stats() -> dict:
    now = time.time()
    if api._stats_cache is not None and now - api._stats_cache[0] < api._STATS_CACHE_TTL:
        return api._stats_cache[1]
    with api._stats_cache_lock:
        now = time.time()
        if api._stats_cache is not None and now - api._stats_cache[0] < api._STATS_CACHE_TTL:
            return api._stats_cache[1]
        payload = api.stats()
        api._stats_cache = (now, payload)
        return payload


def _cached_bridge_health() -> tuple[bool, bool]:
    now = time.time()
    if api._bridge_health_cache is not None and now - api._bridge_health_cache[0] < api._BRIDGE_HEALTH_CACHE_TTL:
        return api._bridge_health_cache[1]
    with api._bridge_health_cache_lock:
        now = time.time()
        if api._bridge_health_cache is not None and now - api._bridge_health_cache[0] < api._BRIDGE_HEALTH_CACHE_TTL:
            return api._bridge_health_cache[1]
        try:
            b = api.get_bridge().health()
            state = (bool(b.get("ok")), bool(b.get("loggedIn")))
        except Exception as exc:
            log.warning("bridge health failed: %s", exc)
            state = (False, False)
        api._bridge_health_cache = (now, state)
        return state


# --- Duplicates payload (blockwise pairwise scan) --------------------------

def _dups_payload(threshold: float, limit: int) -> dict:
    people = api._people_all_cached()
    if len(people) < 2:
        return {"duplicates": []}
    means = api.person_mean_embeddings_from_cache()
    mats = []
    ids = []
    by_id = {}
    for p in people:
        emb = means.get(p["id"])
        if emb is not None:
            mats.append(emb)
            ids.append(p["id"])
            by_id[p["id"]] = p
    if len(mats) < 2:
        return {"duplicates": []}
    X = np.stack(mats).astype(np.float32)
    M = X.shape[0]
    import heapq

    block = 1024
    heap: list[tuple[float, int, int]] = []
    for s in range(0, M, block):
        e = min(s + block, M)
        Sb = X[s:e] @ X.T
        rows, cols = np.nonzero(Sb >= threshold)
        keep = (s + rows) < cols
        rows, cols = rows[keep], cols[keep]
        g_rows = s + rows
        vals = Sb[rows, cols].tolist()
        for v, gi, gj in zip(vals, g_rows.tolist(), cols.tolist()):
            if len(heap) < limit:
                heapq.heappush(heap, (v, gi, gj))
            elif v > heap[0][0]:
                heapq.heapreplace(heap, (v, gi, gj))
    hits = sorted(heap, reverse=True)

    dups = []
    for sim, i, j in hits:
        a, b = by_id[ids[i]], by_id[ids[j]]
        dups.append(
            {
                "similarity": round(float(sim), 4),
                "a": {
                    "id": a["id"],
                    "name": a["name"],
                    "photo_count": a["photo_count"],
                    "face_count": a["face_count"],
                    "cover_url": a["cover_url"],
                },
                "b": {
                    "id": b["id"],
                    "name": b["name"],
                    "photo_count": b["photo_count"],
                    "face_count": b["face_count"],
                    "cover_url": b["cover_url"],
                },
            }
        )
    return {"duplicates": dups}


def _suggested_rows(threshold: float) -> list[dict]:
    people = _people_all_cached()
    if len(people) < 2:
        return []
    means = api.person_mean_embeddings_from_cache()
    mats: list[np.ndarray] = []
    ids: list[int] = []
    by_id: dict[int, dict] = {}
    for p in people:
        emb = means.get(p["id"])
        if emb is not None:
            mats.append(emb)
            ids.append(p["id"])
            by_id[p["id"]] = p
    if len(mats) < 2:
        return []
    X = np.stack(mats).astype(np.float32)
    M = X.shape[0]
    counts = np.zeros(M, dtype=np.int32)
    tops: list[list[float]] = [[] for _ in range(M)]

    block = 1024
    for s in range(0, M, block):
        e = min(s + block, M)
        Sb = X[s:e] @ X.T
        rows, cols = np.nonzero(Sb >= threshold)
        keep = (s + rows) < cols
        rows, cols = rows[keep], cols[keep]
        g_rows = s + rows
        vals = Sb[rows, cols].tolist()
        for v, gi, gj in zip(vals, g_rows.tolist(), cols.tolist()):
            counts[gi] += 1
            counts[gj] += 1
            for idx in (gi, gj):
                t = tops[idx]
                if len(t) < 3:
                    t.append(v)
                    t.sort(reverse=True)
                elif v > t[-1]:
                    t[-1] = v
                    t.sort(reverse=True)

    rows = []
    for gi, c in enumerate(counts):
        if not c:
            continue
        p = by_id[ids[gi]]
        rows.append(
            {
                "person_id": ids[gi],
                "name": p["name"],
                "cover_url": p["cover_url"],
                "photo_count": p["photo_count"],
                "face_count": p["face_count"],
                "candidate_count": int(c),
                "top_scores": [round(float(x), 4) for x in tops[gi]],
            }
        )
    rows.sort(
        key=lambda r: (
            0 if r["name"] else 1,
            -r["candidate_count"],
            -(r["top_scores"][0] if r["top_scores"] else 0.0),
            r["person_id"],
        )
    )
    return rows


# --- Merge propagation (used after merges / renames) -----------------------

def _merge_propagate(person_id: int, threshold: float | None = None) -> int:
    from store import assign_face_person, person_mean_embedding, similar_faces

    emb = person_mean_embedding(person_id)
    if emb is None:
        return 0
    thr = threshold if threshold is not None else settings.face_sim_threshold
    assigned = 0
    for sim_row in similar_faces(emb.tobytes(), thr, limit=500):
        if sim_row[2] is None:
            assign_face_person(sim_row[0], person_id)
            assigned += 1
    return assigned


# --- Search helpers --------------------------------------------------------

SEARCH_MAX_LIMIT = 200


def _clamp_search_limit(limit: int) -> int:
    return max(1, min(limit, SEARCH_MAX_LIMIT))


_FACE_DEDUPE_SLACK = 4


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[0]
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.intp)
    k = min(k, n)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def _get_clip_matrix() -> tuple[list[str], np.ndarray]:
    now = time.time()
    if api._clip_cache is not None:
        ts, _, uids, X = api._clip_cache
        if (now - ts) < api._CLIP_CACHE_TTL:
            return uids, X
    with api._clip_cache_lock:
        now = time.time()
        if api._clip_cache is not None:
            ts, _, uids, X = api._clip_cache
            if (now - ts) < api._CLIP_CACHE_TTL:
                return uids, X
        n_now = api.clip_count()
        if api._clip_cache is not None:
            ts, n_cached, uids, X = api._clip_cache
            if n_cached == n_now:
                api._clip_cache = (now, n_cached, uids, X)
                return uids, X
        sidecar = api.read_clip_sidecar()
        if sidecar is not None:
            uids, X = sidecar
            api._clip_cache = (now, len(uids), uids, X)
            return uids, X
        rows = all_clips()
        if not rows:
            return [], np.empty((0, 512), dtype=np.float32)
        uids = [r["photo_uid"] for r in rows]
        X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        api._clip_cache = (now, n_now, uids, X)
        return uids, X


def _semantic_search(vec: np.ndarray, limit: int, user_id: int) -> dict:
    limit = _clamp_search_limit(limit)
    uids, X = _get_clip_matrix()
    if X.size == 0:
        return {"results": [], "total": 0}
    sims = X @ vec
    idx = _topk_indices(sims, limit)
    photo_uids = [uids[i] for i in idx]
    photos = api.get_photos_batch(photo_uids)
    fav_set = api.favorite_uids(user_id, photo_uids)
    face_counts = api.face_counts_for_photos(photo_uids)
    results = []
    for i in idx:
        uid = uids[i]
        photo = photos.get(uid)
        if photo is None:
            continue
        d = _row_to_dict(photo)
        d["favorited_by_me"] = uid in fav_set
        d["face_count"] = face_counts.get(uid, 0)
        d["score"] = float(sims[i])
        results.append(d)
    return {"results": results, "total": len(results)}


def _face_similarity(emb: np.ndarray, limit: int, user_id: int) -> dict:
    limit = _clamp_search_limit(limit)
    data = api._embedding_cache_data()
    mat = data["mat"]
    if mat.shape[0] == 0:
        return {"results": [], "total": 0}
    uids = data["photo_uids"]
    scores = mat @ emb
    n = scores.shape[0]
    order = _topk_indices(scores, min(limit * _FACE_DEDUPE_SLACK, n))
    seen: set = set()
    top_uids: list = []
    top_scores: list = []
    for i in order:
        uid = uids[i]
        if uid in seen:
            continue
        seen.add(uid)
        top_uids.append(uid)
        top_scores.append(float(scores[i]))
        if len(top_uids) >= limit:
            break
    if len(order) < n and len(top_uids) < limit:
        rest = np.setdiff1d(np.arange(n), order)
        rest = rest[np.argsort(-scores[rest])]
        for i in rest:
            if len(top_uids) >= limit:
                break
            uid = uids[i]
            if uid in seen:
                continue
            seen.add(uid)
            top_uids.append(uid)
            top_scores.append(float(scores[i]))
    photos = api.get_photos_batch(top_uids)
    fav_set = api.favorite_uids(user_id, top_uids)
    face_counts = api.face_counts_for_photos(top_uids)
    results = []
    for uid, score in zip(top_uids, top_scores):
        photo = photos.get(uid)
        if photo is None:
            continue
        d = _row_to_dict(photo)
        d["favorited_by_me"] = uid in fav_set
        d["face_count"] = face_counts.get(uid, 0)
        d["score"] = score
        results.append(d)
    return {"results": results, "total": len(results)}


# --- Admin user-row serializer --------------------------------------------

def _user_row_public(row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
        "disabled": bool(row["disabled"]),
        "totp_enabled": bool(row["totp_enabled"]),
    }


# --- Face crop helpers (used by /cover and /crop) -------------------------

def _face_row(face_id: int):
    from store import get_conn

    with get_conn() as conn:
        row = conn.execute(
            """SELECT f.id, f.photo_uid, f.person_id, f.bbox
               FROM faces f JOIN photos ph ON ph.uid = f.photo_uid
               WHERE f.id=?""",
            (face_id,),
        ).fetchone()
        return row


def _face_crop_bytes(face_id: int) -> bytes | None:
    """Crop a face from its photo's cached thumbnail using the normalized bbox.

    Caches the result on disk under `crops/{face_id}.jpg` so subsequent
    requests serve a plain file (and `api_person_cover` / `api_face_crop`
    can use `FileResponse` with immutable cache headers).
    """
    from PIL import Image

    from bridge_client import is_valid_uid, uid_invalid_reason

    cache_path = _crop_cache_path(face_id)
    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except OSError:
            pass

    row = _face_row(face_id)
    if row is None:
        return None
    if not is_valid_uid(row["photo_uid"]):
        log.warning(
            "discarding face crop: invalid photo uid %r (%s)",
            row["photo_uid"], uid_invalid_reason(row["photo_uid"]),
        )
        return None
    thumb = settings.thumb_dir / f"{row['photo_uid']}.webp"
    if not thumb.exists():
        return None
    bbox = json.loads(row["bbox"])
    x, y, w, h = bbox
    try:
        img = Image.open(thumb).convert("RGB")
        iw, ih = img.size
        left = int(x * iw)
        top = int(y * ih)
        right = int((x + w) * iw)
        bottom = int((y + h) * ih)
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

    with api._crop_lock:
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


# --- Binary endpoint helpers (used by /thumb /full /cover /crop) ----------

def _serve_thumb(uid: str, row) -> Response:
    from fastapi.responses import FileResponse

    from bridge_client import is_valid_uid, uid_invalid_reason

    if not is_valid_uid(uid):
        log.warning(
            "discarding media request: invalid uid %r (%s)",
            uid, uid_invalid_reason(uid),
        )
        raise HTTPException(404, "no thumbnail")
    p = settings.thumb_dir / f"{uid}.webp"
    if not p.exists():
        raise HTTPException(404, "thumbnail file missing")
    return FileResponse(p, media_type="image/webp", headers=api._IMMUTABLE_HEADERS)


async def _serve_full(uid: str, row, request: Request) -> Response:
    from fastapi.responses import StreamingResponse

    from bridge_client import BridgeTransientError, get_bridge

    range_header = request.headers.get("range")
    async with api._full_semaphore:
        try:
            resp = await asyncio.wait_for(
                get_bridge().full_photo_async(
                    uid, range_header=range_header,
                    timeout_ms=int(api._FULL_TIMEOUT_SEC * 1000),
                ),
                timeout=api._FULL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            log.warning(
                "full photo timed out after %.1fs for %s; returning 504",
                api._FULL_TIMEOUT_SEC, uid,
            )
            api._record_full_res_failure()
            raise HTTPException(504, "full photo fetch timed out — try again later")
        except BridgeTransientError as exc:
            log.warning("full photo bridge transient for %s: %s", uid, exc)
            api._record_full_res_failure()
            detail = exc.args[0] if exc.args else "bridge transient error"
            raise HTTPException(
                status_code=exc.status_code,
                detail=detail,
                headers={"Retry-After": str(int(exc.retry_after_sec or 1))},
            )
        except Exception as exc:
            log.warning("full photo fetch failed for %s: %s", uid, exc)
            api._record_full_res_failure()
            raise HTTPException(502, "bridge fetch failed")

    if resp.status_code not in (200, 206):
        log.warning("full photo bridge error for %s: status %s", uid, resp.status_code)
        api._record_full_res_failure()
        await resp.aclose()
        raise HTTPException(resp.status_code, "bridge error")

    content_type = resp.headers.get("content-type", "application/octet-stream")
    headers = {"Cache-Control": "no-store"}
    for h in ("content-length", "accept-ranges", "content-range"):
        v = resp.headers.get(h)
        if v:
            headers[h] = v

    first_chunk = b""
    if content_type == "application/octet-stream":
        log.warning("full photo %s returned octet-stream; sniffing magic bytes", uid)
        try:
            first_chunk = await resp.aread(1 << 16)
        except Exception:
            pass
        sniffed = _sniff_image_type(first_chunk)
        headers["Content-Type"] = sniffed or "image/jpeg"

    async def gen():
        try:
            if first_chunk:
                yield first_chunk
            async for chunk in resp.aiter_bytes(1 << 16):
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        headers=headers,
    )


def _serve_person_cover(person_id: int, person) -> Response:
    from fastapi.responses import FileResponse

    from bridge_client import is_valid_uid, uid_invalid_reason

    face_id = person["cover_face_id"]
    if face_id is None:
        if not person["cover_uid"] or not is_valid_uid(person["cover_uid"]):
            log.warning(
                "discarding cover request: invalid uid %r (%s)",
                person["cover_uid"], uid_invalid_reason(person["cover_uid"]),
            )
            raise HTTPException(404, "no cover available")
        p = settings.thumb_dir / f"{person['cover_uid']}.webp"
        if not p.exists():
            raise HTTPException(404, "thumbnail file missing")
        return FileResponse(p, media_type="image/webp", headers=api._IMMUTABLE_HEADERS)
    cache_path = _crop_cache_path(face_id)
    if not cache_path.exists():
        crop = _face_crop_bytes(face_id)
        if crop is None:
            raise HTTPException(404, "cover face crop unavailable")
    return FileResponse(cache_path, media_type="image/jpeg", headers=api._IMMUTABLE_HEADERS)


def _serve_face_crop(face_id: int, face) -> Response:
    from fastapi.responses import FileResponse

    cache_path = _crop_cache_path(face_id)
    if not cache_path.exists():
        crop = _face_crop_bytes(face_id)
        if crop is None:
            raise HTTPException(404, "face crop unavailable")
    return FileResponse(cache_path, media_type="image/jpeg", headers=api._IMMUTABLE_HEADERS)


# --- Indexer proxy helpers (used by /api/status + admin sync endpoints) ---

def _indexer_is_local() -> bool:
    """True when this process is running the indexer threads (RUN_INDEXER=1)."""
    import indexer
    return bool(getattr(indexer, "_runtime", {}).get("threads"))


def _empty_indexer_state(pending_db: int | None = None) -> dict:
    out = {
        "started_at": None,
        "last_sync": None,
        "last_sync_error": None,
        "last_cluster": None,
        "last_gps": None,
        "pending_in_queue": 0,
        "threads": {},
        "remote": True,
    }
    if pending_db is not None:
        out["pending_db"] = int(pending_db)
    return out


def _fetch_remote_indexer_state() -> dict:
    now = time.time()
    if api._indexer_proxy_cache is not None and now - api._indexer_proxy_cache[0] < api._INDEXER_PROXY_CACHE_TTL:
        return api._indexer_proxy_cache[1]
    url = settings.indexer_status_url.rstrip("/") + "/status"
    headers = {}
    token = (settings.indexer_token or "").strip()
    if token:
        headers["X-Indexer-Token"] = token
    try:
        resp = api._get_indexer_proxy_client().get(url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        if now - api._indexer_proxy_last_warn >= api._INDEXER_PROXY_LOG_THROTTLE:
            log.warning("indexer status proxy failed (%s): %s", url, exc)
            api._indexer_proxy_last_warn = now
        try:
            pending_db = (_cached_stats().get("photos") or {}).get("pending", 0)
        except Exception:
            pending_db = None
        payload = _empty_indexer_state(pending_db=pending_db)
        payload["proxy_ok"] = False
        payload["proxy_error"] = type(exc).__name__
    payload.setdefault("pending_db", 0)
    payload.setdefault("pending_in_queue", 0)
    payload.setdefault("proxy_ok", True)
    api._indexer_proxy_cache = (now, payload)
    return payload


def _indexer_proxy_json(method: str, path: str, body: dict | None = None) -> dict:
    url = settings.indexer_status_url.rstrip("/") + path
    headers = {}
    token = (settings.indexer_token or "").strip()
    if token:
        headers["X-Indexer-Token"] = token
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        resp = api._get_indexer_proxy_client().request(method, url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise HTTPException(502, f"indexer proxy {method} {path} failed: {exc}")


def _merged_indexer_state() -> dict:
    from indexer import get_indexer_state

    if _indexer_is_local():
        return get_indexer_state()
    return _fetch_remote_indexer_state()


# --- Crop prewarm worker (started by main.py) -----------------------------

def start_crop_prewarm_worker() -> None:
    """Background thread that pre-generates every people cover crop once, so
    the People page serves plain files instead of ~30 ms PIL encodes on a
    ~97% cache-miss grid. Runs in the parent process only (with uvicorn
    workers>1 the parent is the only place these daemon threads live).
    Resumable: skips crops that already exist, and rescans periodically so
    newly created people eventually get covered too."""
    import concurrent.futures

    def _loop() -> None:
        import time as _time

        from store import get_conn

        log = logging.getLogger("crop-prewarm")
        while True:
            try:
                with get_conn() as conn:
                    rows = conn.execute(
                        "SELECT cover_face_id FROM people "
                        "WHERE cover_face_id IS NOT NULL ORDER BY id"
                    ).fetchall()
                missing = [
                    r[0] for r in rows if not _crop_cache_path(r[0]).exists()
                ]
                if not missing:
                    _time.sleep(300.0)
                    continue
                log.info("crop prewarm: %d cover crops missing, generating", len(missing))
                done = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futs = [pool.submit(_face_crop_bytes, fid) for fid in missing]
                    for _ in concurrent.futures.as_completed(futs):
                        done += 1
                        if done % 500 == 0:
                            log.info(
                                "crop prewarm: %d/%d done", done, len(missing)
                            )
                log.info("crop prewarm: finished %d crops", done)
                _time.sleep(300.0)
            except Exception:
                log.exception("crop prewarm iteration failed")
                _time.sleep(300.0)

    t = threading.Thread(target=_loop, name="crop-prewarm", daemon=True)
    t.start()
    log.info("crop prewarm worker started")


# --- Anchors cache (used by /api/photos/anchors) --------------------------

def _anchors_payload() -> dict:
    """Year-month anchors for the date rail. Cached for `_ANCHORS_CACHE_TTL`."""
    import datetime as _dt

    from store import photo_anchors

    now = time.time()
    if api._anchors_cache is not None and now - api._anchors_cache[0] < api._ANCHORS_CACHE_TTL:
        return api._anchors_cache[1]

    anchors = []
    for r in photo_anchors():
        ym = r["ym"]
        try:
            label = _dt.datetime.strptime(ym, "%Y-%m").strftime("%b %Y")
        except Exception:
            label = ym
        anchors.append({"ym": ym, "label": label, "first_ts": r["first_ts"]})
    payload = {"anchors": anchors}
    api._anchors_cache = (now, payload)
    return payload

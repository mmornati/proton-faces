"""The indexing pipeline.

Runs in the background of the FastAPI process:
  1. sync loop   — fetch timeline from the bridge, diff against SQLite,
                   discovering new and deleted photos.
  2. downloader  — for each new photo, ask the bridge to fetch its 512px
                   WebP thumbnail into DATA_DIR/work/.
  3. workers     — for each downloaded photo: run face detection + CLIP,
                   write results to SQLite, move thumbnail into the cache,
                   then delete the work file (nothing is kept on disk beyond
                   the small thumbnail).
  4. cluster loop — periodically group new faces into people.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from bridge_client import get_bridge
from clip import embed_pil
from cluster import cluster_once
from config import settings
from faces import detect_faces
from geocode import reverse_geocode
from store import (
    claim_photo_for_download,
    claim_photo_for_processing,
    get_photos,
    init_db,
    insert_clip,
    insert_face,
    mark_deleted,
    set_photo_done,
    set_photo_error,
    set_photo_deleted,
    upsert_photos,
)

log = logging.getLogger("indexer")

_pending: queue.Queue[str] = queue.Queue()  # uids whose thumbnail is ready


def _thumb_path(uid: str) -> Path:
    return settings.thumb_dir / f"{uid}.webp"


def _work_path(uid: str) -> Path:
    return settings.work_dir / f"{uid}.webp"


def _norm_bbox(bbox: list, w: int, h: int) -> list:
    x1, y1, x2, y2 = bbox
    return [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]


# --- sync loop -------------------------------------------------------------

def _sync_once() -> None:
    bridge = get_bridge()

    if settings.sync_limit:
        # Limited sync (testing): fetch full metadata for the N most recent.
        items = bridge.timeline(settings.sync_limit)
        rows = _rows_from_items(items)
        new = upsert_photos(rows)
        # Skip 'gone' detection on a limited sync (would flag everything older
        # as deleted).
        log.info("sync: %d items, %d new (limited)", len(rows), new)
        return

    # Full sync: diff a cheap uid listing against the local index, then fetch
    # full metadata only for photos we haven't seen before.
    ids = bridge.timeline_ids()
    remote = {i["uid"] for i in ids}
    with _db_conn() as conn:
        stored = {r["uid"] for r in conn.execute("SELECT uid FROM photos")}

    new_uids = sorted(remote - stored)
    gone = sorted(stored - remote)
    if gone:
        mark_deleted(gone)

    if new_uids:
        items = bridge.nodes(new_uids)
        rows = _rows_from_items(items)
        new = upsert_photos(rows)
        log.info("sync: %d remote, %d new, %d gone", len(remote), new, len(gone))
    else:
        log.info("sync: %d remote, no new, %d gone", len(remote), len(gone))

    # Retry photos that failed previously (e.g. transient network errors).
    with _db_conn() as conn:
        conn.execute("UPDATE photos SET status='new' WHERE status='error'")


def _rows_from_items(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        if item.get("missing"):
            continue
        rows.append(
            {
                "uid": item["uid"],
                "name": item.get("name"),
                "media_type": item.get("mediaType"),
                "capture_time": _epoch(item.get("captureTime")),
                "sha1": item.get("sha1"),
                "albums": item.get("albums") or [],
            }
        )
    return rows


def _epoch(ts) -> int | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return int(ts)
    try:
        import datetime as _dt

        return int(_dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def _db_conn():
    import sqlite3

    conn = sqlite3.connect(settings.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


# --- downloader ------------------------------------------------------------

def _downloader_loop() -> None:
    bridge = get_bridge()
    while True:
        try:
            photos = get_photos("new", limit=settings.thumbnails_batch)
            if not photos:
                time.sleep(5)
                continue
            uids = [r["uid"] for r in photos]
            claimed = [u for u in uids if claim_photo_for_download(u)]
            if not claimed:
                time.sleep(5)
                continue
            try:
                resp = bridge.thumbnails(claimed)
            except Exception as exc:
                log.warning("thumbnail batch failed: %s", exc)
                for u in claimed:
                    set_photo_error(u, str(exc)[:300])
                continue
            for r in resp.get("results", []):
                uid = r.get("uid")
                if not uid:
                    continue
                if r.get("ok"):
                    _pending.put(uid)
                else:
                    err = r.get("error", "thumbnail unavailable")
                    if "no image preview" in str(err).lower():
                        # e.g. videos: nothing to index, but we did our best.
                        set_photo_done(uid, "", None, None)
                    else:
                        set_photo_error(uid, str(err)[:300])
        except Exception as exc:  # pragma: no cover
            log.exception("downloader loop error: %s", exc)
            time.sleep(10)


# --- recognition worker ---------------------------------------------------

def _worker_loop() -> None:
    while True:
        try:
            uid = _pending.get(timeout=10)
        except queue.Empty:
            # No queued thumbnails; pick any 'downloading' photo left over
            # from a previous run and reprocess it.
            row = get_photos("downloading", limit=1)
            if not row:
                time.sleep(5)
                continue
            uid = row[0]["uid"]
        try:
            _process_one(uid)
        except Exception as exc:  # pragma: no cover
            log.exception("failed processing %s: %s", uid, exc)
            set_photo_error(uid, str(exc)[:300])


def _process_one(uid: str) -> None:
    if not claim_photo_for_processing(uid):
        return

    work = _work_path(uid)
    if not work.exists():
        set_photo_error(uid, "work file missing")
        return

    with Image.open(work) as img:
        w, h = img.size
        rgb = img.convert("RGB")
        arr = np.asarray(rgb)
        bgr = arr[:, :, ::-1].copy()  # PIL -> OpenCV BGR

    clip_vec = embed_pil(rgb)

    gps = None
    place = None
    photo = _photo_row(uid)
    if photo and photo["gps_lat"] is not None:
        gps = (photo["gps_lat"], photo["gps_lng"])
        place = reverse_geocode(gps[0], gps[1]) if gps[0] else None

    faces = detect_faces(bgr)
    face_count = len(faces)
    for f in faces:
        insert_face(
            photo_uid=uid,
            person_id=None,
            confidence=f["confidence"],
            bbox=_norm_bbox(f["bbox"], w, h),
            embedding=f["embedding"].tobytes(),
        )

    # Persist thumbnail into the cache, then remove the work file.
    final = _thumb_path(uid)
    work.replace(final)

    if clip_vec is not None:
        insert_clip(uid, clip_vec.tobytes())

    set_photo_done(uid, final.name, gps, place)
    log.debug("processed %s: %d faces, clip=%s", uid, face_count, clip_vec is not None)


def _photo_row(uid: str):
    with _db_conn() as conn:
        return conn.execute("SELECT * FROM photos WHERE uid=?", (uid,)).fetchone()


# --- cluster loop ----------------------------------------------------------

def _cluster_loop() -> None:
    while True:
        time.sleep(settings.cluster_interval)
        try:
            cluster_once()
        except Exception as exc:  # pragma: no cover
            log.exception("cluster loop error: %s", exc)


# --- GPS enrichment (local Takeout sidecars) ------------------------------

def _gps_from_local(photo_uid: str) -> None:
    """Backfill GPS from a local Google Takeout export when available.

    Matches by photo uid <-> filename heuristics is fragile; this is a no-op
    unless a future implementation provides a lookup. Left as an extension
    point — see README.
    """
    _ = photo_uid


# --- public entry point ----------------------------------------------------

def start() -> list[threading.Thread]:
    """Start all background threads. Returns the started threads."""
    init_db()
    threads = [
        threading.Thread(target=_sync_loop, name="sync", daemon=True),
        threading.Thread(target=_downloader_loop, name="downloader", daemon=True),
        threading.Thread(target=_cluster_loop, name="cluster", daemon=True),
    ]
    for i in range(settings.workers):
        threads.append(threading.Thread(target=_worker_loop, name=f"worker-{i}", daemon=True))
    for t in threads:
        t.start()
    log.info("indexer started with %d workers", settings.workers)
    return threads


def _sync_loop() -> None:
    while True:
        try:
            _sync_once()
        except Exception as exc:  # pragma: no cover
            log.exception("sync loop error: %s", exc)
        time.sleep(settings.sync_interval)


def backfill_gps() -> int:
    """One-shot: attach GPS from a local Takeout export by sha1 match.

    Google Takeout stores one `<photo>.<ext>.supplemental-metadata.json` sidecar
    next to each photo. We sha1 the local photo file (which equals Proton's
    contentHash) and join it against the indexed timeline.
    """
    if not settings.photos_dir:
        return 0
    root = Path(settings.photos_dir)
    if not root.exists():
        return 0

    import hashlib

    def _sha1(path: Path) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # Map local file sha1 -> (lat, lng) by deriving the photo path from each sidecar.
    import json as _json

    sha1_to_gps: dict[str, tuple[float, float]] = {}
    for sidecar in root.rglob("*.supplemental-metadata.json"):
        try:
            data = _json.loads(sidecar.read_text())
        except Exception:
            continue
        geo = data.get("geoData") or {}
        lat, lng = geo.get("latitude"), geo.get("longitude")
        if not (lat and lng):
            continue
        # sidecar name is "<photo>.<ext>.supplemental-metadata.json"
        photo_path = sidecar.parent / sidecar.name.removesuffix(".supplemental-metadata.json")
        if not photo_path.exists():
            continue
        sha1_to_gps[_sha1(photo_path)] = (lat, lng)
    log.info("gps backfill: %d local photos with GPS", len(sha1_to_gps))

    matched = 0
    with _db_conn() as conn:
        photos = conn.execute(
            "SELECT uid, sha1 FROM photos WHERE sha1 IS NOT NULL"
        ).fetchall()
        for row in photos:
            gps = sha1_to_gps.get(row["sha1"])
            if gps:
                conn.execute(
                    "UPDATE photos SET gps_lat=?, gps_lng=? WHERE uid=?",
                    (gps[0], gps[1], row["uid"]),
                )
                matched += 1
    log.info("gps backfill matched %d photos", matched)
    return matched


def cleanup_deleted() -> None:
    """Remove local thumbnails for photos that were deleted from Proton."""
    with _db_conn() as conn:
        rows = conn.execute(
            "SELECT uid, thumb_path FROM photos WHERE status='deleted'"
        ).fetchall()
    for r in rows:
        if r["thumb_path"]:
            p = settings.thumb_dir / r["thumb_path"]
            if p.exists():
                p.unlink(missing_ok=True)
        set_photo_deleted(r["uid"])
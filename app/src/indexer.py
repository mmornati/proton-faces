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
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from bridge_client import BridgeTransientError, get_bridge
from clip import embed_pil
from cluster import cluster_once
from config import settings
from faces import detect_faces
from geocode import reverse_geocode_many
from store import (
    backfill_fullres_images,
    claim_photo_for_download,
    claim_photo_for_full,
    claim_photo_for_processing,
    confirm_deletions,
    get_photos,
    init_db,
    insert_clip,
    insert_face,
    mark_deleted,
    mark_pending_removal,
    reset_stuck_fullres,
    set_photo_done,
    set_photo_duration,
    set_photo_error,
    set_photo_full,
    set_photo_deleted,
    sync_albums,
    upsert_photos,
)

log = logging.getLogger("indexer")

_pending: queue.Queue[str] = queue.Queue()  # uids whose thumbnail is ready

# Lightweight runtime state surfaced via the `/api/status` endpoint.
# Plain int/float writes are GIL-atomic, so we don't need a lock.
_runtime: dict = {
    "started_at": None,        # float epoch seconds, set in start()
    "last_sync": None,         # float epoch seconds, set in _sync_once()
    "last_sync_error": None,   # str or None
    "last_cluster": None,      # float epoch seconds, set after cluster_once()
    "last_gps": None,          # float epoch seconds, set after gps run
    "threads": {},             # name -> bool alive
    "pending_in_queue": 0,     # _pending.qsize() snapshot
}


def _record_threads(threads: list[threading.Thread]) -> None:
    _runtime["threads"] = {t.name: True for t in threads}


def get_indexer_state() -> dict:
    """Snapshot of the indexer's runtime state for `/api/status`.

    Cheap: no DB queries, no I/O. Safe to call frequently.
    """
    try:
        pending = _pending.qsize()
    except Exception:  # pragma: no cover
        pending = 0
    threads = {t.name: bool(t.is_alive()) for t in threading.enumerate()}
    # Prefer the authoritative set recorded at start(); fall back to whatever
    # happens to be alive now if start() hasn't recorded anything yet.
    recorded = _runtime.get("threads") or {}
    live = {n: threads.get(n, False) for n in recorded}
    _runtime["pending_in_queue"] = pending
    # If this process never started the indexer (the post-split default:
    # the API container, with RUN_INDEXER unset), surface an honest
    # "remote" state so the UI doesn't show a wall of dead threads. The
    # indexer container owns the live runtime.
    if not recorded:
        return {
            "started_at": None,
            "last_sync": None,
            "last_sync_error": None,
            "last_cluster": None,
            "last_gps": None,
            "pending_in_queue": 0,
            "threads": {},
            "remote": True,
        }
    return {
        "started_at": _runtime.get("started_at"),
        "last_sync": _runtime.get("last_sync"),
        "last_sync_error": _runtime.get("last_sync_error"),
        "last_cluster": _runtime.get("last_cluster"),
        "last_gps": _runtime.get("last_gps"),
        "pending_in_queue": pending,
        "threads": live,
    }


def _rebuild_pending() -> None:
    """Re-queue photos that were mid-pipeline when the process last stopped.

    `_pending` is in-memory only, so after a restart every photo whose
    thumbnail/work file was already generated but never processed is stuck in
    'downloading' status. Workers only process one such leftover per 10s
    timeout, which would take days for a large backlog. Re-queue any
    'downloading' photo that already has its work file on disk so the workers
    resume immediately.
    """
    reenqueued = 0
    offset = 0
    batch = 2000
    while True:
        rows = get_photos("downloading", limit=batch, offset=offset)
        if not rows:
            break
        for row in rows:
            uid = row["uid"]
            if _work_path(uid).exists():
                _pending.put(uid)
                reenqueued += 1
        offset += len(rows)
    if reenqueued:
        log.info("rebuild_pending: re-queued %d downloaded photos", reenqueued)


def _thumb_path(uid: str) -> Path:
    return settings.thumb_dir / f"{uid}.webp"


def _work_path(uid: str) -> Path:
    return settings.work_dir / f"{uid}.webp"


def _norm_bbox(bbox: list, w: int, h: int) -> list:
    x1, y1, x2, y2 = bbox
    return [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]


# --- sync loop -------------------------------------------------------------

def _sync_once() -> None:
    _runtime["last_sync"] = time.time()
    _runtime["last_sync_error"] = None
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

    # Two-phase deletion with grace period (issue from Sep 1 Proton outage):
    # stale timeline listings can briefly drop uids, which would otherwise
    # cascade into a mass false-deletion. Stage them as pending_removal first;
    # confirm_deletions() promotes them to 'deleted' only after
    # grace_cycles * SYNC_INTERVAL seconds have elapsed without the uid
    # coming back. If the uid reappears in `remote` before then, the
    # upsert_photos reclaim path resets it to 'new' (and clears was_deleted_at).
    if gone:
        staged = mark_pending_removal(gone)
        grace_seconds = max(1, settings.grace_cycles) * max(1, settings.sync_interval)
        confirmed = confirm_deletions(grace_seconds=grace_seconds)
        log.info(
            "sync: %d missing → %d newly pending_removal, %d confirmed deleted "
            "(grace=%ds, %d still in grace)",
            len(gone), staged, confirmed, grace_seconds,
            len(gone) - staged - confirmed,
        )

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

    # DEMO_MODE only: inject GPS/place from the fixture so the Places view
    # works out of the box. No-op outside demo mode.
    if os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from demo import apply_demo_gps
            apply_demo_gps()
        except Exception as exc:  # pragma: no cover
            log.warning("demo gps inject failed: %s", exc)


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
                        # No server-side preview. Both HEIC/HEIF images AND
                        # videos get routed to the fullres loop: the loop
                        # dispatches on media_type to either decode-with-Pillow
                        # or extract-a-frame-with-ffmpeg.
                        if _is_image(uid) or _is_video(uid):
                            set_photo_full(uid)
                        else:
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

    # GPS/place is enriched by the gps loop (subprocess) — never geocode from
    # a worker thread: reverse_geocoder forks a multiprocessing pool which
    # deadlocks inside the app's threaded process.
    photo = _photo_row(uid)
    gps = (
        (photo["gps_lat"], photo["gps_lng"])
        if photo and photo["gps_lat"] is not None
        else None
    )
    place = photo["place"] if photo else None

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


def _is_image(uid: str) -> bool:
    row = _photo_row(uid)
    return bool(row and row["media_type"] and str(row["media_type"]).startswith("image/"))


def _is_video(uid: str) -> bool:
    row = _photo_row(uid)
    return bool(row and row["media_type"] and str(row["media_type"]).startswith("video/"))


# --- fullres loop (local thumbnails for images without server preview) ----

def _fullres_loop() -> None:
    """Download full-res photos that lack a server preview and generate a thumbnail.

    Two strategies, dispatched on media_type:

    - **Image** (HEIC/HEIF and anything Pillow+heif can decode): download full-res
      bytes once, decode with Pillow, downscale to 512px WebP. Full-res is
      discarded. The thumbnail then flows through the normal worker pipeline
      (CLIP + face detection).

    - **Video**: download the full-res video once, extract a mid-roll poster
      frame via ffmpeg, downscale to 512px WebP, and read the duration via
      ffprobe. Videos skip CLIP/faces and are marked done immediately.

    In both cases the full-res bytes are deleted after processing; only the
    small thumbnail (or for videos, the poster) is kept on disk.
    """
    while True:
        try:
            photos = get_photos("full", limit=1)
            if not photos:
                # Crash recovery: any photo stuck in 'fullres' gets retried.
                stuck = get_photos("fullres", limit=1)
                if not stuck:
                    time.sleep(5)
                    continue
                photos = stuck
            uid = photos[0]["uid"]
            if not claim_photo_for_full(uid):
                continue

            photo = _photo_row(uid)
            bridge = get_bridge()
            tmp = settings.work_dir / f"{uid}.download"
            try:
                resp = bridge.full_photo(uid)
                try:
                    resp.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_bytes(1 << 16):
                            fh.write(chunk)
                finally:
                    resp.close()

                if _is_video(uid):
                    _video_poster(tmp, uid)
                    tmp.unlink(missing_ok=True)
                    log.info("fullres: generated video poster for %s", uid)
                    with _db_conn() as conn:
                        conn.execute(
                            "UPDATE photos SET status='done', thumb_path=?, processed_at=?, error=NULL WHERE uid=?",
                            (_thumb_path(uid).name, int(time.time()), uid),
                        )
                else:
                    _resize_to_thumb(tmp, _work_path(uid))
                    log.info("fullres: generated thumbnail for %s", uid)
                    tmp.unlink(missing_ok=True)
                    # Hand off to the normal worker pipeline.
                    with _db_conn() as conn:
                        conn.execute(
                            "UPDATE photos SET status='downloading' WHERE uid=?", (uid,)
                        )
                    _pending.put(uid)
            except BridgeTransientError as exc:
                # Bridge returned 429/502/503 — back off rather than failing
                # the row. Release the claim (set status back to 'full' so
                # other rows can be tried in the meantime; the startup reset
                # hook + periodic resets will retry). Sleep at least the
                # upstream's Retry-After hint, otherwise a sensible default.
                log.warning(
                    "fullres transient %s for %s; sleeping %.0fs before retry",
                    exc.status_code, uid, exc.retry_after_sec,
                )
                tmp.unlink(missing_ok=True)
                with _db_conn() as conn:
                    conn.execute(
                        "UPDATE photos SET status='full', processed_at=?, error=NULL WHERE uid=?",
                        (int(time.time()), uid),
                    )
                # Default to 30s when no Retry-After is supplied; clamp to the
                # same 600s ceiling we use in the bridge parser.
                time.sleep(min(600.0, max(exc.retry_after_sec, 30.0)))
            except Exception as exc:
                log.warning("fullres failed for %s: %s", uid, exc)
                tmp.unlink(missing_ok=True)
                set_photo_error(uid, str(exc)[:300])
        except Exception as exc:  # pragma: no cover
            log.exception("fullres loop error: %s", exc)
            time.sleep(10)


def _video_poster(src: Path, uid: str) -> None:
    """Extract a poster frame from a video and write it as 512px WebP.

    Picks the seek point at ~10% of the duration (skips black leader frames
    common in phone-captured video). Reads the duration via ffprobe so the UI
    can show the clip length. Requires ``ffmpeg`` and ``ffprobe`` on PATH
    (the app Dockerfile installs both).
    """
    import subprocess

    # Probe duration first — ffmpeg can do it too but ffprobe is more robust
    # against truncated files and reports the container-format duration.
    duration_sec: float | None = None
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(src),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0 and out.stdout.strip():
            duration_sec = float(out.stdout.strip())
    except Exception as exc:  # pragma: no cover
        log.warning("ffprobe failed for %s: %s", uid, exc)

    # Seek to ~10% in — avoids 0-second black first frames typical of phone video.
    seek: list[str] = []
    if duration_sec and duration_sec > 1:
        seek = ["-ss", f"{duration_sec * 0.1:.2f}"]

    dest = _thumb_path(uid)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *seek, "-i", str(src),
        "-frames:v", "1",
        "-vf", "scale='min(512,iw)':'min(512,ih)':force_original_aspect_ratio=decrease",
        "-update", "1",  # write a single image file, not a mov/mjpeg
        "-q:v", "5",      # mjpeg quality (we re-encode to WebP next, this is just a fast frame)
        str(dest),
    ]
    subprocess.run(cmd, check=True, timeout=120)
    # Re-encode the extracted frame as actual WebP (ffmpeg's webp muxer is slow;
    # using Pillow matches the image-thumb pipeline for cache uniformity).
    if dest.exists():
        from PIL import Image
        with Image.open(dest) as img:
            img = img.convert("RGB")
            img.save(dest, format="WEBP", quality=82, method=6)

    if duration_sec is not None:
        set_photo_duration(uid, duration_sec)


def _resize_to_thumb(src: Path, dest: Path, max_side: int = 512) -> None:
    """Decode src (any Pillow format incl. HEIC) and write a max-512px WebP.

    WebP gives us ~30% smaller thumbs than JPEG at comparable visual quality,
    and the API endpoint serves ``image/webp`` natively — so the bytes and the
    extension finally match.
    """
    from PIL import Image

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except Exception:  # pragma: no cover
        pass
    with Image.open(src) as img:
        img.thumbnail((max_side, max_side))
        img = img.convert("RGB")
        img.save(dest, format="WEBP", quality=82, method=6)


# --- cluster loop ----------------------------------------------------------

def _cluster_loop() -> None:
    while True:
        time.sleep(settings.cluster_interval)
        try:
            cluster_once()
            _runtime["last_cluster"] = time.time()
        except Exception as exc:  # pragma: no cover
            log.exception("cluster loop error: %s", exc)


# --- albums sync ----------------------------------------------------------

def _albums_loop() -> None:
    while True:
        try:
            _sync_albums_once()
        except Exception as exc:  # pragma: no cover
            log.exception("albums sync error: %s", exc)
        time.sleep(settings.sync_interval)


def _sync_albums_once() -> int:
    """Fetch album names from the bridge and recompute local covers/counts."""
    try:
        data = get_bridge().albums()
        albums = data.get("albums", []) if isinstance(data, dict) else data
    except Exception as exc:
        log.warning("albums fetch failed: %s", exc)
        return 0
    if not albums:
        return 0
    n = sync_albums(albums)
    log.info("albums: synced %d albums", n)
    return n


# --- public entry point ----------------------------------------------------

def start() -> list[threading.Thread]:
    """Start all background threads. Returns the started threads."""
    init_db()
    _runtime["started_at"] = time.time()
    # Recover photos stuck in 'processing' from a previous crash/hang.
    with _db_conn() as conn:
        conn.execute("UPDATE photos SET status='downloading' WHERE status='processing'")
    try:
        n = backfill_fullres_images()
        if n:
            log.info("backfilled %d image photos for local thumbnail generation", n)
    except Exception as exc:  # pragma: no cover
        log.warning("fullres backfill failed: %s", exc)
    # Unstick videos (and any other media) that have been parked in
    # status='full' for too long. Triggered on every startup so a bridge
    # outage (e.g. the Sep 1 partial outage that left 2,286 videos stuck)
    # drains automatically when the bridge recovers.
    try:
        requeued, parked = reset_stuck_fullres(retry_after_sec=settings.fullres_retry_after_sec)
        if requeued or parked:
            log.info(
                "stuck full-res recovery: %d re-queued, %d parked as error",
                requeued, parked,
            )
    except Exception as exc:  # pragma: no cover
        log.warning("stuck fullres recovery failed: %s", exc)
    _rebuild_pending()
    try:
        _sync_albums_once()
    except Exception as exc:  # pragma: no cover
        log.warning("initial albums sync failed: %s", exc)
    threads = [
        threading.Thread(target=_sync_loop, name="sync", daemon=True),
        threading.Thread(target=_downloader_loop, name="downloader", daemon=True),
        threading.Thread(target=_fullres_loop, name="fullres", daemon=True),
        threading.Thread(target=_cluster_loop, name="cluster", daemon=True),
        threading.Thread(target=_gps_loop, name="gps", daemon=True),
        threading.Thread(target=_albums_loop, name="albums", daemon=True),
    ]
    for i in range(settings.workers):
        threads.append(threading.Thread(target=_worker_loop, name=f"worker-{i}", daemon=True))
    for t in threads:
        t.start()
    _record_threads(threads)
    log.info("indexer started with %d workers", settings.workers)
    return threads


def _sync_loop() -> None:
    while True:
        try:
            _sync_once()
            try:
                cleanup_deleted()
            except Exception as exc:  # pragma: no cover
                log.warning("cleanup_deleted failed: %s", exc)
        except Exception as exc:  # pragma: no cover
            log.exception("sync loop error: %s", exc)
            _runtime["last_sync_error"] = f"{type(exc).__name__}: {exc}"
        time.sleep(settings.sync_interval)


# --- GPS enrichment (local Takeout sidecars) ------------------------------

def backfill_gps(rebuild_cache: bool = False) -> int:
    """Attach GPS from a local Takeout export by sha1 match.

    Google Takeout stores one `<photo>.<ext>.supplemental-metadata.json` sidecar
    next to each photo. We sha1 the local photo file (which equals Proton's
    contentHash) and join it against the indexed timeline.

    The sha1 -> (lat, lng) map is expensive to build (~136k hashes on the
    server), so it is persisted to data/gps_sha1_cache.json and reloaded on
    later runs unless rebuild_cache=True.
    """
    if not settings.photos_dir:
        return 0
    root = Path(settings.photos_dir)
    if not root.exists():
        return 0

    import hashlib
    import json as _json

    cache_path = settings.data_dir / "gps_sha1_cache.json"

    def _sha1(path: Path) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    sha1_to_gps: dict[str, tuple[float, float]] = {}
    if not rebuild_cache and cache_path.exists():
        try:
            raw = _json.loads(cache_path.read_text())
            sha1_to_gps = {
                k: (float(v[0]), float(v[1])) for k, v in raw.items()
            }
            log.info("gps backfill: loaded %d entries from cache", len(sha1_to_gps))
            return _apply_gps(sha1_to_gps)
        except Exception as exc:  # pragma: no cover
            log.warning("gps cache unreadable (%s); rebuilding", exc)

    # Map local file sha1 -> (lat, lng) by deriving the photo path from each sidecar.
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

    try:
        cache_path.write_text(
            _json.dumps({k: list(v) for k, v in sha1_to_gps.items()})
        )
        log.info("gps backfill: wrote %d entries to cache", len(sha1_to_gps))
    except Exception as exc:  # pragma: no cover
        log.warning("gps cache write failed: %s", exc)

    return _apply_gps(sha1_to_gps)


def _apply_gps(sha1_to_gps: dict[str, tuple[float, float]]) -> int:
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


def enrich_places() -> int:
    """Reverse-geocode every photo that has GPS but no place yet (idempotent)."""
    with _db_conn() as conn:
        rows = conn.execute(
            "SELECT uid, gps_lat, gps_lng FROM photos "
            "WHERE gps_lat IS NOT NULL AND gps_lng IS NOT NULL AND place IS NULL"
        ).fetchall()
    if not rows:
        return 0

    points = [(r["gps_lat"], r["gps_lng"]) for r in rows]
    by_point = reverse_geocode_many(points)
    matched = 0
    with _db_conn() as conn:
        for row, place in zip(rows, [by_point[(r["gps_lat"], r["gps_lng"])] for r in rows]):
            if not place:
                continue
            conn.execute(
                "UPDATE photos SET place=? WHERE uid=?", (place, row["uid"])
            )
            matched += 1
    log.info("gps place enrichment: %d photos", matched)
    return matched


def _gps_loop() -> None:
    """Run GPS backfill + place enrichment in a *subprocess*.

    reverse_geocoder forks a multiprocessing pool on first use, which
    deadlocks when called from a thread inside the app process. Running the
    same work as a child process (python main.py --backfill-gps) keeps the
    fork in a single-threaded process, where it works reliably.
    """
    while True:
        time.sleep(settings.gps_interval)
        try:
            _run_gps_subprocess()
            _runtime["last_gps"] = time.time()
        except Exception as exc:  # pragma: no cover
            log.exception("gps loop error: %s", exc)


def _run_gps_subprocess() -> None:
    import subprocess
    import sys

    if not settings.photos_dir:
        return
    log.info("gps loop: starting backfill+enrich subprocess")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "main", "--backfill-gps"],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=60 * 60,
        )
        log.info(
            "gps loop: subprocess exit=%d stdout=%s stderr=%s",
            proc.returncode,
            proc.stdout[-300:],
            proc.stderr[-300:],
        )
    except subprocess.TimeoutExpired:  # pragma: no cover
        log.warning("gps loop: subprocess timed out after 1h")


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
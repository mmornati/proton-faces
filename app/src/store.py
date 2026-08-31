"""SQLite persistence layer for proton-faces."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    uid          TEXT PRIMARY KEY,
    name         TEXT,
    media_type   TEXT,
    capture_time INTEGER,
    sha1         TEXT,
    albums       TEXT,           -- JSON array of album node uids
    size_bytes   INTEGER,        -- Proton claimedSize (UI display, video poster gate)
    duration_sec REAL,           -- video duration (ffprobe); NULL for images
    favorited    INTEGER NOT NULL DEFAULT 0,    -- local star
    archived     INTEGER NOT NULL DEFAULT 0,    -- hidden from default grids
    hidden       INTEGER NOT NULL DEFAULT 0,    -- user hid it (e.g. resolved duplicate)
    tags         TEXT,           -- JSON array of freeform user tags
    status       TEXT NOT NULL DEFAULT 'new',  -- new|downloading|done|error|deleted
    thumb_path   TEXT,           -- relative path under DATA_DIR/thumbs
    gps_lat      REAL,
    gps_lng      REAL,
    place        TEXT,           -- reverse-geocoded human place name
    processed_at INTEGER,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
CREATE INDEX IF NOT EXISTS idx_photos_place  ON photos(place);
CREATE INDEX IF NOT EXISTS idx_photos_time   ON photos(capture_time);
CREATE INDEX IF NOT EXISTS idx_photos_favorited ON photos(favorited);
CREATE INDEX IF NOT EXISTS idx_photos_archived  ON photos(archived);
CREATE INDEX IF NOT EXISTS idx_photos_hidden   ON photos(hidden);
CREATE INDEX IF NOT EXISTS idx_photos_sha1     ON photos(sha1);
-- Partial index covering exactly the rows done_photos() returns: status='done'
-- AND a non-empty thumb_path. SQLite walks it in DESC order with no table scan
-- and no temp sort, dropping /api/photos cold latency from ~150 ms to a few
-- ms on 79 k-row DBs (see issue #5). Tiny, write-time-only-maintained index.
CREATE INDEX IF NOT EXISTS idx_photos_done_time
  ON photos(capture_time DESC)
  WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '';

CREATE TABLE IF NOT EXISTS people (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT,
    cover_uid     TEXT,          -- representative photo uid
    cover_face_id INTEGER,       -- representative face id (for face-crop covers)
    created       INTEGER
);

CREATE TABLE IF NOT EXISTS faces (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_uid TEXT NOT NULL REFERENCES photos(uid) ON DELETE CASCADE,
    person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
    confidence REAL,
    bbox      TEXT,              -- JSON [x, y, w, h] normalized
    embedding BLOB               -- raw float32 (512,)
);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_photo  ON faces(photo_uid);
CREATE INDEX IF NOT EXISTS idx_faces_person_photo ON faces(person_id, photo_uid);

CREATE TABLE IF NOT EXISTS clips (
    photo_uid TEXT PRIMARY KEY REFERENCES photos(uid) ON DELETE CASCADE,
    embedding BLOB               -- raw float32 (512,) CLIP embedding
);

CREATE TABLE IF NOT EXISTS albums (
    uid          TEXT PRIMARY KEY,
    name         TEXT,
    cover_uid    TEXT,            -- representative (newest) photo uid
    photo_count  INTEGER,
    start_ts     INTEGER,         -- earliest capture_time in the album (sort key)
    end_ts       INTEGER,         -- latest capture_time in the album
    synced_at    INTEGER
);
"""

_lock = threading.Lock()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        # Migrations first: existing DBs need ALTER TABLE ADD COLUMN before the
        # _SCHEMA block runs CREATE INDEX against those new columns.
        migrate(conn)
        conn.executescript(_SCHEMA)


def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column migrations for older databases."""
    pcols = {r["name"] for r in conn.execute("PRAGMA table_info(photos)")}
    if "size_bytes" not in pcols:
        conn.execute("ALTER TABLE photos ADD COLUMN size_bytes INTEGER")
    if "duration_sec" not in pcols:
        conn.execute("ALTER TABLE photos ADD COLUMN duration_sec REAL")
    if "favorited" not in pcols:
        conn.execute("ALTER TABLE photos ADD COLUMN favorited INTEGER NOT NULL DEFAULT 0")
    if "archived" not in pcols:
        conn.execute("ALTER TABLE photos ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
    if "hidden" not in pcols:
        conn.execute("ALTER TABLE photos ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
    if "tags" not in pcols:
        conn.execute("ALTER TABLE photos ADD COLUMN tags TEXT")
    # Indexes for the new columns.
    idx = {r["name"] for r in conn.execute("PRAGMA index_list(photos)")}
    if "idx_photos_favorited" not in idx:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_favorited ON photos(favorited)")
    if "idx_photos_archived" not in idx:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_archived ON photos(archived)")
    if "idx_photos_hidden" not in idx:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_hidden ON photos(hidden)")
    if "idx_photos_sha1" not in idx:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_sha1 ON photos(sha1)")
    # Partial index for done_photos() — added in issue #5. Idempotent: if the
    # index already exists (current _SCHEMA branch), the IF NOT EXISTS is a
    # no-op. Keeps the migration path stable for older DBs that pre-date the
    # schema change.
    if "idx_photos_done_time" not in idx:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_photos_done_time "
            "ON photos(capture_time DESC) "
            "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != ''"
        )
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(people)")}
    if "cover_face_id" not in cols:
        conn.execute("ALTER TABLE people ADD COLUMN cover_face_id INTEGER")
    # Backfill a cover face for people clustered before cover_face_id existed.
    conn.execute(
        """UPDATE people
           SET cover_face_id = (
               SELECT f.id FROM faces f
               WHERE f.person_id = people.id
               ORDER BY f.id LIMIT 1
           )
           WHERE cover_face_id IS NULL
             AND EXISTS (SELECT 1 FROM faces f WHERE f.person_id = people.id)"""
    )
    acols = {r["name"] for r in conn.execute("PRAGMA table_info(albums)")}
    if "start_ts" not in acols:
        conn.execute("ALTER TABLE albums ADD COLUMN start_ts INTEGER")
    if "end_ts" not in acols:
        conn.execute("ALTER TABLE albums ADD COLUMN end_ts INTEGER")


# --- photos ---------------------------------------------------------------

def upsert_photos(rows: list[dict]) -> int:
    """Insert or update photo rows from the bridge timeline.

    Returns number of brand-new photos discovered.
    """
    if not rows:
        return 0
    new = 0
    with get_conn() as conn:
        for r in rows:
            existing = conn.execute("SELECT status FROM photos WHERE uid=?", (r["uid"],)).fetchone()
            if existing is None:
                new += 1
            conn.execute(
                """INSERT INTO photos (uid, name, media_type, capture_time, sha1, albums, size_bytes, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'new')
                   ON CONFLICT(uid) DO UPDATE SET
                       name=excluded.name,
                       media_type=excluded.media_type,
                       capture_time=excluded.capture_time,
                       sha1=excluded.sha1,
                       albums=excluded.albums,
                       size_bytes=excluded.size_bytes,
                       status=CASE WHEN photos.status='deleted' THEN 'new' ELSE photos.status END
                """,
                (
                    r["uid"],
                    r.get("name"),
                    r.get("media_type"),
                    r.get("capture_time"),
                    r.get("sha1"),
                    json.dumps(r.get("albums", [])) if r.get("albums") else None,
                    r.get("size"),
                ),
            )
    return new


def mark_deleted(uids: list[str]) -> None:
    if not uids:
        return
    with get_conn() as conn:
        conn.executemany(
            "UPDATE photos SET status='deleted' WHERE uid=? AND status!='deleted'",
            [(u,) for u in uids],
        )


def get_photos(status: str, limit: int = 500, offset: int = 0) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM photos WHERE status=? ORDER BY capture_time ASC LIMIT ? OFFSET ?",
            (status, limit, offset),
        ).fetchall()


def claim_photo_for_download(uid: str) -> bool:
    """Atomically move a photo from 'new' to 'downloading'."""
    with _lock, get_conn() as conn:
        cur = conn.execute(
            "UPDATE photos SET status='downloading' WHERE uid=? AND status='new'", (uid,)
        )
        return cur.rowcount == 1


def claim_photo_for_processing(uid: str) -> bool:
    """Atomically move a photo from 'downloading' to 'processing'."""
    with _lock, get_conn() as conn:
        cur = conn.execute(
            "UPDATE photos SET status='processing' WHERE uid=? AND status='downloading'", (uid,)
        )
        return cur.rowcount == 1


def set_photo_done(uid: str, thumb_path: str, gps: tuple[float, float] | None, place: str | None) -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE photos SET status='done', thumb_path=?, gps_lat=?, gps_lng=?, place=?,
               processed_at=?, error=NULL WHERE uid=?""",
            (
                thumb_path,
                gps[0] if gps else None,
                gps[1] if gps else None,
                place,
                int(time.time()),
                uid,
            ),
        )


def set_photo_full(uid: str) -> None:
    """Mark a photo as needing a locally-generated thumbnail (no server preview)."""
    with get_conn() as conn:
        conn.execute("UPDATE photos SET status='full', error=NULL WHERE uid=?", (uid,))


def set_photo_duration(uid: str, duration_sec: float | None) -> None:
    """Persist the ffprobe-measured video duration (no-op for None)."""
    if duration_sec is None:
        return
    with get_conn() as conn:
        conn.execute("UPDATE photos SET duration_sec=? WHERE uid=?", (duration_sec, uid))


def claim_photo_for_full(uid: str) -> bool:
    """Atomically move a photo from 'full' to 'fullres' (full-res being downloaded)."""
    with _lock, get_conn() as conn:
        cur = conn.execute(
            "UPDATE photos SET status='fullres' WHERE uid=? AND status='full'", (uid,)
        )
        return cur.rowcount == 1


def backfill_fullres_images() -> int:
    """Requeue photos that finished without a thumbnail.

    Used at startup to pick up images (e.g. HEIC) that finished before we
    generated local thumbnails, AND videos that pre-date the ffmpeg poster
    pipeline. Videos get a separate requeue so the fullres loop picks the
    right strategy per media type.
    """
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE photos SET status='full', error=NULL "
            "WHERE status='done' AND (thumb_path IS NULL OR thumb_path='') "
            "AND media_type LIKE 'image/%'"
        )
        n = cur.rowcount
        # Videos go to the same 'full' bucket; the fullres loop dispatches on
        # media_type, not status.
        cur = conn.execute(
            "UPDATE photos SET status='full', error=NULL "
            "WHERE status='done' AND (thumb_path IS NULL OR thumb_path='') "
            "AND media_type LIKE 'video/%'"
        )
        n += cur.rowcount
    return n


def set_photo_error(uid: str, error: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE photos SET status='error', error=? WHERE uid=?", (error, uid))


def set_photo_deleted(uid: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE photos SET status='deleted' WHERE uid=?", (uid,))


def get_photo(uid: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM photos WHERE uid=?", (uid,)).fetchone()


def stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        done = conn.execute("SELECT COUNT(*) FROM photos WHERE status='done'").fetchone()[0]
        done_with_thumb = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != ''"
        ).fetchone()[0]
        done_without_thumb = done - done_with_thumb
        pending = conn.execute("SELECT COUNT(*) FROM photos WHERE status IN ('new','downloading','processing')").fetchone()[0]
        faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        clips = conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        people = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        by_status = {
            r["status"]: r["n"]
            for r in conn.execute("SELECT status, COUNT(*) AS n FROM photos GROUP BY status")
        }
    return {
        "photos": {
            "total": total,
            "done": done,
            "done_with_thumb": done_with_thumb,
            "done_without_thumb": done_without_thumb,
            "pending": pending,
            "by_status": by_status,
        },
        "faces": faces,
        "clips": clips,
        "people": people,
    }


# --- faces & people -------------------------------------------------------

def insert_face(photo_uid: str, person_id: int | None, confidence: float, bbox: list, embedding: bytes) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO faces (photo_uid, person_id, confidence, bbox, embedding) VALUES (?,?,?,?,?)",
            (photo_uid, person_id, confidence, json.dumps(bbox), sqlite3.Binary(embedding)),
        )
        return cur.lastrowid


def all_face_rows() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, photo_uid, person_id, embedding FROM faces"
        ).fetchall()


def faces_without_person(limit: int = 5000) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM faces WHERE person_id IS NULL ORDER BY id LIMIT ?", (limit,)
        ).fetchall()


def assign_face_person(face_id: int, person_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))


def unassign_face(face_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE faces SET person_id=NULL WHERE id=?", (face_id,))


def create_person(name: str | None, cover_uid: str | None, cover_face_id: int | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO people (name, cover_uid, cover_face_id, created) VALUES (?,?,?,?)",
            (name, cover_uid, cover_face_id, int(time.time())),
        )
        return cur.lastrowid


def update_person_cover(person_id: int, cover_uid: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE people SET cover_uid=? WHERE id=? AND cover_uid IS NULL",
            (cover_uid, person_id),
        )


def set_person_cover_face(person_id: int, cover_face_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE people SET cover_face_id=? WHERE id=?", (cover_face_id, person_id))


def get_person(person_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT p.*, COUNT(f.id) AS face_count, COUNT(DISTINCT f.photo_uid) AS photo_count "
            "FROM people p LEFT JOIN faces f ON f.person_id = p.id WHERE p.id=? GROUP BY p.id",
            (person_id,),
        ).fetchone()


def face_embedding(face_id: int) -> bytes | None:
    with get_conn() as conn:
        row = conn.execute("SELECT embedding FROM faces WHERE id=?", (face_id,)).fetchone()
        return row["embedding"] if row else None


def faces_for_photo(photo_uid: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT f.id, f.person_id, f.confidence, f.bbox, p.name AS person_name
               FROM faces f LEFT JOIN people p ON p.id = f.person_id
               WHERE f.photo_uid=? ORDER BY f.id""",
            (photo_uid,),
        ).fetchall()


def unassigned_faces(limit: int = 500) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT f.id, f.photo_uid, f.confidence,
                      json_extract(f.bbox, '$[0]') AS bbox_x,
                      json_extract(f.bbox, '$[1]') AS bbox_y,
                      json_extract(f.bbox, '$[2]') AS bbox_w,
                      json_extract(f.bbox, '$[3]') AS bbox_h,
                      ph.thumb_path
               FROM faces f JOIN photos ph ON ph.uid = f.photo_uid
               WHERE f.person_id IS NULL AND ph.status='done'
               ORDER BY f.id ASC LIMIT ?""",
            (limit,),
        ).fetchall()


def similar_faces(embedding: bytes, threshold: float, limit: int = 200) -> list[sqlite3.Row]:
    """Faces (id, photo_uid, person_id, sim) whose cosine similarity to `embedding` is >= threshold."""
    rows = all_face_rows()
    emb = np.frombuffer(embedding, dtype=np.float32)
    out = []
    for r in rows:
        fe = np.frombuffer(r["embedding"], dtype=np.float32)
        s = float(fe @ emb)
        if s >= threshold:
            out.append((r["id"], r["photo_uid"], r["person_id"], s))
    out.sort(key=lambda t: t[3], reverse=True)
    return out[:limit]


def rename_person(person_id: int, name: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE people SET name=? WHERE id=?", (name, person_id))


def find_person_by_name(name: str, exclude_id: int | None = None) -> sqlite3.Row | None:
    """Return the first person with an exact (case-insensitive) name match."""
    with get_conn() as conn:
        if exclude_id is None:
            return conn.execute(
                "SELECT * FROM people WHERE LOWER(name)=LOWER(?) LIMIT 1", (name,)
            ).fetchone()
        return conn.execute(
            "SELECT * FROM people WHERE LOWER(name)=LOWER(?) AND id<>? LIMIT 1",
            (name, exclude_id),
        ).fetchone()


def merge_person(source_id: int, target_id: int) -> None:
    """Merge source person into target: move all faces, keep target's cover if
    unset, then delete the source row. The target must exist and differ."""
    if source_id == target_id:
        return
    with get_conn() as conn:
        # backfill cover fields on the target from the source when unset
        src = conn.execute(
            "SELECT name, cover_uid, cover_face_id FROM people WHERE id=?", (source_id,)
        ).fetchone()
        tgt = conn.execute(
            "SELECT name, cover_uid, cover_face_id FROM people WHERE id=?", (target_id,)
        ).fetchone()
        if src and tgt:
            # Keep the target's name; only inherit the source's when the target
            # has none (so merging a named person into an unknown one doesn't
            # lose the name).
            if not tgt["name"] and src["name"]:
                conn.execute(
                    "UPDATE people SET name=? WHERE id=?", (src["name"], target_id)
                )
            if not tgt["cover_uid"] and src["cover_uid"]:
                conn.execute(
                    "UPDATE people SET cover_uid=? WHERE id=?", (src["cover_uid"], target_id)
                )
            if not tgt["cover_face_id"] and src["cover_face_id"]:
                conn.execute(
                    "UPDATE people SET cover_face_id=? WHERE id=?",
                    (src["cover_face_id"], target_id),
                )
        conn.execute(
            "UPDATE faces SET person_id=? WHERE person_id=?", (target_id, source_id)
        )
        conn.execute("DELETE FROM people WHERE id=?", (source_id,))


def person_mean_embedding(person_id: int) -> np.ndarray | None:
    """Mean of a person's face embeddings (L2-normalized), or None."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT embedding FROM faces WHERE person_id=? AND embedding IS NOT NULL",
            (person_id,),
        ).fetchall()
    if not rows:
        return None
    mats = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    mean = mats.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        return None
    return (mean / norm).astype(np.float32)


def person_mean_embeddings() -> dict[int, np.ndarray]:
    """Mean embedding per person, computed with a single query.

    Returns {person_id: L2-normalized mean embedding} for every person that
    has at least one face embedding.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT person_id, embedding FROM faces "
            "WHERE person_id IS NOT NULL AND embedding IS NOT NULL"
        ).fetchall()
    if not rows:
        return {}
    by_person: dict[int, list[np.ndarray]] = {}
    for r in rows:
        by_person.setdefault(r["person_id"], []).append(
            np.frombuffer(r["embedding"], dtype=np.float32)
        )
    out: dict[int, np.ndarray] = {}
    for pid, mats in by_person.items():
        mean = np.stack(mats).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            out[pid] = (mean / norm).astype(np.float32)
    return out


def all_people(q: str | None = None, limit: int | None = None, offset: int = 0) -> list[sqlite3.Row]:
    """People ordered by photo_count DESC.

    With `q`, restricts to people whose name matches the LIKE pattern
    (case-insensitive prefix is encouraged). With `limit`/`offset`, paginates.
    """
    sql = (
        "SELECT p.id, p.name, p.cover_uid, p.cover_face_id, "
        "       COUNT(f.id) AS face_count, "
        "       COUNT(DISTINCT f.photo_uid) AS photo_count "
        "FROM people p LEFT JOIN faces f ON f.person_id = p.id "
    )
    params: list = []
    if q:
        sql += "WHERE LOWER(p.name) LIKE LOWER(?) "
        params.append(f"%{q}%")
    sql += "GROUP BY p.id ORDER BY photo_count DESC, p.id ASC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params += [limit, offset]
    with get_conn() as conn:
        return conn.execute(sql, params).fetchall()


def count_people(q: str | None = None) -> int:
    """Total people (optionally matching `q`). Cheap — uses no joins."""
    with get_conn() as conn:
        if q:
            return conn.execute(
                "SELECT COUNT(*) FROM people WHERE LOWER(name) LIKE LOWER(?)",
                (f"%{q}%",),
            ).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]


def photos_for_person(person_id: int, limit: int = 200, offset: int = 0) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT ph.* FROM photos ph
               WHERE ph.uid IN (
                   SELECT DISTINCT f.photo_uid FROM faces f
                   WHERE f.person_id=? AND f.photo_uid IS NOT NULL
               )
               AND ph.status='done'
               ORDER BY ph.capture_time DESC LIMIT ? OFFSET ?""",
            (person_id, limit, offset),
        ).fetchall()


def count_faces_for_person(person_id: int) -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM faces WHERE person_id=?", (person_id,)).fetchone()[0]


# --- clips ----------------------------------------------------------------

def insert_clip(photo_uid: str, embedding: bytes) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO clips (photo_uid, embedding) VALUES (?,?) ON CONFLICT(photo_uid) DO UPDATE SET embedding=excluded.embedding",
            (photo_uid, sqlite3.Binary(embedding)),
        )


def all_clips() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT photo_uid, embedding FROM clips").fetchall()


def clip_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]


# --- search helpers -------------------------------------------------------

def search_photos_by_place(query: str, limit: int = 200, offset: int = 0) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM photos WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' AND place IS NOT NULL AND place LIKE ? ORDER BY capture_time DESC LIMIT ? OFFSET ?",
            (f"%{query}%", limit, offset),
        ).fetchall()


def place_stats(limit: int = 500) -> list[sqlite3.Row]:
    """Aggregate distinct places with photo counts, most popular first.

    Returns rows with (place, city, photo_count) where city is the first
    segment of `place` (before the comma).
    """
    with get_conn() as conn:
        return conn.execute(
            "SELECT place, COUNT(*) AS photo_count FROM photos "
            "WHERE status='done' AND place IS NOT NULL "
            "GROUP BY place ORDER BY photo_count DESC LIMIT ?",
            (limit,),
        ).fetchall()


def map_markers(limit: int = 1000) -> list[sqlite3.Row]:
    """Aggregate distinct places with photo counts and mean GPS coordinates.

    Returns rows with (place, city, photo_count, lat, lng, cover_uid) where city is the
    first segment of `place` (before the comma) and cover_uid is the uid of a
    representative photo for that place.

    Performance note (issue #5): the previous implementation used a correlated
    scalar subquery for ``cover_uid`` which SQLite executes once per output
    row (``EXPLAIN QUERY PLAN`` reported ``CORRELATED SCALAR SUBQUERY 1``).
    On the live 79 k-row DB this drove /api/map cold latency to ~134 ms.

    The new implementation computes ``MAX(capture_time)`` per place in a CTE
    (walked by ``idx_photos_place``), then joins back to ``photos`` on
    ``(place, capture_time)`` to recover the cover uid. The cover join
    probes the new partial index ``idx_photos_done_time`` so the cover CTE
    is built once and the JOIN is O(places) rather than
    O(rows × correlated subquery). On the live prod DB this drops cold
    latency to ~24 ms (well under the 60 ms target). ``uid`` is used as a
    tiebreaker for the rare case where two photos share both ``place`` and
    ``capture_time`` (clock-skew duplicates): without it the JOIN would
    emit multiple cover rows for the same place and inflate the result
    set. Validated against the prod DB: byte-for-byte identical to the
    previous correlated-subquery formulation.
    """
    with get_conn() as conn:
        return conn.execute(
            """
            WITH places AS (
              SELECT place,
                     COUNT(*) AS photo_count,
                     AVG(gps_lat) AS lat,
                     AVG(gps_lng) AS lng
              FROM photos
              WHERE status='done'
                AND place IS NOT NULL
                AND gps_lat IS NOT NULL
                AND gps_lng IS NOT NULL
                AND thumb_path IS NOT NULL
                AND thumb_path != ''
              GROUP BY place
            ),
            newest AS (
              SELECT place, MAX(capture_time) AS cover_ts
              FROM photos
              WHERE status='done'
                AND thumb_path IS NOT NULL
                AND thumb_path != ''
                AND place IS NOT NULL
              GROUP BY place
            ),
            cover AS (
              SELECT place, uid AS cover_uid
              FROM (
                SELECT n.place, p.uid,
                       ROW_NUMBER() OVER (
                         PARTITION BY n.place ORDER BY p.uid ASC
                       ) AS rn
                FROM newest n
                JOIN photos p
                  ON p.place = n.place
                 AND p.capture_time = n.cover_ts
                 AND p.status='done'
                 AND p.thumb_path IS NOT NULL
                 AND p.thumb_path != ''
              )
              WHERE rn = 1
            )
            SELECT p.place,
                   p.photo_count,
                   p.lat,
                   p.lng,
                   c.cover_uid
            FROM places p
            LEFT JOIN cover c ON c.place = p.place
            ORDER BY p.photo_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def done_photos(limit: int = 200, offset: int = 0, before: int | None = None,
                only_favorites: bool = False, include_archived: bool = True) -> list[sqlite3.Row]:
    """Indexed photos with thumbnails, newest first.

    `before` optionally restricts to photos captured at or before the given
    epoch timestamp (used as a date-anchor jump).

    `only_favorites=True` returns only favorited photos (used by the Favorites
    view). `include_archived=False` hides archived photos from the default
    grid — they remain accessible from the Archive view.
    """
    sql = (
        "SELECT * FROM photos "
        "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
        "AND hidden = 0"
    )
    params: list = []
    if only_favorites:
        sql += " AND favorited = 1"
    if not include_archived:
        sql += " AND archived = 0"
    if before is not None:
        sql += " AND capture_time <= ?"
        params.append(before)
    sql += " ORDER BY capture_time DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    with get_conn() as conn:
        return conn.execute(sql, params).fetchall()


def archived_photos(limit: int = 200, offset: int = 0) -> list[sqlite3.Row]:
    """Photos the user has archived, newest first."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM photos "
            "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
            "AND hidden = 0 AND archived = 1 "
            "ORDER BY capture_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()


def set_favorited(uid: str, favorited: bool) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE photos SET favorited=? WHERE uid=?", (1 if favorited else 0, uid))


def set_archived(uid: str, archived: bool) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE photos SET archived=? WHERE uid=?", (1 if archived else 0, uid))


def set_hidden(uid: str, hidden: bool) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE photos SET hidden=? WHERE uid=?", (1 if hidden else 0, uid))


def get_tags(uid: str) -> list[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT tags FROM photos WHERE uid=?", (uid,)).fetchone()
    if not row or not row["tags"]:
        return []
    try:
        return list(json.loads(row["tags"]))
    except Exception:
        return []


def set_tags(uid: str, tags: list[str]) -> list[str]:
    """Replace the tag set for a photo. De-duped, lower-cased, blanks dropped."""
    clean: list[str] = []
    seen: set[str] = set()
    for t in tags:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        clean.append(t)
    with get_conn() as conn:
        conn.execute(
            "UPDATE photos SET tags=? WHERE uid=?",
            (json.dumps(clean) if clean else None, uid),
        )
    return clean


def all_tags() -> list[sqlite3.Row]:
    """Distinct user tags with counts (photos tagged with them)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT tags FROM photos WHERE status='done' AND tags IS NOT NULL AND tags != ''"
        ).fetchall()
    counts: dict[str, int] = {}
    for r in rows:
        try:
            for t in json.loads(r["tags"]):
                counts[t] = counts.get(t, 0) + 1
        except Exception:
            pass
    return [sqlite3.Row((t, n)) for t, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def photos_by_tag(tag: str, limit: int = 200, offset: int = 0) -> list[sqlite3.Row]:
    """Photos that carry the given tag (case-insensitive)."""
    needle = json.dumps([tag.lower()])[1:-1]  # exact-match the JSON substring
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM photos "
            "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
            "AND hidden = 0 AND tags LIKE ? "
            "ORDER BY capture_time DESC LIMIT ? OFFSET ?",
            (f"%{needle}%", limit, offset),
        ).fetchall()


def duplicate_groups(limit: int = 500) -> list[list[sqlite3.Row]]:
    """Photos whose Proton content-hash (sha1) appears more than once.

    Returns a list of groups (each is a list of sqlite3.Row). Groups are
    ordered by total photo count DESC so the worst offenders appear first.
    A ``hidden`` flag on individual rows lets the user dismiss a duplicate
    while keeping the original — that's why we filter ``hidden=0`` here.
    """
    with get_conn() as conn:
        groups = conn.execute(
            "SELECT sha1 FROM photos "
            "WHERE status='done' AND sha1 IS NOT NULL AND sha1 != '' "
            "GROUP BY sha1 HAVING COUNT(*) > 1 "
            "ORDER BY COUNT(*) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out: list[list[sqlite3.Row]] = []
        for g in groups:
            members = conn.execute(
                "SELECT * FROM photos "
                "WHERE sha1 = ? AND status='done' "
                "ORDER BY hidden ASC, capture_time DESC",
                (g["sha1"],),
            ).fetchall()
            out.append(members)
    return out


def person_map_markers(person_id: int, limit: int = 500) -> list[sqlite3.Row]:
    """Aggregate GPS-tagged photos for one person into clustered map markers.

    Same shape as ``map_markers`` but restricted to photos carrying faces of
    this person. Drives the per-person "where I've seen them" map.
    """
    with get_conn() as conn:
        return conn.execute(
            "SELECT photos.place, COUNT(*) AS photo_count, "
            "AVG(photos.gps_lat) AS lat, AVG(photos.gps_lng) AS lng, "
            "(SELECT uid FROM photos p2 "
            "  WHERE p2.place = photos.place AND p2.status='done' "
            "  AND p2.thumb_path IS NOT NULL AND p2.thumb_path != '' "
            "  AND EXISTS (SELECT 1 FROM faces f "
            "    WHERE f.photo_uid = p2.uid AND f.person_id = ?) "
            "  ORDER BY p2.capture_time DESC LIMIT 1) AS cover_uid "
            "FROM photos "
            "WHERE photos.status='done' AND photos.hidden = 0 "
            "AND photos.gps_lat IS NOT NULL AND photos.gps_lng IS NOT NULL "
            "AND photos.place IS NOT NULL "
            "AND EXISTS (SELECT 1 FROM faces f "
            "  WHERE f.photo_uid = photos.uid AND f.person_id = ?) "
            "GROUP BY photos.place ORDER BY photo_count DESC LIMIT ?",
            (person_id, person_id, limit),
        ).fetchall()


def memories_for_today(month: int, day: int, limit: int = 200) -> list[sqlite3.Row]:
    """Photos whose capture_time falls on (month, day) in any previous year.

    "On this day" / "memories" feature — pairs with the current calendar
    date so the UI can surface "X years ago today" thumbnails. Excludes the
    current year so the same-day photo doesn't dominate.
    """
    with get_conn() as conn:
        return conn.execute(
            "SELECT *, CAST((julianday('now') - julianday(capture_time, 'unixepoch')) AS INTEGER) AS age_days "
            "FROM photos "
            "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
            "AND hidden = 0 "
            "AND strftime('%m', capture_time, 'unixepoch') = ? "
            "AND strftime('%d', capture_time, 'unixepoch') = ? "
            "AND capture_time IS NOT NULL "
            "ORDER BY capture_time DESC LIMIT ?",
            (f"{int(month):02d}", f"{int(day):02d}", limit),
        ).fetchall()


def photo_anchors(limit: int = 500) -> list[sqlite3.Row]:
    """Group done-with-thumb photos by year-month for the date rail.

    Returns rows (ym, first_ts) where ym is 'YYYY-MM' and first_ts is the
    largest capture_time in that month (the newest photo of the month).
    """
    with get_conn() as conn:
        return conn.execute(
            "SELECT substr(date(capture_time, 'unixepoch'), 1, 7) AS ym, "
            "       MAX(capture_time) AS first_ts "
            "FROM photos "
            "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
            "  AND capture_time IS NOT NULL "
            "GROUP BY ym ORDER BY ym DESC LIMIT ?",
            (limit,),
        ).fetchall()


# --- albums ---------------------------------------------------------------

def sync_albums(albums: list[dict]) -> int:
    """Upsert album names from the bridge, then recompute local counts/covers.

    `albums` is [{uid, name}]. Covers and photo counts are derived from the
    local index (newest done-with-thumb photo per album), so no extra Proton
    downloads are needed.
    """
    now = int(time.time())
    with get_conn() as conn:
        for a in albums:
            conn.execute(
                """INSERT INTO albums (uid, name, cover_uid, photo_count, synced_at)
                   VALUES (?, ?, NULL, NULL, ?)
                   ON CONFLICT(uid) DO UPDATE SET name=excluded.name, synced_at=excluded.synced_at""",
                (a["uid"], a.get("name"), now),
            )
        # Recompute cover + count for every album from the local index.
        rows = conn.execute(
            """SELECT p.uid, p.capture_time, p.albums FROM photos p
               WHERE p.status='done' AND p.thumb_path IS NOT NULL AND p.thumb_path != ''
                 AND p.albums IS NOT NULL AND p.albums != ''"""
        ).fetchall()
        counts: dict[str, int] = {}
        covers: dict[str, tuple[int, str]] = {}
        spans: dict[str, tuple[int | None, int | None]] = {}
        for r in rows:
            try:
                uids = json.loads(r["albums"])
            except Exception:
                continue
            for u in uids:
                counts[u] = counts.get(u, 0) + 1
                # Track newest capture_time per album to pick the cover, and
                # the min/max span to order albums chronologically.
                cur = covers.get(u)
                ts = r["capture_time"] or 0
                if cur is None or ts >= cur[0]:
                    covers[u] = (ts, r["uid"])
                lo, hi = spans.get(u, (None, None))
                if ts:
                    spans[u] = (
                        ts if lo is None or ts < lo else lo,
                        ts if hi is None or ts > hi else hi,
                    )
        for u, n in counts.items():
            lo, hi = spans.get(u, (None, None))
            conn.execute(
                "UPDATE albums SET photo_count=?, cover_uid=?, start_ts=?, end_ts=? WHERE uid=?",
                (n, covers.get(u, (None, None))[1], lo, hi, u),
            )
    return len(albums)


def all_albums() -> list[sqlite3.Row]:
    """Albums ordered chronologically by their earliest photo, newest first."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM albums WHERE photo_count IS NOT NULL "
            "ORDER BY (start_ts IS NULL), start_ts DESC, name ASC"
        ).fetchall()


def album_photos(album_uid: str, limit: int = 200, offset: int = 0) -> list[sqlite3.Row]:
    """Done-with-thumb photos in an album, newest first."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM photos "
            "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
            "  AND albums LIKE ? "
            "ORDER BY capture_time DESC LIMIT ? OFFSET ?",
            (f'%"{album_uid}"%', limit, offset),
        ).fetchall()
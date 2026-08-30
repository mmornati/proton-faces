"""SQLite persistence layer for proton-faces."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    uid          TEXT PRIMARY KEY,
    name         TEXT,
    media_type   TEXT,
    capture_time INTEGER,
    sha1         TEXT,
    albums       TEXT,           -- JSON array of album node uids
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

CREATE TABLE IF NOT EXISTS people (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT,
    cover_uid TEXT,              -- representative photo uid
    created   INTEGER
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

CREATE TABLE IF NOT EXISTS clips (
    photo_uid TEXT PRIMARY KEY REFERENCES photos(uid) ON DELETE CASCADE,
    embedding BLOB               -- raw float32 (512,) CLIP embedding
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
        conn.executescript(_SCHEMA)


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
                """INSERT INTO photos (uid, name, media_type, capture_time, sha1, albums, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'new')
                   ON CONFLICT(uid) DO UPDATE SET
                       name=excluded.name,
                       media_type=excluded.media_type,
                       capture_time=excluded.capture_time,
                       sha1=excluded.sha1,
                       albums=excluded.albums,
                       status=CASE WHEN photos.status='deleted' THEN 'new' ELSE photos.status END
                """,
                (
                    r["uid"],
                    r.get("name"),
                    r.get("media_type"),
                    r.get("capture_time"),
                    r.get("sha1"),
                    json.dumps(r.get("albums", [])) if r.get("albums") else None,
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


def get_photos(status: str, limit: int = 500) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM photos WHERE status=? ORDER BY capture_time ASC LIMIT ?",
            (status, limit),
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
        pending = conn.execute("SELECT COUNT(*) FROM photos WHERE status IN ('new','downloading','processing')").fetchone()[0]
        faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        clips = conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        people = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        by_status = {
            r["status"]: r["n"]
            for r in conn.execute("SELECT status, COUNT(*) AS n FROM photos GROUP BY status")
        }
    return {
        "photos": {"total": total, "done": done, "pending": pending, "by_status": by_status},
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
        return conn.execute("SELECT id, photo_uid, embedding FROM faces").fetchall()


def faces_without_person(limit: int = 5000) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM faces WHERE person_id IS NULL ORDER BY id LIMIT ?", (limit,)
        ).fetchall()


def assign_face_person(face_id: int, person_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))


def create_person(name: str | None, cover_uid: str | None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO people (name, cover_uid, created) VALUES (?,?,?)",
            (name, cover_uid, int(time.time())),
        )
        return cur.lastrowid


def update_person_cover(person_id: int, cover_uid: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE people SET cover_uid=? WHERE id=? AND cover_uid IS NULL",
            (cover_uid, person_id),
        )


def rename_person(person_id: int, name: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE people SET name=? WHERE id=?", (name, person_id))


def all_people() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT p.id, p.name, p.cover_uid, COUNT(f.id) AS face_count,
                      COUNT(DISTINCT f.photo_uid) AS photo_count
               FROM people p LEFT JOIN faces f ON f.person_id = p.id
               GROUP BY p.id ORDER BY photo_count DESC"""
        ).fetchall()


def photos_for_person(person_id: int, limit: int = 1000) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT DISTINCT ph.* FROM faces f
               JOIN photos ph ON ph.uid = f.photo_uid
               WHERE f.person_id=? AND ph.status='done'
               ORDER BY ph.capture_time ASC LIMIT ?""",
            (person_id, limit),
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


# --- search helpers -------------------------------------------------------

def search_photos_by_place(query: str, limit: int = 200) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM photos WHERE status='done' AND place IS NOT NULL AND place LIKE ? ORDER BY capture_time DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()


def done_photos(limit: int = 200, offset: int = 0) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM photos WHERE status='done' ORDER BY capture_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
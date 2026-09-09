"""SQLite persistence layer for proton-faces."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager

import numpy as np

from config import settings
from sidecar import read_face_sidecar

# Bumped whenever `migrate()` adds one-time data backfills or creates new
# schema objects that old DBs must also gain. init_db() records this in
# `PRAGMA user_version` once migrations have run, so each backfill runs at
# most once per database.
_SCHEMA_VERSION = 1

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
    favorited    INTEGER NOT NULL DEFAULT 0,    -- legacy: starred by anyone (kept for backward compat / migration)
    archived     INTEGER NOT NULL DEFAULT 0,    -- hidden from default grids
    hidden       INTEGER NOT NULL DEFAULT 0,    -- user hid it (e.g. resolved duplicate)
    tags         TEXT,           -- JSON array of freeform user tags
    -- new|downloading|processing|done|error|deleted|pending_removal|full|fullres
    status       TEXT NOT NULL DEFAULT 'new',
    thumb_path   TEXT,           -- relative path under DATA_DIR/thumbs
    gps_lat      REAL,
    gps_lng      REAL,
    place        TEXT,           -- reverse-geocoded human place name
    processed_at INTEGER,
    error        TEXT,
    -- unix epoch when this row was confirmed deleted (NULL if never). Survives
    -- reclaim so we have a historical record.
    was_deleted_at INTEGER,
    -- times a `status='full'` row has been re-queued; capped to avoid infinite loops
    retry_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
-- Composite for the indexer poll query WHERE status=? ORDER BY capture_time
-- (downloader/worker/fullres loops run it every few seconds). Turns a scan +
-- temp sort into a single ordered range walk.
CREATE INDEX IF NOT EXISTS idx_photos_status_time ON photos(status, capture_time);
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

-- Normalized tag/album membership (issue #85). photos.tags / photos.albums
-- remain the JSON source of truth; these join tables are maintained by
-- set_tags() and upsert_photos()/sync_albums() so tag and album lookups can
-- be sargable indexed JOINs instead of LIKE scans on the whole photo set.
CREATE TABLE IF NOT EXISTS photo_tags (
    photo_uid TEXT NOT NULL REFERENCES photos(uid) ON DELETE CASCADE,
    tag       TEXT NOT NULL COLLATE NOCASE,
    PRIMARY KEY (photo_uid, tag)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_photo_tags_tag ON photo_tags(tag, photo_uid);

CREATE TABLE IF NOT EXISTS photo_albums (
    photo_uid TEXT NOT NULL REFERENCES photos(uid) ON DELETE CASCADE,
    album_uid TEXT NOT NULL,
    PRIMARY KEY (photo_uid, album_uid)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_photo_albums_album ON photo_albums(album_uid, photo_uid);

-- Expression indexes (issue #85). These match the exact expressions used by
-- memories_for_today() (strftime '%m-%d' equality is sargable) and
-- photo_anchors() (GROUP BY ym + MAX(capture_time) is an index-only scan).
-- Partial to the done+thumb rows those hot paths query.
CREATE INDEX IF NOT EXISTS idx_photos_month_day
  ON photos(strftime('%m-%d', capture_time, 'unixepoch'))
  WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '';
CREATE INDEX IF NOT EXISTS idx_photos_ym
  ON photos(substr(date(capture_time, 'unixepoch'), 1, 7), capture_time)
  WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '';

CREATE TABLE IF NOT EXISTS people (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT,
    cover_uid     TEXT,          -- representative photo uid
    cover_face_id INTEGER,       -- representative face id (for face-crop covers)
    created       INTEGER,
    face_count    INTEGER NOT NULL DEFAULT 0,   -- denormalized: COUNT(faces.person_id)
    photo_count   INTEGER NOT NULL DEFAULT 0    -- denormalized: COUNT(DISTINCT photo_uid)
);
CREATE INDEX IF NOT EXISTS idx_people_photo_count ON people(photo_count DESC);

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

-- --- multi-user auth + per-user favorites ----------------------------------
-- Added in PR-9: local family accounts (no public Proton OAuth exists).
-- Each user has a bcrypt password hash and a role. Active bearer tokens are
-- tracked here too so the FastAPI app can validate them without consulting
-- the bridge. ON DELETE CASCADE drops tokens + favorites when a user is
-- removed so a family member leaving the household leaves no trace.

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name  TEXT,
    password_hash TEXT NOT NULL,                   -- bcrypt cost 12, never plaintext
    role          TEXT NOT NULL DEFAULT 'read'
                  CHECK (role IN ('read','write','admin')),
    created_at    INTEGER NOT NULL,
    last_login_at INTEGER,
    disabled      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS auth_tokens (
    token       TEXT PRIMARY KEY,                  -- sha256(32 random bytes hex)
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('access','refresh')),
    expires_at  INTEGER NOT NULL,
    created_at  INTEGER NOT NULL,
    user_agent  TEXT,
    ip          TEXT
);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_exp  ON auth_tokens(expires_at);

CREATE TABLE IF NOT EXISTS user_favorites (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    photo_uid  TEXT    NOT NULL REFERENCES photos(uid) ON DELETE CASCADE,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, photo_uid)
);
CREATE INDEX IF NOT EXISTS idx_user_favorites_photo ON user_favorites(photo_uid);
"""

_lock = threading.Lock()

# Thread-local persistent connections: each thread gets one connection per db
# path and keeps it for the lifetime of the thread.  This avoids the
# connect/PRAGMA/close cycle on every call — the ~2 MB page cache survives
# across queries, which matters a lot on HDD-backed indexes.
_local = threading.local()


def _close_local_conns() -> None:
    """Close all connections held in the current thread's local storage."""
    conns = getattr(_local, "_conns", None)
    if conns is None:
        return
    for conn in conns.values():
        try:
            conn.close()
        except Exception:
            pass
    _local._conns = {}


def _get_persistent_conn(db_path: str, timeout: int = 30) -> sqlite3.Connection:
    """Return a persistent thread-local connection for *db_path*.

    PRAGMAs are run once at connect time.  The connection is never closed by
    the caller — it stays open for the lifetime of the thread.
    """
    conns: dict[str, sqlite3.Connection] = getattr(_local, "_conns", None)
    if conns is None:
        conns = {}
        _local._conns = conns
    conn = conns.get(db_path)
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # The index lives on a spinning HDD: synchronous=FULL makes every
        # commit fsync the disk and the small default WAL checkpoint (4 MB)
        # stalls readers while the indexer is draining. NORMAL keeps WAL
        # durability for app crashes (only an OS power-loss can lose the last
        # commits, which is acceptable for a rebuildable index) and a 64 MB
        # checkpoint amortizes the checkpoint across many more writes.
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA wal_autocheckpoint=16000")
        # Cache / performance pragmas — run once per connection lifetime.
        conn.execute("PRAGMA cache_size=-64000")       # 64 MB page cache
        conn.execute("PRAGMA mmap_size=268435456")     # 256 MB mmap
        conn.execute("PRAGMA temp_store=MEMORY")
        conns[db_path] = conn
    return conn


@contextmanager
def get_conn() -> sqlite3.Connection:
    conn = _get_persistent_conn(str(settings.db_path))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class TransactionRollback(Exception):
    """Raise inside a `with transaction():` block to roll it back without error.

    Used to abort a batch of writes (e.g. a lost photo claim) without turning
    a normal no-op into a failure: the block is rolled back and the exception
    is swallowed.
    """


@contextmanager
def transaction() -> sqlite3.Connection:
    """Run several writes on the caller's persistent connection as one commit.

    The indexer uses this to collapse ~6 separate commits per photo (claim,
    face inserts, clip insert, done) into a single fsync-heavy commit — the
    dominant per-photo cost on HDD-backed indexes. Any exception rolls the
    whole batch back, leaving the prior committed state (an uncommitted photo
    simply keeps its old status and is retried — same crash semantics as the
    per-step commits it replaces).

    Store helpers called inside **must** receive ``conn=`` so they execute on
    this transaction instead of committing on their own connection.
    """
    conn = _get_persistent_conn(str(settings.db_path))
    conn.execute("BEGIN")
    try:
        yield conn
    except TransactionRollback:
        conn.rollback()
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


@contextmanager
def _with_conn(conn: sqlite3.Connection | None) -> sqlite3.Connection:
    """Yield *conn* when given (the caller owns the transaction), else open a
    commit-on-exit connection. Lets store helpers accept an optional
    connection while keeping their single-call behaviour unchanged."""
    if conn is not None:
        yield conn
    else:
        with get_conn() as c:
            yield c


def init_db() -> None:
    with get_conn() as conn:
        # Migrations first: existing DBs need ALTER TABLE ADD COLUMN before the
        # _SCHEMA block runs CREATE INDEX against those new columns.
        migrate(conn)
        conn.executescript(_SCHEMA)
        # After _SCHEMA ran, finish any version-gated backfills that need the
        # new tables to exist (CREATE TABLE IF NOT EXISTS is idempotent).
        _backfill_issue85(conn)
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def _backfill_issue85(conn: sqlite3.Connection) -> None:
    """One-time backfill of photo_tags / photo_albums from the JSON columns.

    Runs only when the database predates issue #85's join tables (detected via
    the join table + a source photo still carrying JSON that isn't yet
    mirrored). The write paths keep the tables in sync from here on.
    """
    if conn.execute("PRAGMA user_version").fetchone()[0] >= _SCHEMA_VERSION:
        return
    if "photo_tags" not in {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}:
        return  # fresh DB: _SCHEMA just created the (empty) tables
    rows = conn.execute("SELECT uid, tags, albums FROM photos").fetchall()
    for uid, tags, albums in rows:
        try:
            tag_list = json.loads(tags) if tags else []
        except (ValueError, TypeError):
            tag_list = []
        try:
            album_list = json.loads(albums) if albums else []
        except (ValueError, TypeError):
            album_list = []
        if tag_list:
            conn.executemany(
                "INSERT OR IGNORE INTO photo_tags (photo_uid, tag) VALUES (?, ?)",
                [(uid, t) for t in tag_list],
            )
        if album_list:
            conn.executemany(
                "INSERT OR IGNORE INTO photo_albums (photo_uid, album_uid) VALUES (?, ?)",
                [(uid, a) for a in album_list],
            )


def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column migrations for older databases.

    Safe to run before `_SCHEMA` is applied (init_db does both): if the
    `photos` table doesn't exist yet, there's nothing to migrate.
    """
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if "photos" not in tables:
        return
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
    if "was_deleted_at" not in pcols:
        conn.execute("ALTER TABLE photos ADD COLUMN was_deleted_at INTEGER")
    if "retry_count" not in pcols:
        conn.execute("ALTER TABLE photos ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
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
    if "idx_photos_status_pending" not in idx:
        # Helps the grace-period sweep: list rows stuck in pending_removal
        # long enough to be confirmed-deleted.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_photos_status_pending "
            "ON photos(status, processed_at) "
            "WHERE status='pending_removal'"
        )
    if "idx_photos_status_time" not in idx:
        # Composite for the indexer poll query (WHERE status=? ORDER BY
        # capture_time). One-time CREATE INDEX, safe on existing DBs.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_photos_status_time "
            "ON photos(status, capture_time)"
        )
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
    # Denormalized face/photo counts (issue #81). One-time backfill; idempotent
    # because we only recount people whose cached counts are 0 while they still
    # own faces, plus people who own faces but have a 0 count. The columns
    # default to 0 on new rows, so this only fixes pre-existing data.
    if "face_count" not in cols:
        conn.execute("ALTER TABLE people ADD COLUMN face_count INTEGER NOT NULL DEFAULT 0")
    if "photo_count" not in cols:
        conn.execute("ALTER TABLE people ADD COLUMN photo_count INTEGER NOT NULL DEFAULT 0")
    if "idx_people_photo_count" not in {r["name"] for r in conn.execute("PRAGMA index_list(people)")}:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_photo_count ON people(photo_count DESC)")
    conn.execute(
        """UPDATE people
           SET face_count = (SELECT COUNT(*) FROM faces f WHERE f.person_id = people.id),
               photo_count = (SELECT COUNT(DISTINCT f.photo_uid) FROM faces f
                              WHERE f.person_id = people.id AND f.photo_uid IS NOT NULL)
           WHERE face_count = 0
              OR photo_count = 0"""
    )
    acols = {r["name"] for r in conn.execute("PRAGMA table_info(albums)")}
    if "start_ts" not in acols:
        conn.execute("ALTER TABLE albums ADD COLUMN start_ts INTEGER")
    if "end_ts" not in acols:
        conn.execute("ALTER TABLE albums ADD COLUMN end_ts INTEGER")
    # Migration for issue #34: clear all plaintext auth_tokens so they are
    # re-issued as SHA-256 hashes on next login. Safe to run repeatedly.
    if "auth_tokens" in tables:
        conn.execute("DELETE FROM auth_tokens")


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
                       -- Reclaim: if the row was marked deleted or pending_removal
                       -- (i.e. missing from the timeline for >= grace cycles),
                       -- bring it back to 'new' so the indexer reprocesses it.
                       -- Clear was_deleted_at so the historical record reflects
                       -- the *current* state (i.e. no longer deleted).
                       status=CASE WHEN photos.status IN ('deleted','pending_removal')
                                   THEN 'new' ELSE photos.status END,
                       was_deleted_at=CASE WHEN photos.status IN ('deleted','pending_removal')
                                           THEN NULL ELSE photos.was_deleted_at END,
                       retry_count=CASE WHEN photos.status IN ('deleted','pending_removal')
                                        THEN 0 ELSE photos.retry_count END
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
            # Mirror album membership into the normalized join table so
            # album_photos() can use a sargable indexed JOIN (issue #85).
            membership = r.get("albums") or []
            conn.execute("DELETE FROM photo_albums WHERE photo_uid=?", (r["uid"],))
            if membership:
                conn.executemany(
                    "INSERT OR IGNORE INTO photo_albums (photo_uid, album_uid) VALUES (?, ?)",
                    [(r["uid"], a) for a in membership],
                )
    return new


def mark_pending_removal(uids: list[str]) -> int:
    """Stage uids for deletion. The row becomes status='pending_removal' until
    `confirm_deletions(grace_cycles=...)` is called; that helper promotes any
    pending_removal rows whose processed_at is older than the grace window to
    status='deleted' with was_deleted_at populated.

    Idempotent: rows already in 'pending_removal' are left untouched
    (preserving their original processed_at as the start of the grace timer),
    rows already in 'deleted' are left untouched (preserve historical record),
    rows in any other status get moved to 'pending_removal'.

    Returns number of rows newly transitioned to pending_removal.
    """
    if not uids:
        return 0
    now = int(time.time())
    placeholders = ",".join("?" * len(uids))
    with get_conn() as conn:
        cur = conn.execute(
            f"""UPDATE photos
                   SET status='pending_removal',
                       processed_at=CASE WHEN status='pending_removal' THEN processed_at ELSE ? END
                 WHERE uid IN ({placeholders})
                   AND status NOT IN ('deleted','pending_removal')""",
            (now, *uids),
        )
        return cur.rowcount


def confirm_deletions(grace_seconds: int, now: int | None = None) -> int:
    """Promote pending_removal rows to deleted once the grace window has passed.

    A row that has been in pending_removal for >= grace_seconds is considered
    truly gone from Proton and is moved to 'deleted' with was_deleted_at set.

    Returns number of rows confirmed.
    """
    now = int(time.time()) if now is None else now
    with get_conn() as conn:
        cur = conn.execute(
            """UPDATE photos
                  SET status='deleted',
                      was_deleted_at=COALESCE(was_deleted_at, ?)
                WHERE status='pending_removal'
                  AND processed_at IS NOT NULL
                  AND processed_at <= ?""",
            (now, now - grace_seconds),
        )
        return cur.rowcount


def reset_stuck_fullres(retry_after_sec: int, max_retry_count: int = 5, now: int | None = None) -> int:
    """Re-queue `status='full'` rows whose processed_at is older than
    retry_after_sec. Bumps retry_count so we never spin forever on a poison
    row — once retry_count >= max_retry_count the row is parked in
    `status='error'` with a descriptive error string.

    Returns (requeued, parked) tuple.
    """
    now = int(time.time()) if now is None else now
    requeued = 0
    parked = 0
    with get_conn() as conn:
        cur = conn.execute(
            """UPDATE photos
                  SET status='new',
                      retry_count=retry_count+1
                WHERE status='full'
                  AND processed_at IS NOT NULL
                  AND processed_at <= ?
                  AND retry_count < ?""",
            (now - retry_after_sec, max_retry_count),
        )
        requeued = cur.rowcount
        cur = conn.execute(
            """UPDATE photos
                  SET status='error',
                      error='stuck in full for > max_retry_count cycles; not retrying',
                      retry_count=retry_count+1
                WHERE status='full'
                  AND processed_at IS NOT NULL
                  AND processed_at <= ?
                  AND retry_count >= ?""",
            (now - retry_after_sec, max_retry_count),
        )
        parked = cur.rowcount
    return requeued, parked


def mark_deleted(uids: list[str]) -> None:
    """Hard-delete (kept for tests and explicit use). Production code should
    use mark_pending_removal + confirm_deletions for the grace-period path."""
    if not uids:
        return
    now = int(time.time())
    with get_conn() as conn:
        conn.execute(
            f"""UPDATE photos
                   SET status='deleted',
                       was_deleted_at=COALESCE(was_deleted_at, ?)
                 WHERE uid IN ({",".join("?" * len(uids))})
                   AND status!='deleted'""",
            (now, *uids),
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


def claim_photos_for_download(uids: list[str]) -> list[str]:
    """Atomically claim a batch of 'new' photos for download.

    Uses a single UPDATE ... RETURNING so one connection/commit replaces N
    per-uid claims. Only the uids that were actually in status 'new' are
    returned (the rest are left for the next cycle).
    """
    if not uids:
        return []
    with _lock, get_conn() as conn:
        cur = conn.execute(
            f"""UPDATE photos
                   SET status='downloading'
                 WHERE uid IN ({",".join("?" * len(uids))})
                   AND status='new'
                 RETURNING uid""",
            uids,
        )
        return [row[0] for row in cur.fetchall()]


def claim_photo_for_processing(uid: str, conn: sqlite3.Connection | None = None) -> bool:
    """Atomically move a photo from 'downloading' to 'processing'.

    When *conn* is given the claim runs inside that caller-owned transaction
    (the whole per-photo sequence then commits as one). Otherwise it makes its
    own quick commit, serialized against other claims by `_lock`.
    """
    if conn is not None:
        cur = conn.execute(
            "UPDATE photos SET status='processing' WHERE uid=? AND status='downloading'", (uid,)
        )
        return cur.rowcount == 1
    with _lock, get_conn() as conn:
        cur = conn.execute(
            "UPDATE photos SET status='processing' WHERE uid=? AND status='downloading'", (uid,)
        )
        return cur.rowcount == 1


def set_photo_done(
    uid: str,
    thumb_path: str,
    gps: tuple[float, float] | None,
    place: str | None,
    conn: sqlite3.Connection | None = None,
) -> None:
    with _with_conn(conn) as c:
        c.execute(
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


def set_photo_error(uid: str, error: str, conn: sqlite3.Connection | None = None) -> None:
    with _with_conn(conn) as c:
        c.execute("UPDATE photos SET status='error', error=? WHERE uid=?", (error, uid))


def set_photo_deleted(uid: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE photos SET status='deleted' WHERE uid=?", (uid,))


def get_photo(uid: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM photos WHERE uid=?", (uid,)).fetchone()


def get_photos_batch(uids: list[str]) -> dict[str, sqlite3.Row]:
    """Fetch multiple photos by uid in a single query.

    Returns a dict keyed by uid (only uids that exist in the DB).
    """
    if not uids:
        return {}
    placeholders = ",".join("?" * len(uids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM photos WHERE uid IN ({placeholders})", uids
        ).fetchall()
    return {r["uid"]: r for r in rows}


def stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        done = conn.execute("SELECT COUNT(*) FROM photos WHERE status='done'").fetchone()[0]
        done_with_thumb = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != ''"
        ).fetchone()[0]
        done_without_thumb = done - done_with_thumb
        done_without_thumb_videos = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE status='done' "
            "AND (thumb_path IS NULL OR thumb_path='') "
            "AND media_type LIKE 'video/%'"
        ).fetchone()[0]
        done_without_thumb_other = done_without_thumb - done_without_thumb_videos
        # Real denominator: photos actually in the library (excludes 'deleted',
        # which are kept only as historical record). `orphan` surfaces them
        # separately so the UI can show "X indexed, Y orphan" honestly.
        active_total = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE status != 'deleted'"
        ).fetchone()[0]
        # Pending covers all in-flight statuses including the new
        # pending_removal grace-period state so the footer reflects current
        # work, not the misleading '0' from the old counter.
        pending = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE status IN "
            "('new','downloading','processing','pending_removal')"
        ).fetchone()[0]
        # Videos (and other media) currently stuck waiting for full-res /
        # poster extraction. Separate from `pending` because they take much
        # longer and deserve their own visibility.
        stuck_fullres = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE status='full' "
            "AND media_type LIKE 'video/%'"
        ).fetchone()[0]
        orphan = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE status='deleted'"
        ).fetchone()[0]
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
            "active_total": active_total,
            "done": done,
            "done_with_thumb": done_with_thumb,
            "done_without_thumb": done_without_thumb,
            "done_without_thumb_videos": done_without_thumb_videos,
            "done_without_thumb_other": done_without_thumb_other,
            "pending": pending,
            "stuck_fullres": stuck_fullres,
            "orphan": orphan,
            "by_status": by_status,
        },
        "faces": faces,
        "clips": clips,
        "people": people,
    }


# --- faces & people -------------------------------------------------------

def _recount_person(person_id: int, conn: sqlite3.Connection | None = None) -> None:
    """Refresh the denormalized face_count / photo_count for one person.

    Cheap per-person correlated recount, used at every face<->person write
    site so `all_people()` and `get_person()` can read cached counts instead
    of running an expensive join + COUNT(DISTINCT) on every request.

    Passing *conn* keeps the recount inside an enclosing transaction.
    """
    with _with_conn(conn) as c:
        c.execute(
            """UPDATE people
               SET face_count = (SELECT COUNT(*) FROM faces f WHERE f.person_id = people.id),
                   photo_count = (SELECT COUNT(DISTINCT f.photo_uid) FROM faces f
                                  WHERE f.person_id = people.id AND f.photo_uid IS NOT NULL)
               WHERE id = ?""",
            (person_id,),
        )


def _gc_unnamed_person(person_id: int) -> bool:
    """Delete an anonymous, face-less placeholder person row if one exists.

    When a face is unassigned or moved to another person and the source ends up
    with no remaining faces, the now-empty unnamed row would otherwise linger
    forever (named people and any person still owning a face row are untouched).
    Returns True when the row was deleted.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """DELETE FROM people
               WHERE id = ?
                 AND (name IS NULL OR name = '')
                 AND face_count = 0
                 AND NOT EXISTS (SELECT 1 FROM faces WHERE person_id = ?)""",
            (person_id, person_id),
        )
        return cur.rowcount > 0


def delete_empty_people() -> int:
    """Sweep every anonymous, face-less placeholder person row.

    The targeted `_gc_unnamed_person` keeps write paths clean, but rows orphaned
    before that guard existed (e.g. faces bulk-deleted outside the current code
    paths) accumulate in bulk here. The single DELETE re-checks the full
    predicate atomically so a face assigned concurrently is never swept; the
    age guard keeps an in-flight clusterer create/assign from being caught in
    between. Idempotent; returns how many rows were deleted.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """DELETE FROM people
               WHERE (name IS NULL OR name = '')
                 AND face_count = 0
                 AND created < ?
                 AND NOT EXISTS (SELECT 1 FROM faces WHERE person_id = people.id)""",
            (int(time.time()) - 3600,),
        )
        return cur.rowcount


def insert_face(
    photo_uid: str,
    person_id: int | None,
    confidence: float,
    bbox: list,
    embedding: bytes,
    conn: sqlite3.Connection | None = None,
) -> int:
    with _with_conn(conn) as c:
        cur = c.execute(
            "INSERT INTO faces (photo_uid, person_id, confidence, bbox, embedding) VALUES (?,?,?,?,?)",
            (photo_uid, person_id, confidence, json.dumps(bbox), sqlite3.Binary(embedding)),
        )
        fid = cur.lastrowid
    if person_id is not None:
        _recount_person(person_id, conn)
    return fid


def count_faces_for_photo(photo_uid: str, conn: sqlite3.Connection | None = None) -> int:
    """How many face rows a photo already has.

    Used by the worker to skip re-running face detection when a photo is
    reprocessed (e.g. a reclaimed photo restored after a false deletion).
    """
    with _with_conn(conn) as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM faces WHERE photo_uid=?", (photo_uid,)
        ).fetchone()
        return int(row["n"])


def clip_exists(photo_uid: str, conn: sqlite3.Connection | None = None) -> bool:
    """Whether a photo already has a CLIP embedding (to skip recomputing it)."""
    with _with_conn(conn) as c:
        row = c.execute(
            "SELECT 1 FROM clips WHERE photo_uid=? LIMIT 1", (photo_uid,)
        ).fetchone()
        return row is not None


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
        row = conn.execute("SELECT person_id FROM faces WHERE id=?", (face_id,)).fetchone()
        old = row["person_id"] if row else None
        conn.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
    if old is not None and old != person_id:
        _recount_person(old)
        _gc_unnamed_person(old)
    _recount_person(person_id)


def assign_faces_person_bulk(face_ids: list[int], person_id: int) -> None:
    """Assign many faces to `person_id` in a single transaction.

    Chunked so each generated `UPDATE ... IN (...)` stays under SQLite's 999
    placeholder limit. Runs on one connection/commit instead of one per face.
    """
    for start in range(0, len(face_ids), _SQL_CHUNK):
        chunk = face_ids[start : start + _SQL_CHUNK]
        qmarks = ",".join("?" * len(chunk))
        with get_conn() as conn:
            conn.execute(
                "UPDATE faces SET person_id=? WHERE id IN (" + qmarks + ")",
                (person_id, *chunk),
            )
    if face_ids:
        _recount_person(person_id)


def unassign_face(face_id: int) -> None:
    with get_conn() as conn:
        row = conn.execute("SELECT person_id FROM faces WHERE id=?", (face_id,)).fetchone()
        old = row["person_id"] if row else None
        conn.execute("UPDATE faces SET person_id=NULL WHERE id=?", (face_id,))
    if old is not None:
        _recount_person(old)
        _gc_unnamed_person(old)


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
        return conn.execute("SELECT p.* FROM people p WHERE p.id=?", (person_id,)).fetchone()


def people_by_ids(person_ids: list[int]) -> list[sqlite3.Row]:
    """Batch variant of `get_person`: people + face/photo counts for many ids.



    Order is unspecified; callers should map by id. Returns [] when empty.


    """
    if not person_ids:
        return []
    qmarks = ",".join("?" * len(person_ids))
    with get_conn() as conn:
        return conn.execute(
            f"SELECT p.* FROM people p WHERE p.id IN ({qmarks})",
            person_ids,
        ).fetchall()


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


# Module-level cache of every face embedding as one (N,512) float32 matrix.
# The indexer adds faces continuously, so the cache is short-lived. TTL is
# sized so the ~460 MB matrix / ~13 s load is amortized across requests (the
# interactive face-suggest and face-search endpoints reuse it), while still
# refreshing often enough for the background auto-tag path to see new faces
# within a couple of minutes.
_EMBEDDING_CACHE_TTL = 120.0
_embedding_cache: dict | None = None
_embedding_cache_ts = 0.0
_embedding_cache_lock = threading.Lock()
_embedding_cache_refreshing = False


def _build_embedding_cache() -> dict:
    """Build the face-embedding matrix from the mmap sidecar or the DB.

    Prefers the mmap sidecar written by the indexer; falls back to the
    DB-based cache when sidecar files are absent (first run before indexer
    upgrade). Runs without holding `_embedding_cache_lock` so concurrent
    callers keep serving the stale cache while a refresh builds.
    """
    sidecar = read_face_sidecar()
    if sidecar is not None:
        return sidecar
    ids: list[int] = []
    photo_uids: list[str] = []
    person_ids: list[int | None] = []
    vecs: list[np.ndarray] = []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, photo_uid, person_id, embedding FROM faces ORDER BY id"
        ).fetchall()
    for r in rows:
        ids.append(r["id"])
        photo_uids.append(r["photo_uid"])
        person_ids.append(r["person_id"])
        vecs.append(np.frombuffer(r["embedding"], dtype=np.float32))
    mat = np.stack(vecs) if vecs else np.zeros((0, 512), dtype=np.float32)
    return {
        "ids": np.asarray(ids, dtype=np.int64),
        "photo_uids": photo_uids,
        "person_ids": person_ids,
        "mat": mat,
    }


def _embedding_cache_data() -> dict:
    """Face-embedding matrix with stale-while-revalidate semantics.

    Returns the cached matrix immediately when fresh. On expiry it serves the
    stale matrix and, guarded against duplication, starts a background thread
    that rebuilds and swaps it — so a ~13 s reload never blocks callers. The
    very first load (empty cache) is synchronous because there is no stale
    copy to serve. The lock is held only for the pointer/flag swap, not for
    the build.
    """
    global _embedding_cache, _embedding_cache_ts, _embedding_cache_refreshing
    now = time.time()
    if _embedding_cache is not None and now - _embedding_cache_ts < _EMBEDDING_CACHE_TTL:
        return _embedding_cache
    if _embedding_cache is None:
        # First-ever load: no stale copy to serve yet, build synchronously.
        with _embedding_cache_lock:
            now = time.time()
            if _embedding_cache is not None and now - _embedding_cache_ts < _EMBEDDING_CACHE_TTL:
                return _embedding_cache
            if _embedding_cache is None:
                _embedding_cache = _build_embedding_cache()
                _embedding_cache_ts = time.time()
        return _embedding_cache
    # Expired but a stale copy exists: return it and kick off one background
    # refresh. The refreshing flag (set/cleared under the short lock) prevents
    # a thundering herd of concurrent reloads.
    with _embedding_cache_lock:
        now = time.time()
        if _embedding_cache is not None and now - _embedding_cache_ts < _EMBEDDING_CACHE_TTL:
            return _embedding_cache
        if _embedding_cache_refreshing:
            return _embedding_cache
        _embedding_cache_refreshing = True
    threading.Thread(target=_background_refresh_embedding, daemon=True).start()
    return _embedding_cache


def _background_refresh_embedding() -> None:
    """Rebuild the embedding cache off the caller's thread and swap it in."""
    global _embedding_cache, _embedding_cache_ts, _embedding_cache_refreshing
    try:
        built = _build_embedding_cache()
        with _embedding_cache_lock:
            _embedding_cache = built
            _embedding_cache_ts = time.time()
    finally:
        _embedding_cache_refreshing = False


def similar_faces(embedding: bytes, threshold: float, limit: int = 200) -> list[sqlite3.Row]:
    """Faces (id, photo_uid, person_id, sim) whose cosine similarity to `embedding` is >= threshold."""
    emb = np.frombuffer(embedding, dtype=np.float32)
    data = _embedding_cache_data()
    mat = data["mat"]
    if mat.shape[0] == 0:
        return []
    scores = mat @ emb
    idx = np.flatnonzero(scores >= threshold)
    if idx.size == 0:
        return []
    top = idx[np.argsort(scores[idx])[::-1][:limit]]
    ids = data["ids"]
    uids = data["photo_uids"]
    pids = data["person_ids"]
    return [(int(ids[i]), uids[i], pids[i], float(scores[i])) for i in top]


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
    _recount_person(target_id)


# SQLite caps parameter placeholders at 999; keep the chunk small enough that
# every generated query (UPDATE has 1 extra placeholder) stays well below it.
_SQL_CHUNK = 500


def merge_people_bulk(source_ids: list[int], target_id: int) -> int:
    """Merge many people into `target_id`, committing once per chunk.

    Mirrors `merge_person` per source: backfill the target's name/cover from
    the sources when unset, re-parent every face onto the target, then delete
    the source rows. Unlike calling `merge_person` N times (one connection +
    commit each), each ~500-id chunk of sources runs its whole merge on a
    single connection and commits once, so a 49k-people dedupe campaign is a
    few dozen commits, not one per source.

    Each source-id merge is independently valid, so committing per chunk means
    the WAL writer lock is held only for the duration of one chunk instead of
    the whole campaign — indexer claims and API writes stay unblocked during
    large merges, and a failure (or a concurrent deletion of the target)
    leaves the already-merged chunks persisted.

    Missing / repeated ids and the target itself are skipped. Returns how many
    source people were actually merged (existing rows deleted).
    """
    seen: set[int] = set()
    ids: list[int] = []
    for sid in source_ids:
        if sid == target_id or sid in seen:
            continue
        seen.add(sid)
        ids.append(sid)
    if not ids:
        return 0
    merged = 0
    for start in range(0, len(ids), _SQL_CHUNK):
        chunk = ids[start : start + _SQL_CHUNK]
        qmarks = ",".join("?" * len(chunk))
        with get_conn() as conn:
            # Re-read the target per chunk so the backfill sees the state
            # committed by the previous chunk (and any concurrent writer).
            tgt = conn.execute(
                "SELECT name, cover_uid, cover_face_id FROM people WHERE id=?", (target_id,)
            ).fetchone()
            if tgt is None:
                return merged
            t_name, t_cover_uid, t_cover_face_id = tgt["name"], tgt["cover_uid"], tgt["cover_face_id"]
            for src in conn.execute(
                "SELECT name, cover_uid, cover_face_id FROM people "
                f"WHERE id IN ({qmarks})",
                chunk,
            ).fetchall():
                if not t_name and src["name"]:
                    t_name = src["name"]
                    conn.execute("UPDATE people SET name=? WHERE id=?", (t_name, target_id))
                if not t_cover_uid and src["cover_uid"]:
                    t_cover_uid = src["cover_uid"]
                    conn.execute("UPDATE people SET cover_uid=? WHERE id=?", (t_cover_uid, target_id))
                if not t_cover_face_id and src["cover_face_id"]:
                    t_cover_face_id = src["cover_face_id"]
                    conn.execute(
                        "UPDATE people SET cover_face_id=? WHERE id=?",
                        (t_cover_face_id, target_id),
                    )
                merged += 1
            conn.execute(
                "UPDATE faces SET person_id=? WHERE person_id IN (" + qmarks + ")",
                (target_id, *chunk),
            )
            conn.execute("DELETE FROM people WHERE id IN (" + qmarks + ")", chunk)
    _recount_person(target_id)
    return merged


def face_ids_for_people(person_ids: list[int]) -> list[int]:
    """All face ids belonging to any of `person_ids`, in one chunked query.

    Used to bulk-drop crop-cache files for many people at once instead of one
    connection per person. Returns [] when empty.
    """
    if not person_ids:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for start in range(0, len(person_ids), _SQL_CHUNK):
        chunk = person_ids[start : start + _SQL_CHUNK]
        qmarks = ",".join("?" * len(chunk))
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT id FROM faces WHERE person_id IN (" + qmarks + ")", chunk
            ).fetchall()
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r["id"])
    return out


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

_person_means_cache: dict[int, np.ndarray] | None = None
_person_means_cache_ts = 0.0


def person_mean_embeddings_from_cache() -> dict[int, np.ndarray]:
    """Person-mean embeddings derived from the shared cached face matrix.

    Unlike `person_mean_embeddings()` (which issues a fresh ~180 k-row SQLite
    fetch on every call), this reuses `_embedding_cache_data()` so the
    expensive load is amortized across requests within the cache TTL. The
    means are computed vectorized via ``np.add.reduceat`` over the sorted
    person-id axis. Used by the interactive "top matches" suggest endpoint.


    The computed means are themselves cached, keyed to the embedding-matrix
    load timestamp: recomputed only when the underlying matrix reloads
    (every ~120 s), NOT on every suggest request.

    """
    global _person_means_cache, _person_means_cache_ts
    data = _embedding_cache_data()
    mat = data["mat"]
    person_ids = data["person_ids"]
    if _person_means_cache is not None and _person_means_cache_ts == _embedding_cache_ts:
        return _person_means_cache
    if mat.shape[0] == 0 or not person_ids:
        _person_means_cache = {}
        _person_means_cache_ts = _embedding_cache_ts
        return _person_means_cache
    mask = np.array([p is not None for p in person_ids], dtype=bool)
    if not mask.any():
        _person_means_cache = {}
        _person_means_cache_ts = _embedding_cache_ts
        return _person_means_cache
    sub = mat[mask]
    spids = np.array([p for p in person_ids if p is not None], dtype=np.int64)
    order = np.argsort(spids, kind="stable")
    sorted_pids = spids[order]
    sorted_mat = sub[order]
    unique, first_idx = np.unique(sorted_pids, return_index=True)
    sums = np.add.reduceat(sorted_mat, first_idx, axis=0)
    counts = np.diff(np.append(first_idx, len(sorted_pids)))[:, None]
    means = sums / counts
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    out: dict[int, np.ndarray] = {}
    for i, u in enumerate(unique):
        n = norms[i, 0]
        if n != 0:
            out[int(u)] = (means[i] / n).astype(np.float32)
    _person_means_cache = out
    _person_means_cache_ts = _embedding_cache_ts
    return out


def all_people(q: str | None = None, limit: int | None = None, offset: int = 0) -> list[sqlite3.Row]:
    """People ordered by photo_count DESC.

    With `q`, restricts to people whose name matches the LIKE pattern
    (case-insensitive prefix is encouraged). With `limit`/`offset`, paginates.
    """
    sql = (
        "SELECT p.id, p.name, p.cover_uid, p.cover_face_id, "
        "       p.face_count, p.photo_count "
        "FROM people p "
    )
    params: list = []
    if q:
        sql += "WHERE LOWER(p.name) LIKE LOWER(?) "
        params.append(f"%{q}%")
    sql += "ORDER BY p.photo_count DESC, p.id ASC"
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


def faces_for_person(person_id: int, limit: int = 500) -> list[sqlite3.Row]:
    """Every face belonging to a person (id, photo_uid, confidence, bbox).

    Used by the cover picker so a person's cover photo can be chosen from
    any of their detected faces. bbox is returned as the raw JSON string;
    callers deserialize it with json.loads.
    """
    with get_conn() as conn:
        return conn.execute(
            """SELECT f.id, f.photo_uid, f.confidence, f.bbox
               FROM faces f
               WHERE f.person_id=? AND f.photo_uid IS NOT NULL
               ORDER BY f.confidence DESC, f.id ASC
               LIMIT ?""",
            (person_id, limit),
        ).fetchall()


# --- clips ----------------------------------------------------------------

def insert_clip(photo_uid: str, embedding: bytes, conn: sqlite3.Connection | None = None) -> None:
    with _with_conn(conn) as c:
        c.execute(
            "INSERT INTO clips (photo_uid, embedding) VALUES (?,?) "
            "ON CONFLICT(photo_uid) DO UPDATE SET embedding=excluded.embedding",
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
            "SELECT * FROM photos WHERE status='done' AND thumb_path IS NOT NULL "
            "AND thumb_path != '' AND place IS NOT NULL AND place LIKE ? "
            "ORDER BY capture_time DESC LIMIT ? OFFSET ?",
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
                only_favorites: bool = False, include_archived: bool = True,
                user_id: int | None = None) -> list[sqlite3.Row]:
    """Indexed photos with thumbnails, newest first.

    `before` optionally restricts to photos captured at or before the given
    epoch timestamp (used as a date-anchor jump).

    `only_favorites=True` returns only photos the given user has favorited
    (used by the Favorites view). `include_archived=False` hides archived
    photos from the default grid — they remain accessible from the Archive view.
    """
    sql = (
        "SELECT * FROM photos "
        "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
        "AND hidden = 0"
    )
    params: list = []
    if only_favorites:
        if user_id is None:
            return []
        sql += " AND uid IN (SELECT photo_uid FROM user_favorites WHERE user_id=?)"
        params.append(user_id)
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
        # Keep the normalized join table in sync so photos_by_tag() stays a
        # sargable indexed JOIN instead of a LIKE scan (issue #85).
        conn.execute("DELETE FROM photo_tags WHERE photo_uid=?", (uid,))
        if clean:
            conn.executemany(
                "INSERT OR IGNORE INTO photo_tags (photo_uid, tag) VALUES (?, ?)",
                [(uid, t) for t in clean],
            )
    return clean


def all_tags() -> list[tuple[str, int]]:
    """Distinct user tags with counts (photos tagged with them)."""
    with get_conn() as conn:
        # CROSS JOIN pins the scan order so the planner drives from the tiny
        # photo_tags table (covering idx_photo_tags_tag scan) instead of
        # scanning every done photo through idx_photos_status (issue #85).
        rows = conn.execute(
            "SELECT pt.tag, COUNT(*) AS n "
            "FROM photo_tags pt CROSS JOIN photos p ON p.uid = pt.photo_uid "
            "WHERE p.status='done' "
            "GROUP BY pt.tag "
            "ORDER BY n DESC, pt.tag ASC"
        ).fetchall()
    return [(r["tag"], r["n"]) for r in rows]


def photos_by_tag(tag: str, limit: int = 200, offset: int = 0) -> list[sqlite3.Row]:
    """Photos that carry the given tag (case-insensitive).

    Sargable via the photo_tags join table (issue #85): the tag equality is
    resolved with an index instead of a LIKE scan over every done photo.
    """
    with get_conn() as conn:
        # CROSS JOIN pins the join order: drive from the tag equality on the
        # covering idx_photo_tags_tag index, then rowid-probe photos. Without
        # it the planner picks idx_photos_hidden/status and scans every done
        # photo probing photo_tags per row (issue #85).
        return conn.execute(
            "SELECT p.* FROM photo_tags pt "
            "CROSS JOIN photos p ON p.uid = pt.photo_uid "
            "WHERE p.status='done' AND p.thumb_path IS NOT NULL AND p.thumb_path != '' "
            "AND p.hidden = 0 AND pt.tag = ? "
            "ORDER BY p.capture_time DESC LIMIT ? OFFSET ?",
            (tag.lower(), limit, offset),
        ).fetchall()


def duplicate_groups(limit: int = 500) -> list[list[sqlite3.Row]]:
    """Photos whose Proton content-hash (sha1) appears more than once.

    Returns a list of groups (each is a list of sqlite3.Row). Groups are
    ordered by total photo count DESC so the worst offenders appear first.
    Members keep the ``hidden ASC, capture_time DESC`` sort so a dismissed
    copy stays visible (styled) at the end of its group.

    Single self-join (issue #90): the derived `g` table picks the top-`limit`
    duplicate sha1s with a ROW_NUMBER window preserving the count-DESC group
    order, so one round-trip returns every member of every group.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT p.* FROM photos p "
            "JOIN ("
            "  SELECT sha1, ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC, sha1) AS grp "
            "  FROM photos "
            "  WHERE status='done' AND sha1 IS NOT NULL AND sha1 != '' "
            "  GROUP BY sha1 HAVING COUNT(*) > 1 "
            "  LIMIT ?"
            ") g ON g.sha1 = p.sha1 "
            "WHERE p.status='done' "
            "ORDER BY g.grp, p.hidden ASC, p.capture_time DESC, p.uid ASC",
            (limit,),
        ).fetchall()
    out: list[list[sqlite3.Row]] = []
    for r in rows:
        if out and out[-1][0]["sha1"] == r["sha1"]:
            out[-1].append(r)
        else:
            out.append([r])
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
            "FROM photos INDEXED BY idx_photos_month_day "
            "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
            "AND hidden = 0 "
            "AND strftime('%m-%d', capture_time, 'unixepoch') = ? "
            "AND capture_time IS NOT NULL "
            "ORDER BY capture_time DESC LIMIT ?",
            (f"{int(month):02d}-{int(day):02d}", limit),
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
            "FROM photos INDEXED BY idx_photos_ym "
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
        membership: dict[str, list[str]] = {}
        for r in rows:
            try:
                uids = json.loads(r["albums"])
            except Exception:
                continue
            membership.setdefault(r["uid"], []).extend(uids)
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
        # Rebuild the normalized join table so album_photos() can use a
        # sargable indexed JOIN (issue #85). photos.albums stays the source of
        # truth; this mirrors it after each sync.
        conn.execute("DELETE FROM photo_albums")
        for uid, uids in membership.items():
            conn.executemany(
                "INSERT OR IGNORE INTO photo_albums (photo_uid, album_uid) VALUES (?, ?)",
                [(uid, u) for u in uids],
            )
        for u, n in counts.items():
            lo, hi = spans.get(u, (None, None))
            conn.execute(
                "UPDATE albums SET photo_count=?, cover_uid=?, start_ts=?, end_ts=? WHERE uid=?",
                (n, covers.get(u, (None, None))[1], lo, hi, u),
            )
    return len(albums)


def album_names(uids: list[str]) -> dict[str, str]:
    """Resolve album uids to their synced names from the local albums table.

    Unknown uids (albums the 10-minute sync hasn't seen yet) are omitted;
    callers decide how to render them.
    """
    if not uids:
        return {}
    marks = ",".join("?" * len(uids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT uid, name FROM albums WHERE uid IN ({marks})",
            tuple(uids),
        ).fetchall()
    return {r["uid"]: r["name"] or r["uid"] for r in rows}


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
            "SELECT p.* FROM photo_albums pa "
            "CROSS JOIN photos p ON p.uid = pa.photo_uid "
            "WHERE p.status='done' AND p.thumb_path IS NOT NULL AND p.thumb_path != '' "
            "AND pa.album_uid = ? "
            "ORDER BY p.capture_time DESC LIMIT ? OFFSET ?",
            (album_uid, limit, offset),
        ).fetchall()


# --- users & auth tokens ---------------------------------------------------

def create_user(username: str, password_hash: str, role: str = "read",
                display_name: str | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, display_name, password_hash, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, display_name or username, password_hash, role, int(time.time())),
        )
        return cur.lastrowid


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at, disabled "
            "FROM users WHERE id=?",
            (user_id,),
        ).fetchone()


def get_user_by_username(username: str) -> sqlite3.Row | None:
    """Returns the full user row including password_hash (server-side only)."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)
        ).fetchone()


def list_users() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at, disabled "
            "FROM users ORDER BY username COLLATE NOCASE"
        ).fetchall()


def update_user(user_id: int, *, display_name: str | None = None,
                role: str | None = None, disabled: bool | None = None,
                password_hash: str | None = None) -> bool:
    """Patch one or more user fields. Returns True if a row was updated."""
    sets: list[str] = []
    params: list = []
    if display_name is not None:
        sets.append("display_name=?")
        params.append(display_name)
    if role is not None:
        if role not in ("read", "write", "admin"):
            raise ValueError(f"invalid role: {role}")
        sets.append("role=?")
        params.append(role)
    if disabled is not None:
        sets.append("disabled=?")
        params.append(1 if disabled else 0)
    if password_hash is not None:
        sets.append("password_hash=?")
        params.append(password_hash)
    if not sets:
        return False
    params.append(user_id)
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id=?", params)
        return cur.rowcount > 0


def delete_user(user_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        return cur.rowcount > 0


def touch_last_login(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET last_login_at=? WHERE id=?",
                     (int(time.time()), user_id))


def _hash_token(token: str) -> str:
    """Return the SHA-256 hex digest of *token*.

    Tokens are stored as hashes so that a DB/backup disclosure does not yield
    live bearer tokens (issue #34). SHA-256 is appropriate here because tokens
    are 256-bit random strings — no slow KDF needed, and lookup stays O(1).
    """
    return hashlib.sha256(token.encode()).hexdigest()


def issue_token(user_id: int, kind: str, expires_in: int, *,
                user_agent: str | None = None, ip: str | None = None) -> str:
    """Mint a new opaque bearer token (32 random bytes hex-encoded).

    The raw token is returned to the caller (and ultimately to the HTTP
    response). Only its SHA-256 hash is stored in the database.
    """
    import secrets
    token = secrets.token_hex(32)
    now = int(time.time())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, kind, expires_at, created_at, user_agent, ip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_hash_token(token), user_id, kind, now + expires_in, now, user_agent, ip),
        )
    return token


def lookup_token(token: str) -> sqlite3.Row | None:
    """Return the (token, user_id, kind, expires_at) row if active, else None.

    Joined with users so the caller can see role/disabled without a second query.
    The input *token* is hashed before the SELECT — the DB only stores digests.
    """
    with get_conn() as conn:
        return conn.execute(
            """SELECT t.token, t.user_id, t.kind, t.expires_at,
                      u.username, u.display_name, u.role, u.disabled
               FROM auth_tokens t JOIN users u ON u.id = t.user_id
               WHERE t.token = ?""",
            (_hash_token(token),),
        ).fetchone()


def revoke_token(token: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM auth_tokens WHERE token=?", (_hash_token(token),))
        return cur.rowcount > 0


def revoke_all_tokens(user_id: int) -> int:
    """Sign the user out of every device. Returns number of tokens deleted."""
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM auth_tokens WHERE user_id=?", (user_id,))
        return cur.rowcount


def purge_expired_tokens() -> int:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM auth_tokens WHERE expires_at < ?", (int(time.time()),))
        return cur.rowcount


def backfill_legacy_favorites(admin_user_id: int) -> int:
    """One-time: copy photos.favorited=1 rows into user_favorites for the admin.

    Called when the first admin is created, only if photos.favorited=1 rows
    exist. After the copy, photos.favorited is cleared (kept as the historical
    "anyone starred this" flag; the per-user view is authoritative). Idempotent
    on the user_favorites side (INSERT OR IGNORE).
    """
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE favorited=1"
        ).fetchone()[0]
        if cur == 0:
            return 0
        conn.execute(
            """INSERT OR IGNORE INTO user_favorites (user_id, photo_uid, created_at)
               SELECT ?, uid, COALESCE(processed_at, ?) FROM photos WHERE favorited=1""",
            (admin_user_id, int(time.time())),
        )
        conn.execute("UPDATE photos SET favorited=0")
        return cur


# --- per-user favorites ----------------------------------------------------

def favorite_photo(user_id: int, photo_uid: str) -> bool:
    """Add to favorites. Returns True if it was already favorited."""
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO user_favorites (user_id, photo_uid, created_at) VALUES (?, ?, ?)",
                (user_id, photo_uid, int(time.time())),
            )
            return False
        except sqlite3.IntegrityError:
            return True


def unfavorite_photo(user_id: int, photo_uid: str) -> bool:
    """Remove from favorites. Returns True if a row was deleted."""
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM user_favorites WHERE user_id=? AND photo_uid=?",
            (user_id, photo_uid),
        )
        return cur.rowcount > 0


def is_favorite(user_id: int, photo_uid: str) -> bool:
    with get_conn() as conn:
        return conn.execute(
            "SELECT 1 FROM user_favorites WHERE user_id=? AND photo_uid=?",
            (user_id, photo_uid),
        ).fetchone() is not None


def favorite_uids(user_id: int, uids: list[str]) -> set[str]:
    """Return the subset of `uids` that the user has favorited (batch query)."""
    if not uids:
        return set()
    placeholders = ",".join("?" * len(uids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT photo_uid FROM user_favorites "
            f"WHERE user_id=? AND photo_uid IN ({placeholders})",
            [user_id, *uids],
        ).fetchall()
    return {r["photo_uid"] for r in rows}

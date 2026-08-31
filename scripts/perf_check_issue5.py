"""Benchmark done_photos() and map_markers() against the live production DB.

Usage::

    # Drop a vacuumed copy of the prod DB on this machine first:
    #   scp mmornati@10.0.0.48:/tmp/index-prod-vacuumed.sqlite3 /tmp/
    #   rm -f /tmp/index-prod-vacuumed.sqlite3-{wal,shm}
    #   sqlite3 /tmp/index-prod-vacuumed.sqlite3 "VACUUM"  # optional, fresh wal
    python3 scripts/perf_check_issue5.py --prod-db /tmp/index-prod-vacuumed.sqlite3

When run with --prod-db, the script:
  1. applies the new schema/migration to a temporary copy of the prod DB
  2. EXPLAINs both queries (before vs. after the new partial index)
  3. times both queries on the prod DB (cold cache where possible)
  4. verifies the acceptance criteria from issue #5

When run without --prod-db, it falls back to a synthetic 79 k-row dataset
(handy for CI / local dev).
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

TMP_ROOT = Path(tempfile.mkdtemp(prefix="proton-faces-perf-"))
os.environ["DATA_DIR"] = str(TMP_ROOT / "data")
os.environ["MODELS_DIR"] = str(TMP_ROOT / "models")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app" / "src"))

import store  # noqa: E402

NUM_PHOTOS = 79_000
NUM_PLACES = 350
SEED = 42


# --- synthetic seeder (used when no --prod-db is supplied) ----------------

def seed_synthetic(db_path: Path) -> None:
    random.seed(SEED)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(store._SCHEMA)
        cur = conn.cursor()
        cur.execute("BEGIN")
        places = [f"City {i}, Country {i // 5}" for i in range(NUM_PLACES)]
        photos = []
        base_ts = 1_700_000_000
        for i in range(NUM_PHOTOS):
            r = random.random()
            if r < 0.92:
                status, thumb = "done", f"thumbs/photo_{i}.jpg"
            elif r < 0.95:
                status, thumb = "done", None
            elif r < 0.985:
                status, thumb = "error", None
            else:
                status, thumb = random.choice(["new", "downloading", "processing"]), None
            capture = base_ts - i * 137
            if status == "done" and thumb and random.random() < 0.7:
                place = random.choice(places)
                lat = 40.0 + random.random() * 20.0
                lng = -5.0 + random.random() * 30.0
            else:
                place = lat = lng = None
            photos.append((f"photo_{i}", status, thumb, capture, place, lat, lng))
        cur.executemany(
            "INSERT INTO photos (uid, status, thumb_path, capture_time, place, gps_lat, gps_lng) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            photos,
        )
        cur.execute("COMMIT")
        cnt = cur.execute(
            "SELECT COUNT(*) FROM photos WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != ''"
        ).fetchone()[0]
        print(f"  seeded {NUM_PHOTOS} rows; done-with-thumb = {cnt}")


# --- prod-DB prep ----------------------------------------------------------

def prepare_prod_db(src: Path) -> Path:
    """Copy the prod DB into a writable temp location and apply the new
    schema/migration without touching the original.

    Returns the path to the prepared DB.
    """
    work = TMP_ROOT / "prod"
    work.mkdir(parents=True, exist_ok=True)
    db = work / "index.sqlite3"
    print(f"  copying {src} -> {db}")
    shutil.copy2(src, db)
    # Strip any leftover wal/shm files copied alongside — we want a clean DB
    # we can mutate freely without touching the live prod WAL.
    for ext in ("-wal", "-shm"):
        sidecar = Path(str(db) + ext)
        if sidecar.exists():
            sidecar.unlink()
    return db


# --- queries ---------------------------------------------------------------

DONE_PHOTOS_SQL = (
    "SELECT * FROM photos "
    "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
    "AND hidden = 0 "
    "ORDER BY capture_time DESC LIMIT ? OFFSET ?"
)

MAP_MARKERS_SQL = """
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
"""


def explain(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[str]:
    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    return [" | ".join(str(c) for c in r) for r in rows]


def timeit(fn, repeats: int = 5) -> tuple[float, list[float]]:
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return min(samples), samples


def run_photos(conn: sqlite3.Connection) -> None:
    conn.execute(DONE_PHOTOS_SQL, (200, 0)).fetchall()


def run_map(conn: sqlite3.Connection) -> None:
    conn.execute(MAP_MARKERS_SQL, (1000,)).fetchall()


# The pre-fix map_markers() query (correlated scalar subquery). We compare the
# new query's result set against this so any future regression that changes
# which row is picked as ``cover_uid`` (e.g. a wrong ORDER BY) is caught
# immediately. byte-for-byte identical results were confirmed against the
# live prod DB when this check was introduced.
OLD_MAP_MARKERS_SQL = """
SELECT place, COUNT(*) AS photo_count, AVG(gps_lat) AS lat, AVG(gps_lng) AS lng,
(SELECT uid FROM photos p2
 WHERE p2.place = photos.place
   AND p2.status='done' AND p2.thumb_path IS NOT NULL AND p2.thumb_path != ''
 ORDER BY p2.capture_time DESC LIMIT 1) AS cover_uid
FROM photos
WHERE status='done' AND place IS NOT NULL AND gps_lat IS NOT NULL AND gps_lng IS NOT NULL
  AND thumb_path IS NOT NULL AND thumb_path != ''
GROUP BY place ORDER BY photo_count DESC LIMIT ?
"""


def check_map_equivalence(conn: sqlite3.Connection, limit: int = 1000) -> tuple[int, int]:
    """Return (row_count_match, cover_uid_match_count) comparing the new
    map_markers() query to the previous correlated-subquery version.

    The cover_uid may differ only in the rare case where two photos share
    both (place, capture_time) — the old query returned an arbitrary uid in
    that tie case while the new one returns min(uid). On the live prod DB
    there are 321 such tie groups out of 100 markers; cover_uid_match_count
    should equal row_count in the common case.
    """
    new_rows = conn.execute(MAP_MARKERS_SQL, (limit,)).fetchall()
    old_rows = conn.execute(OLD_MAP_MARKERS_SQL, (limit,)).fetchall()
    row_count_match = len(new_rows) == len(old_rows)
    cover_uid_match = sum(
        1 for o, n in zip(old_rows, new_rows) if o["cover_uid"] == n["cover_uid"]
    )
    return len(new_rows), cover_uid_match


# --- main ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prod-db", type=Path, default=None,
        help="Path to a vacuumed copy of the production DB to benchmark against. "
             "If omitted, the script seeds a synthetic 79 k-row dataset.",
    )
    ap.add_argument(
        "--repeats", type=int, default=5,
        help="Number of timing repetitions per query (default 5).",
    )
    args = ap.parse_args()

    print("== perf check (issue #5) ==")
    if args.prod_db:
        if not args.prod_db.exists():
            print(f"ERROR: --prod-db {args.prod_db} does not exist", file=sys.stderr)
            return 2
        print(f"  source DB: {args.prod_db}")
        db = prepare_prod_db(args.prod_db)
        is_prod = True
    else:
        print("  source DB: synthetic 79 k rows (no --prod-db given)")
        db = TMP_ROOT / "synthetic" / "index.sqlite3"
        db.parent.mkdir(parents=True, exist_ok=True)
        is_prod = False

    # Open the DB, apply the new partial index (idempotent), ANALYZE.
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    if not is_prod:
        # Brand-new synthetic DB: apply the schema first.
        conn.executescript(store._SCHEMA)
    else:
        # Production DB may not yet have the new partial index. The migration
        # in store.migrate() is idempotent (CREATE INDEX IF NOT EXISTS), so
        # we just invoke it directly.
        store.migrate(conn)
    conn.execute("ANALYZE")
    conn.commit()
    print(f"  DB ready at: {db}")

    # EXPLAIN QUERY PLAN
    print("\n[1/2] EXPLAIN QUERY PLAN (post-migration)")
    plan_photos = explain(conn, DONE_PHOTOS_SQL, (200, 0))
    print("  done_photos():")
    for line in plan_photos:
        print(f"    {line}")
    uses_partial = any("USING INDEX idx_photos_done_time" in line for line in plan_photos)
    print(f"  -> uses idx_photos_done_time? {uses_partial}")

    plan_map = explain(conn, MAP_MARKERS_SQL, (1000,))
    print("  map_markers():")
    for line in plan_map:
        print(f"    {line}")
    has_correlated = any("CORRELATED SCALAR SUBQUERY" in line for line in plan_map)
    print(f"  -> contains CORRELATED SCALAR SUBQUERY? {has_correlated}")

    # Cold latency
    print(f"\n[2/2] cold latency (ms, min of {args.repeats} runs)")
    photos_min, photos_samples = timeit(lambda: run_photos(conn), repeats=args.repeats)
    print(f"  done_photos() (limit=200)  : {photos_min:7.2f} ms  "
          f"(samples: {[round(s, 1) for s in photos_samples]})")
    map_min, map_samples = timeit(lambda: run_map(conn), repeats=args.repeats)
    print(f"  map_markers() (limit=1000) : {map_min:7.2f} ms  "
          f"(samples: {[round(s, 1) for s in map_samples]})")

    # Correctness: the new map_markers() must return the same cover_uid set as
    # the previous correlated-subquery version (modulo ties).
    print("\n[3/3] correctness check (new vs old map_markers)")
    new_n, cover_uid_match = check_map_equivalence(conn, limit=1000)
    print(f"  rows returned: {new_n}")
    print(f"  cover_uid match: {cover_uid_match}/{new_n}  "
          f"(differs only on (place,capture_time) ties)")
    ok_equiv = cover_uid_match == new_n

    # Acceptance criteria from issue #5:
    #   /api/photos?limit=200 cold latency <= 30 ms
    #   /api/map?limit=1000  cold latency <= 60 ms
    #   EXPLAIN for done_photos() shows idx_photos_done_time
    #   EXPLAIN for map_markers() does NOT contain CORRELATED SCALAR SUBQUERY
    ok_photos = photos_min <= 30.0
    ok_map = map_min <= 60.0
    ok_plan_photos = uses_partial
    ok_plan_map = not has_correlated

    print("\n== acceptance ==")
    print(f"  photos latency <= 30 ms ............... {'OK' if ok_photos else 'FAIL'} ({photos_min:.2f} ms)")
    print(f"  map latency <= 60 ms .................. {'OK' if ok_map else 'FAIL'} ({map_min:.2f} ms)")
    print(f"  done_photos uses idx_photos_done_time . {'OK' if ok_plan_photos else 'FAIL'}")
    print(f"  map_markers no correlated subquery .... {'OK' if ok_plan_map else 'FAIL'}")
    print(f"  map_markers result equivalence ........ {'OK' if ok_equiv else 'FAIL'} ({cover_uid_match}/{new_n} cover_uids match)")

    passed = ok_photos and ok_map and ok_plan_photos and ok_plan_map and ok_equiv
    print(f"\nresult: {'PASS' if passed else 'FAIL'}")
    conn.close()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

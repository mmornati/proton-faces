import sqlite3, time

c = sqlite3.connect("/data/index.sqlite3")
c.row_factory = sqlite3.Row

# map_markers as deployed
sql_map = """
WITH places AS (
  SELECT place, COUNT(*) AS photo_count, AVG(gps_lat) AS lat, AVG(gps_lng) AS lng
  FROM photos
  WHERE status='done' AND place IS NOT NULL AND gps_lat IS NOT NULL AND gps_lng IS NOT NULL
        AND thumb_path IS NOT NULL AND thumb_path != ''
  GROUP BY place
),
newest AS (
  SELECT place, MAX(capture_time) AS cover_ts
  FROM photos
  WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != ''
        AND place IS NOT NULL
  GROUP BY place
),
cover AS (
  SELECT place, uid AS cover_uid
  FROM (
    SELECT n.place, p.uid,
           ROW_NUMBER() OVER (PARTITION BY n.place ORDER BY p.uid ASC) AS rn
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
SELECT p.place, p.photo_count, p.lat, p.lng, c.cover_uid
FROM places p LEFT JOIN cover c ON c.place = p.place
ORDER BY p.photo_count DESC LIMIT ?
"""
print("--- map_markers() deployed (cold 5 runs) ---")
for i in range(5):
    t = time.time()
    rows = c.execute(sql_map, (1000,)).fetchall()
    print(f"  run{i+1}: {(time.time()-t)*1000:.1f} ms  rows={len(rows)}")
print("--- plan ---")
for r in c.execute(f"EXPLAIN QUERY PLAN {sql_map}", (1000,)).fetchall():
    print(f"  {r[3]}")

# done_photos as deployed (with hidden=0 filter from PR #2)
sql_photos = (
    "SELECT * FROM photos WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
    "AND hidden = 0 ORDER BY capture_time DESC LIMIT 200"
)
print("\n--- done_photos() deployed (cold 5 runs) ---")
for i in range(5):
    t = time.time()
    c.execute(sql_photos).fetchall()
    print(f"  run{i+1}: {(time.time()-t)*1000:.1f} ms")
print("--- plan ---")
for r in c.execute(f"EXPLAIN QUERY PLAN {sql_photos}").fetchall():
    print(f"  {r[3]}")
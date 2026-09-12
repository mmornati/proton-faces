# Troubleshooting

Common gotchas and how to fix them. If something here doesn't help, open an issue with the indexer logs (`docker compose logs --tail 200 indexer`) and your `.env` (with the session file redacted).

## Startup

### "Bind for :::8080 failed: port is already allocated"

Another service on the host is using port 8080. Change `PORT` in `.env`:

```bash
PORT=8090
docker compose up -d
```

### "permission denied" writing to `/data`

The data volume must be writable by UID 1000 (the user all containers run as).

```bash
sudo chown -R 1000:1000 /srv/proton-faces/data
```

If you're using a host directory, ensure it exists and has the right perms before `docker compose up`.

### "Cannot import name 'BridgeClient' from 'bridge_client'"

Old code in `__pycache__/`. Either rebuild the image:

```bash
docker compose build --pull
docker compose up -d
```

Or, if you're using the local-dev override, delete `app/src/__pycache__/` on the host and restart.

### The bridge can't reach Proton (bridge offline in the footer)

```bash
docker compose logs proton-bridge
```

Look for HTTP errors, DNS failures, or auth failures. If the session expired:

```bash
# Re-export the session
scripts/export-session.sh
docker compose restart proton-bridge
```

### Full-resolution downloads hang (stale Proton SDK cache)

**Symptom:** `/api/photos/{uid}/full` requests take ~30 seconds and return
HTTP 504. Other bridge endpoints (`/health`, `/timeline`, `/thumbnails`)
keep working, and `Bridge reachable` stays `ok` in the admin checks. The
indexer's full-res loop spins on the same uids forever; videos pile up in
`status='full'`. `docker compose logs proton-bridge` shows
`AbortError` or `waitForCondition2` from `getFileDownloader`.

**Root cause:** The Proton SDK persists an entity cache
(`cache-entities.sqlite`) and a crypto cache (`cache-crypto.sqlite`)
under `DATA_DIR` (mounted from `${DATA_MOUNT}` on the host). After a
Proton-side incident that reshuffles node keys — e.g. the 2026-08-27
Frankfurt cooling failure or the 2026-09-01 partial outage — those
caches can go stale. The caches survive container restarts (they live on
the mounted volume), so the hang returns on every bridge boot until the
caches are cleared and the SDK fetches fresh state. Nothing in the SDK
auto-invalidates this scenario, so a manual clear is required.

**One-click fix (admin UI):**

1. Open the admin modal → **Server checks**.
2. Look at the **Bridge cache** row. If it's `stale`, click
   **Clear bridge cache**. The bridge unlinks its cache files and exits;
   compose's `restart: unless-stopped` policy respawns it ~5-10 seconds
   later with a fresh cache. Auth-session state is preserved.
3. Click **Run checks now** to confirm `Bridge cache` is back to `ok`
   and that `/api/photos/{uid}/full` returns 200 again.

**Manual fix (SSH):**

```bash
cd /home/mmornati/proton-faces
docker compose stop proton-bridge
docker compose rm -f proton-bridge
# Move aside (safer than delete — you can inspect if needed)
mv "${DATA_MOUNT:-data}/cache-crypto.sqlite"* "${DATA_MOUNT:-data}/" 2>/dev/null || true
# Or delete outright (the SDK recreates them on startup):
#   rm -f "${DATA_MOUNT:-data}/cache-crypto.sqlite"*
#   rm -f "${DATA_MOUNT:-data}/cache-entities.sqlite"*
docker compose up -d proton-bridge
```

The session file (`auth-session.json`) lives in the same directory but is
NOT named `cache-*`, so it survives both the one-click and manual fixes
— no re-authentication needed.

**Tuning:** the staleness threshold in the admin check is
`BRIDGE_CACHE_STALE_SEC` (default `21600` = 6 h). Lower it to surface
stale caches earlier; raise it if your bridge is legitimately idle for
longer than 6 h between requests.

### Demo mode won't start: "fixture.json missing"

You deleted or moved `app/src/demo_assets/`. Re-fetch:

```bash
python scripts/fetch-demo-assets.py
docker compose --profile demo up -d
```

## Indexing

### Photos stuck in "pending" / "downloading"

The downloader polls every 5 seconds. If it's been 10+ minutes with no progress:

```bash
docker compose logs indexer | grep -E "thumbnail|failed|error"
```

Common causes:

- **Bridge offline.** Check the footer pill or `docker compose ps`.
- **Work dir full.** Check `df -h $(docker volume inspect proton-faces_data --format '{{ .Mountpoint }}')`.
- **Session expired.** Re-export the session file.

### "Name or service not known" in `thumbnail batch failed`

The bridge container can't resolve DNS or the URL is wrong. Check `BRIDGE_URL` in `.env`:

```bash
# Inside the bridge container
docker compose exec proton-bridge env | grep BRIDGE
```

The default `http://proton-bridge:8090` is correct for the default compose stack.

### CLIP / InsightFace not loading

```
FileNotFoundError: /models/clip/vision_model.onnx
```

The image is missing the baked models. Use the published image (`ghcr.io/mmornati/proton-faces-app:latest`), not a custom build unless you've added the model-baking steps from the Dockerfile.

### `insightface` segfault on import

Some Python 3.12 + insightface combinations segfault on first import. The published image pins Python 3.11 for exactly this reason. If you're running locally, use Python 3.11.

### Face count not growing on a new photo

- Verify the photo has a thumbnail: `docker compose exec indexer ls /data/thumbs/` — the uid should be there.
- Verify the photo is `status='done'`: `GET /api/stats`.
- Look for errors: `docker compose logs indexer | grep "failed processing"`.

## People / clustering

### No people clusters

- Fewer than `MIN_CLUSTER_SIZE` (default 2) faces detected → no clusters.
- Run `docker compose exec indexer python -c "from cluster import cluster_once; print(cluster_once())"` to force a cluster run.

### Wrong person merged into wrong person

If only a few faces are wrong, unassign them:

1. Open the wrong person's photo → click the face → **Unassign**.
2. The face moves back to "Unknown person #N".
3. Tag it correctly.

If a whole person was merged into another by mistake (hundreds/thousands of
faces), see [Recovering a merged person](#recovering-a-merged-person) below —
manual unassigning doesn't scale.

### Recovering a merged person

Merging two people only re-parents face rows (their immutable IDs are kept)
and deletes the source person row. It never touches the photos themselves, so
a **pre-merge backup** of the index database contains everything needed to
undo the merge: the backup's face-ID list maps 1:1 onto the live database.

`scripts/recover-merged-person.py` moves those faces back to a recreated
person row and recounts both people. It is a one-shot, idempotent operation —
no app code changes, no re-indexing.

#### Prerequisites

- A backup of `index.sqlite3` taken **before** the merge (see
  [Backups](../user-guide/admin.md#backups) in the admin guide, or
  `scripts/backup.sh`). The script reads it read-only.
- Console access to the server (SSH) and the script on the machine that can
  reach the live database. If the deployment directory isn't a git checkout,
  copy it over: `scp scripts/recover-merged-person.py user@host:/path/to/deploy/`.
- If the merge deleted the person row, **recreate the person in the UI first**
  (name the cluster "Gaia Mornati" again). The script targets the person by
  name, so the recreated row is the recovery target — you do **not** need to
  delete it.

#### Step-by-step (real example: Gaia merged into Bastien)

This is the exact procedure used to recover Gaia Mornati (4,898 faces) from a
mistaken merge into Bastien Mornati.

```bash
# 1. Stop the indexer and the app so nothing writes to the DB mid-recovery.
docker compose stop indexer app

# 2. Snapshot the live DB (belt and braces — the script also snapshots itself).
cp /media/12tb/photos-index/index.sqlite3 \
   /media/12tb/photos-index/_backups/index-pre-recover-$(date +%Y%m%d-%H%M%S).sqlite3

# 3. Preview what will happen (no writes).
python3 scripts/recover-merged-person.py \
    --backup /media/12tb/photos-index/_backups/index-20260910-062505.sqlite3 \
    --name "Gaia Mornati" \
    --dry-run

# 4. Run the recovery for real.
python3 scripts/recover-merged-person.py \
    --backup /media/12tb/photos-index/_backups/index-20260910-062505.sqlite3 \
    --name "Gaia Mornati"

# 5. Restart the services.
docker compose start indexer app
```

The real run printed:

```text
backup person : 'Gaia Mornati' (id 605) — 4898 faces
live target   : 'Gaia Mornati' (id 53707) — 4 faces
  already on target : 4
  to move back      : 4894
  donor person 1680: 4894 faces
snapshot: /media/12tb/photos-index/_backups/index-pre-recover-20260912-092402.sqlite3
after      target: 4898 faces / 4855 photos
after  donor 1680: 11668 faces / 11532 photos
done. Restart the indexer so it rewrites the face sidecar.
```

#### What happens

- **Only face re-parenting** — `UPDATE faces SET person_id=<target>` for the
  backup's face IDs. No merge, no person deletion: Gaia keeps her own row,
  Bastien keeps his.
- **Cover/name backfill** — the backup's `name`, `cover_uid` and
  `cover_face_id` are restored only if the target row has none, so a cover you
  picked on the recreated person is kept.
- **Recount** — `face_count` / `photo_count` are recomputed for both people
  with the same SQL the app uses.
- **Idempotent** — re-running reports "nothing to do".
- **Safe** — snapshots the live DB before writing, aborts (exit 2) if the
  backup DB is missing, the person isn't found, or any backup face ID is
  missing from the live DB, and `--dry-run` previews without writing.

#### When the UI reflects the change

- The **People grid** reads the database directly, so counts and covers are
  correct as soon as the app restarts.
- **Face search / suggested merges** read the mmap face sidecar
  (`index/embeddings_meta.json`), which bakes in `person_ids`. The indexer
  only rewrites it on the next cluster run (`CLUSTER_INTERVAL`, default
  30 min) or when new photos are processed — **not** on restart. Expect the
  sidecar to catch up within ~30 minutes, or force it immediately:

  ```bash
  docker compose exec indexer python -c "from cluster import cluster_once; print(cluster_once())"
  ```

#### Verifying the recovery

- In the UI: open the recovered person — the photo grid should be populated
  again, and the donor person should no longer show those photos.
- Via SQL (adjust paths for your setup):

  ```bash
  python3 -c "
  import sqlite3
  db = sqlite3.connect('/media/12tb/photos-index/index.sqlite3')
  for pid, name in [(53707, 'Gaia'), (1680, 'Bastien')]:
      r = db.execute('SELECT face_count, photo_count FROM people WHERE id=?', (pid,)).fetchone()
      print(name, pid, 'faces=', r[0], 'photos=', r[1])
  print('total faces:', db.execute('SELECT COUNT(*) FROM faces').fetchone()[0])
  "
  ```

  The total face count must be unchanged — a recovery never loses or creates
  faces.

### Suggested merges are missing / wrong

The threshold is `0.40` by default. Adjust in the request:

```bash
curl 'http://localhost:8080/api/people/duplicates?threshold=0.30'
```

Lower = more suggestions, more false positives.

## Places / GPS

### Map is empty

- In real mode: GPS comes from each photo's own EXIF, read when the indexer downloads the full-res original. For photos indexed before that was added, run `docker compose exec indexer python indexer_main.py --backfill-gps-exif` to re-download and extract. See [Places](../user-guide/places.md).
- In demo mode: `apply_demo_gps()` runs every sync cycle; check the indexer logs for `applied GPS for N fixture photos`.

### Wrong city name

`reverse_geocoder` returns the nearest city within its dataset. For remote or unusual locations it can be inaccurate. The photo's own EXIF GPS is authoritative.

## Admin

### Can't delete a user

The server refuses to delete the last admin. Promote another user to `admin` first.

### Forgot admin password

```bash
docker compose exec app python main.py --reset-password admin
```

The command prompts for a new password. Set `ADMIN_PASSWORD=...` to skip the prompt.

## Performance

### /api/photos is slow (>200 ms)

- Run `EXPLAIN QUERY PLAN` on `done_photos()` — the partial index `idx_photos_done_time` should be used.
- Check `VACUUM` has been run recently: `docker compose exec app python -c "import sqlite3; c=sqlite3.connect('/data/index.sqlite3'); c.execute('VACUUM')"`.

### CLIP search is slow (>1 s)

The CLIP matrix cache is 60 s. On a 100k library, rebuild is ~88 MB; should be sub-second. If it's slow, check that `WORKERS` isn't starving the API event loop (default 3; try 2).

### Disk filling up

`thumb_dir_bytes` in `GET /api/status`. Each thumbnail is ~10–80 KB; a 100k library is ~1–8 GB. Delete the data volume to start fresh, or move it to a bigger disk (`DATA_MOUNT`).

## Bridge

### "bridge is not logged in" in the footer

```bash
docker compose logs proton-bridge | tail -30
```

Look for "missing session", "401 Unauthorized", or "session expired". Re-export:

```bash
scripts/export-session.sh
docker compose restart proton-bridge
```

### "no image preview" warnings

Proton doesn't preview HEIC / videos. The indexer handles them in the fullres loop. You should see "fullres: generated thumbnail for …" in the logs; if not, check `ffmpeg` is on PATH (it is in the published image) and that `ffprobe` works.

### `Executable not found in $PATH: "pass"` on bridge startup

The bridge crash-loops with `ValidationError: Failed to load session from pass … Executable not found in $PATH: "pass"` when `PROTON_DRIVE_CREDENTIALS_STORE=pass` is set but the running image has no `pass`/`gnupg` binaries. Two fixes:

- **Plaintext store (default):** set `PROTON_DRIVE_CREDENTIALS_STORE=unsafe_file` in `.env` (or delete the line), then `docker compose up -d`.
- **Encrypted store:** `docker compose pull proton-bridge` to pick up an image with pass support, keep `=pass`, then `docker compose up -d`. The entrypoint migrates your existing session into the encrypted store automatically — see [Session file → Encrypted store](../getting-started/session-export.md#encrypted-store-pass).

Two gotchas:

- `docker compose restart` does **not** re-read `.env` — only `up -d` (recreate) applies env changes. Editing `.env` and restarting leaves the old container crash-looping.
- The variable only exists since PR #129. Before that, compose hardcoded `unsafe_file`, so a stale `PROTON_DRIVE_CREDENTIALS_STORE=pass` line in an old `.env` was silently ignored and activates on upgrade.

## Upgrading

### Schema migration error

`init_db()` runs idempotent column migrations. If you see `ALTER TABLE … duplicate column`, you're running against a DB that already has the migration applied — safe to ignore. If you see other errors, capture the full stack trace and open an issue.

### Image not updating after `docker compose pull`

Old containers might still be running:

```bash
docker compose ps
docker compose down
docker compose up -d
```

---

Still stuck? Open an issue on GitHub with `docker compose logs` (sanitized) and your `.env` (with the session file removed).

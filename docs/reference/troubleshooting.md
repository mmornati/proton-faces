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

Unassign the bad faces:

1. Open the wrong person's photo → click the face → **Unassign**.
2. The face moves back to "Unknown person #N".
3. Tag it correctly.

### Suggested merges are missing / wrong

The threshold is `0.40` by default. Adjust in the request:

```bash
curl 'http://localhost:8080/api/people/duplicates?threshold=0.30'
```

Lower = more suggestions, more false positives.

## Places / GPS

### Map is empty

- In real mode: you need a Google Takeout export mounted at `/takeout`. See [Places](../user-guide/places.md).
- In demo mode: `apply_demo_gps()` runs every sync cycle; check the indexer logs for `applied GPS for N fixture photos`.

### Wrong city name

`reverse_geocoder` returns the nearest city within its dataset. For remote or unusual locations it can be inaccurate. Use the Takeout export's GPS as authoritative.

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

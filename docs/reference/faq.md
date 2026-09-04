# FAQ

## General

### What is Proton Faces?

A self-hosted app that turns your end-to-end encrypted Proton Drive Photos into a searchable library. Face recognition, object/scene search, place search, all running locally on your hardware.

### Is it an official Proton project?

No. This is an independent project by [Marco Mornati](https://github.com/mmornati). "Proton", "Proton Drive" and "Proton Photos" are trademarks of Proton AG. Use at your own risk.

### Is it free?

Yes. MIT licensed. No paid tier, no telemetry, no upsell.

### Does it work with Google Photos / iCloud / Dropbox?

No. It works with Proton Drive Photos specifically because that's the only cloud-photo provider that does end-to-end encryption with a usable third-party SDK. A future extension could read from any local photo folder (Google Takeout exports are already supported as a side-channel for GPS data).

### Do I need a GPU?

No. All ML runs on CPU through ONNX Runtime.

### How big a library can it handle?

On a modern CPU, the indexer processes ~1–2 photos/sec. A 100k-photo library takes roughly a day to do the first pass. After that, only new photos are processed (5-second polling window).

## Privacy

### Are my photos uploaded anywhere?

No. The bridge is **strictly read-only** against Proton — no uploads, no edits, no deletes. The only network calls go to Proton's servers.

### Are my full-resolution photos stored on disk?

No. The fullres loop downloads HEIC / videos once, generates a 512px thumbnail, and discards the full-res bytes. The thumbnail (WebP, ~10–80 KB per photo) is the only thing kept.

### Is there any telemetry?

No. No analytics SDK, no remote logs, no error reporting to third parties.

### Where do face embeddings live?

In `DATA_DIR/index.sqlite3`, as a 512-float32 BLOB per face. Same for CLIP embeddings (one per photo). Your data is in your hands.

## Demo mode

### What is demo mode?

A built-in mode that replaces the Proton bridge with a fixture of 82 CC0/Unsplash photos. Lets you try the app without a Proton account. See the [demo-mode guide](../getting-started/demo-mode.md).

### Does demo mode touch my real Proton account?

No. The bridge container is skipped entirely in demo mode.

### Can I extend demo mode with my own photos?

Yes — but you'll need to rebuild the app image with your fixture. The simpler path: use the real install with a session file.

## Performance

### Why is the first sync slow?

Every new photo gets:

1. A timeline diff (cheap).
2. A thumbnail fetch (network-bound).
3. InsightFace face detection (~50–200 ms).
4. CLIP image embedding (~100 ms).
5. SQLite writes (cheap).

You can lower `SYNC_LIMIT` for testing or `WORKERS` for low-resource boxes.

### My photos are stuck in `pending`. What do I do?

Wait — the sync runs every `SYNC_INTERVAL` (5 min default). If it's been hours, check `docker compose logs indexer` for errors. A common cause is the bridge session expiring; export a new one and `docker compose restart proton-bridge`.

### My CPU is pegged. How do I reduce load?

Lower `WORKERS` (default 3, can go to 1) and/or `CLUSTER_INTERVAL` (default 30 min — raising this only delays clustering, not indexing).

### The map is empty. Why?

Either no photos have GPS yet, or `enrich_places()` hasn't run. In real mode, mount a Google Takeout export and run `docker compose exec indexer python indexer_main.py --backfill-gps`. In demo mode, `apply_demo_gps()` runs automatically every sync.

## People / face tagging

### How do I re-cluster everyone from scratch?

`docker compose exec indexer python indexer_main.py` doesn't expose a recluster command. The simplest path:

```sql
-- from inside the indexer container, with sqlite3
DELETE FROM people;
UPDATE faces SET person_id = NULL;
```

Then wait for the next `CLUSTER_INTERVAL` (or restart the indexer to trigger immediately).

### Two clusters should be the same person. What do I do?

Three options:

1. Click **Merge into…** on one of the cards in the People tab.
2. Rename one cluster to the other's name (auto-merges).
3. Click a suggested merge from the strip above the grid.

### A face was tagged wrongly. How do I fix it?

Open the photo → click the face box → **Unassign**. Then tag the right person (or create a new name).

## Configuration

### Can I run the indexer and the app on different machines?

Not currently — they share the SQLite index and need a fast local connection (read: a shared filesystem, or a remote DB like Postgres). Out of scope for the current release.

### Can I run the bridge on a different machine?

Not really — the bridge writes thumbnails into `DATA_DIR` and the indexer reads from there. They need shared disk.

### Can I use Postgres instead of SQLite?

Not currently. The vector store + per-row atomic claims are tuned for SQLite. A future migration is possible but not planned.

### How do I move the data volume to a bigger disk?

Set `DATA_MOUNT` in `.env` to a host path on the new disk, then `docker compose down && docker compose up -d`. The data is preserved across restarts as long as the directory is mounted.

## API

### Where do I get an API token?

`POST /api/auth/login` with `{username, password}` returns `{access_token, refresh_token, user}`. Use the access token in `Authorization: Bearer …`.

### How do I refresh a token?

`POST /api/auth/refresh` with `{refresh_token}` returns a new access token.

### Can I use OAuth?

No — Proton doesn't publish an OAuth provider for third-party apps. The bearer-token model is local-only.

## Maintenance

### How do I update to a newer image?

```bash
docker compose pull
docker compose up -d
```

The data volume is preserved. Schema migrations are automatic.

### How do I back up?

Either click **Backup now** in the admin area, or:

```bash
docker compose exec app python main.py --backfill-gps # ... no wait, that's GPS
# Manual backup:
docker compose exec app python -c "
import sqlite3
con = sqlite3.connect('/data/index.sqlite3')
con.execute('VACUUM INTO \"/data/_backups/index-manual.sqlite3\"')
"
```

Or just `cp data/index.sqlite3 backup.sqlite3` after `docker compose stop` (consistent against WAL).

### How do I restore?

```bash
docker compose down
cp backup.sqlite3 data/index.sqlite3
docker compose up -d
```

The indexer will pick up the restored DB and re-sync the missing thumbnails.

---

**Next:** [Troubleshooting](troubleshooting.md) for the common gotchas.

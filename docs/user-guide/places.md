# Places

The Places tab is a **Leaflet world map** with one clustered marker per city you've photographed in. Click any marker (or any chip in the city list below the map) to filter the photos grid to that place.

![Places map](../assets/screenshots/places.png){ loading=lazy }

## Where place names come from

Proton's API doesn't expose photo location, so two paths feed the `photos.place` column:

### 1. Google Takeout export (real mode)

If you mount a local Google Takeout export at `/takeout` (compose-level `PHOTOS_MOUNT`), the indexer's GPS backfill job runs automatically:

```bash
docker compose exec indexer python indexer_main.py --backfill-gps
```

What it does:

- Walks `*.supplemental-metadata.json` sidecars under the export root.
- sha1-hashes every photo file and pairs it with the sidecar's GPS coordinates.
- Joins against the indexed timeline by sha1 (Proton uses the same content hash), so **no full-res download is needed**.
- Caches the sha1 → GPS map in `DATA_DIR/gps_sha1_cache.json` for fast re-runs.
- Runs reverse-geocoding (`reverse_geocoder` library, offline data files) on every photo that has GPS but no place yet.

The cache file is `data/gps_sha1_cache.json`. Re-running `python indexer_main.py --backfill-gps` reuses it. Force a rebuild with `--rebuild-cache` after adding new Takeout exports.

### 2. Demo fixture (demo mode)

In demo mode, place names are baked into the fixture (`app/src/demo_assets/fixture.json`). The `apply_demo_gps()` hook in `app/src/demo.py` writes them into SQLite after every sync.

## The map

- **OpenStreetMap tiles** served by Leaflet. No API key required.
- **Clustered markers** — nearby cities are merged into a single cluster marker with a count badge; click to zoom in.
- **Marker click** — opens a small popup with the city name, the photo count, and a thumbnail of a representative photo. The popup's **"Show photos"** link filters the photos grid to that place.
- **"Show photos"** also returns the same set of photos as `/api/photos?place=Paris`.

## The city list

Below the map, a row of chips lists every city in your library, sorted by photo count. Each chip shows the city name and the photo count. Click a chip → filter the grid.

## The per-person map

Click **Map** on any card in the **People** tab. A Leaflet map opens filtered to photos that contain that person's face — the same `/api/people/{id}/map` endpoint serves both views.

![Person map](../assets/screenshots/person-detail.png){ loading=lazy }

## The Places view — performance

- `/api/map` returns up to 1000 markers by default; the marker clusterer on the client keeps the UI smooth even on 100k-photo libraries.
- `/api/places` is paginated (`limit=500`) for the city list chip strip.
- Place reverse-geocoding runs once per photo, in a background loop (`GPS_INTERVAL` default 6 h). The work is idempotent: re-running enriches only photos where `place IS NULL`.

## What about iPhone photos without GPS?

Most iPhones embed GPS into the EXIF. If the EXIF is stripped on export, Google Takeout will still have it (in the sidecar). If neither is available, the photo won't appear on the map — it just won't have a place label.

## API endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/map?limit=1000` | Markers for the world map |
| `GET /api/places?limit=500` | City list for the chip strip |
| `GET /api/photos?place=Paris` | Photos from a place |
| `GET /api/people/{id}/map` | Per-person map |
| `GET /api/search?q=Lille` | Text search also matches place names |

---

**Next:** [Albums & tags](albums-tags.md) covers Proton albums (read-only) and your free-form tags.

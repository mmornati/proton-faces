# Places

The Places tab is a **Leaflet world map** with one clustered marker per city you've photographed in. Click any marker (or any chip in the city list below the map) to filter the photos grid to that place.

![Places map](../assets/screenshots/places.png){ loading=lazy }

## Where place names come from

Proton's API doesn't expose photo location, so two paths feed the `photos.place` column:

### 1. EXIF GPS (real mode)

The indexer reads GPS coordinates straight from each photo's own EXIF when it downloads the full-res original (the fullres loop). Coordinates are persisted to `photos.gps_lat` / `photos.gps_lng` before the original is deleted, then reverse-geocoded to a place name.

For photos indexed before this was added, a one-shot backfill re-downloads originals and reads their EXIF:

```bash
docker compose exec indexer python indexer_main.py --backfill-gps-exif
```

What it does:

- Sweeps photos with no GPS yet (optionally restricted to a media type with `--gps-media-type image/heic`).
- Re-downloads each original via the bridge and reads the EXIF GPS block.
- Persists the coordinates and runs reverse-geocoding (`reverse_geocoder` library, offline data files) on every photo that has GPS but no place yet.

The `gps` worker loop runs a bounded batch of this backfill automatically each cycle (`--gps-limit 50`), so the backlog drains without saturating your home connection.

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

Most iPhones embed GPS into the EXIF, and the indexer reads it from the original when it downloads the full-res file. If the EXIF was stripped before upload (or the camera never recorded a position), the photo won't appear on the map — it just won't have a place label.

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

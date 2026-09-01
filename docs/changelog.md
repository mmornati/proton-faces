# Changelog

All notable changes to proton-faces are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- **MkDocs documentation site** with dark-slate + amber Material theme, deployed to GitHub Pages ([docs site](https://mmornati.github.io/proton-faces/)).
- **`DEMO_MODE`** (`DEMO_MODE=1`) — runs the full app without a Proton account. Built-in fixture of 82 CC0/Unsplash photos (32 face portraits + 50 picsum scenes), auto-created `demo` admin user, GPS-injected for the places demo.
- **`docker compose --profile demo`** brings up only `indexer-demo` + `app-demo`; the proton-bridge container is skipped.
- **`scripts/fetch-demo-assets.py`** — fetches the demo fixture (idempotent, re-runnable).
- **`scripts/capture_screenshots.py`** — Playwright-driven capture of every UI view (PNG screenshots + MP4 screencasts). Used to generate the docs assets.

### Changed

- `compose.yml` now includes the demo profile services.
- `.env.example` documents `DEMO_MODE` and `DEMO_ADMIN_PASSWORD`.

## 0.9.0 — Multi-user auth (PR-9)

### Added

- Local multi-user accounts (username + bcrypt password).
- Roles: `read`, `write`, `admin`.
- Bearer-token authentication with separate access (8 h) and refresh (30 d) tokens.
- Per-user favorites (`user_favorites` table).
- Admin area (gear-icon modal): overview, health checks, backups, schedule, users.

## 0.8.0 — Three-container split (issue #4)

### Changed

- Recognition pipeline moved to its own `indexer` container.
- `app` container serves the FastAPI API + web UI only by default.
- Single-process layout still available via `RUN_INDEXER=1`.

## 0.7.0 — HEIC + video support

### Added

- Local thumbnail generation for HEIC/HEIF photos (no Proton preview).
- Video poster extraction via ffmpeg; ffprobe-based duration.

## 0.6.0 — Face tagging propagation

### Added

- Naming one face auto-tags every unassigned face with cosine similarity above `FACE_SIM_THRESHOLD`.
- Suggested merges (pairwise cluster similarity ≥ 0.40).

## 0.5.0 — Places

### Added

- GPS reverse-geocoding via `reverse_geocoder`.
- Leaflet world map with clustered city markers.
- Per-person map view.

## 0.4.0 — Search

### Added

- Zero-shot CLIP text search.
- Face search by example (multipart upload).
- Free-text search debounced 250 ms.

## 0.3.0 — People + clustering

### Added

- InsightFace (RetinaFace + ArcFace) for detection + embeddings.
- HDBSCAN clustering, incremental.
- People grid with face-crop covers.
- Rename / merge / unassign.

## 0.2.0 — Indexing pipeline

### Added

- Proton bridge (Bun + Proton SDK).
- Thumbnail downloader.
- Recognition workers.
- SQLite index + numpy vector store.

## 0.1.0 — Initial release

### Added

- FastAPI app + vanilla-JS web UI.
- Three-tab navigation (Photos, Favorites, Archive).
- Date rail, "On this day" strip.
- Stars, archive, hide.

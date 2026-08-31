<div align="center">

# 📸 Proton Faces

**Private, self-hosted face · object · location search for your Proton Drive Photos.**

Everything runs on your own hardware inside Docker — your photos never leave your machine, and
nothing is ever written back to Proton Drive.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Made with FastAPI](https://img.shields.io/badge/made%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![ML: ONNX Runtime](https://img.shields.io/badge/ML-ONNX%20Runtime-000000.svg)](https://onnxruntime.ai)
[![Stack: Docker](https://img.shields.io/badge/stack-Docker-2496ED.svg)](https://www.docker.com)
[![Privacy: no telemetry](https://img.shields.io/badge/privacy-no%20telemetry-brightgreen.svg)](#privacy)

<br>

[![Watch the demo — Proton Faces](https://res.cloudinary.com/blog-mornati-net/video/upload/so_3/v1788111273/aqarlh1taqbygcsode6o.jpg)](https://player.cloudinary.com/embed/?cloud_name=blog-mornati-net&public_id=aqarlh1taqbygcsode6o)

<br>

*Search your whole library like Google Photos — but 100% private and offline. Click the frame to play the demo.*

</div>

---

## ✨ What it does

Proton Photos are **end-to-end encrypted** — so nobody but you (and your own machine) can ever
look at them. That also means *you* have to do the searching. Proton Faces turns your encrypted
photo library into a fully searchable archive, without ever uploading a single byte back:

| Capability | How it works | Status |
|---|---|---|
| 👥 **People** | RetinaFace + ArcFace detect and embed every face; HDBSCAN clusters them into persons you can name | ✅ |
| 🔍 **"Who is this?"** | Drop a photo of a face → find every other photo of the same person | ✅ |
| 🏷️ **Face tagging UX** | Face-crop covers, clickable face boxes on each photo, name one face and **all look-alikes are auto-tagged** | ✅ |
| 📍 **Places** | GPS reverse-geocoding → **interactive world map** with clustered markers (Leaflet + OSM) | ✅ |
| 📝 **Free-text search** | Zero-shot CLIP — type *"dog"*, *"car"*, *"beach"*, *"Lille"* | ✅ |
| 📱 **iPhone (HEIC) photos** | Proton serves no preview → we decode the full-res file locally and generate our own thumbnail | ✅ |
| 🗺️ **Unassigned queue** | Review faces that didn't cluster yet and name them in bulk | ✅ |
| 🎥 **Videos** | Detected and indexed, hidden from photo grids (no preview available) | ✅ |

---

## 🎬 Demo

The video above shows the app in action: people clustering, face tagging, text search,
the places map and photo browsing. The poster frame is served by Cloudinary; clicking it
opens the Cloudinary player (a direct MP4 is also committed at `docs/demo.mp4`).

> **Why a clickable frame and not an inline `<video>`?** GitHub's README sanitizer strips
> `<video>` and `<iframe>` tags that point to external hosts. The only way to embed a video
> that *autoplays inline* on a GitHub README is to drag-and-drop the MP4 into the README
> editor (GitHub then hosts it on `user-attachments`). Want that? Just open the README in
> GitHub's web editor, drag `docs/demo.mp4` onto it, and replace this frame with the
> generated `<video>` snippet. The repo copy is compressed to ~2 MB so it uploads fast.

> **Why MP4 and not a GIF?** The original clip is ~35 MB as video; the repo copy is
> compressed to ~2 MB. As a GIF it would balloon to *hundreds of megabytes* — and GitHub
> refuses to render images over 25 MB. Video is smaller, sharper, and keeps its sound.

---

## 🏗️ How it works

```
                       ┌──────────────────────────────────────────────────┐
                       │                    LOCAL SERVER                   │
                       │                                                  │
  Proton Photos ──────▶│  proton-bridge (Bun + Proton Drive SDK)          │
  (E2E encrypted)      │  • timeline diff (incremental, NDJSON-streamed)  │
                       │  • 512px thumbnail fetch (Type1)                 │
                       │  • full-res download on demand                   │
                       │                    │                             │
                       │                    ▼                             │
                       │  proton-faces (Python + FastAPI)                 │
                       │  • InsightFace (RetinaFace + ArcFace)            │
                       │  • CLIP ViT-B/32 (ONNX Runtime, no PyTorch)      │
                       │  • SQLite + numpy vector store                   │
                       │  • HDBSCAN people clustering                     │
                       │  • reverse-geocoder for place names              │
                       │  • pillow-heif for local HEIC thumbnails         │
                       │  • vanilla-JS web UI on :8080                    │
                       └──────────────────────────────────────────────────┘
```

- The **bridge** authenticates with your existing Proton session and is the **only** component
  that ever talks to Proton. It is **strictly read-only** — no uploads, no writes, no deletions.
- Every photo is processed **once**: thumbnail downloaded (or decoded locally for HEIC) →
  recognition run (faces + CLIP) → small 512px thumbnail cached → original bytes discarded.
- The whole pipeline is **resumable** and runs in the background: add photos to Proton and the
  index catches up automatically.

## Why two containers?

| Container       | Base                              | Role                                                        |
|-----------------|-----------------------------------|-------------------------------------------------------------|
| `proton-bridge` | `oven/bun` + Proton SDK monorepo  | The only component that talks to Proton (auth, thumbnails)  |
| `proton-faces`  | `python:3.11-slim`                | All machine learning, indexing, search API, web UI          |

The published npm `@protontech/drive-sdk` cannot run standalone — its authentication module is
not published — so the bridge is built inside the Proton Drive SDK monorepo at image build time
(pinned tag `cli/v0.8.0`). You never need to clone it yourself; the Docker build does it
automatically.

---

## 🚀 Quick start

### 1. Get a Proton session file

`proton-faces` authenticates with the session file the Proton Drive CLI normally keeps in your
keyring / `pass` store:

```bash
# If you have the CLI session in `pass`:
pass show ch.proton.drive/drive-sdk-cli/auth-session > credentials/auth-session.json
```

> The file contains your account tokens. Keep it private — it is mounted read-only into the
> bridge container and should never be committed to git (see `.gitignore`).

### 2. Configure

```bash
cp .env.example .env
# DATA_DIR=/srv/proton-faces/data   # where thumbnails + indexes live
# PHOTOS_DIR=                        # optional: local Google Takeout export for GPS enrichment
```

### 3. Start

```bash
docker compose up -d
```

Then open **http://localhost:8080**.

Prebuilt images are published to the GitHub Container Registry, so `docker compose up` pulls
them — no building on the server. To build from source instead, use
`docker compose up -d --build`. On a shared box, `scripts/build.sh` shows how to cap BuildKit's
CPU usage so a build never starves your other services.

The indexer starts immediately and is fully resumable. The first run processes your whole
library (roughly 1–2 s per photo on a modern CPU — a 100k-photo library takes about a day), and
the web UI becomes useful right away as results stream in.

> **Two-process layout (issue #4).** `docker compose up` now starts three containers:
> `proton-bridge`, `indexer`, and `app`. The `indexer` container owns the
> recognition pipeline so face detection can't preempt the FastAPI event loop
> on the box's CPU cores. Both processes share the same SQLite index via WAL
> mode and per-row atomic claims — no extra coordination. If you'd rather keep
> the legacy single-process layout (everything in `app`), set `RUN_INDEXER=1`
> in `.env`.

### 👥 Users & roles

proton-faces ships a small multi-user account system. Proton does not publish
an OAuth/OIDC provider, so each family member gets a local account
(username + bcrypt password) instead of "Sign in with Proton". The single
Proton session still drives the library — that lives in the bridge exactly
as before — and every user on your server sees the same library under their
own identity.

| Role | Browse / search / map / albums / places / memories | Per-user ★ favorites | Tags, archive, hide, face naming, person rename/merge |
|------|:-:|:-:|:-:|
| **read**  | ✅ | ✅ | ❌ |
| **write** | ✅ | ✅ | ✅ |
| **admin** | ✅ | ✅ | ✅ + future admin menu (`/api/admin/users/*` for CRUD) |

Anyone on `:8080` must log in. Tokens are bearer JWT-free (opaque random
hex), stored in your browser's `localStorage` under `pf.auth`. Access TTL
defaults to 8 hours, refresh to 30 days (configurable via `AUTH_ACCESS_TTL`
and `AUTH_REFRESH_TTL`).

Other admin tasks (after the first admin is created):

```bash
# Add more users (admin-only endpoint also exists at POST /api/admin/users):
scripts/create-admin.sh dad
scripts/create-admin.sh kid

# Reset a forgotten password (revokes all that user's sessions):
docker compose exec app python main.py --reset-password mom
```

Per-user favorites live in the `user_favorites` table; the legacy global
`photos.favorited` column is kept only as a one-time backfill target the
first time `--create-admin` runs on an existing library.

---

## 🖥️ Usage / web UI

- **Search bar** — free text: *"dog"*, *"car"*, *"beach"*, *"Lille"*. Zero-shot CLIP. **Live
  search-as-you-type** (250 ms debounce) re-runs results as you type.
- **Photos** — infinite grid of cached thumbnails; click any photo for the full-resolution
  download (fetched from Proton on demand, never stored). Videos get a play badge + duration
  overlay and open inline with HTML5 playback + HTTP Range seeking. Above the grid, an **On this
  day** strip resurfaces photos you took on today's date in past years.
- **Favorites** — star any photo (toolbar or per-card button). The Favorites view filters the
  grid to your starred photos.
- **Archive** — hide photos from the main grid without deleting them; visit the Archive tab to
  restore them.
- **Tags** — free-form lowercase labels on any photo. Browse all tags from the Tags tab and
  click one to filter the grid.
- **People** — auto-clustered persons with face-crop avatars. Click a person to see their
  photos, click the name to rename. **Map** plots every geotagged photo of that person on a
  Leaflet map.
- **Face tagging** — every photo shows clickable boxes around detected faces. Name a face and
  **every similar unassigned face is tagged automatically**; assign to an existing person, or
  unassign.
- **Duplicates** — groups photos by content hash; hide any you don't want.
- **Unassigned** — a queue of every face that hasn't been clustered yet, so you can name the
  stragglers.
- **Places** — a **Leaflet world map** with clustered city markers, plus a city list. Click any
  marker to filter the photo grid to that place.
- **Search by example** — drop a photo of a face to find every photo containing that person.

---

## 🔍 Search capabilities

| What you type / do              | Backed by                          | Notes                                    |
|---------------------------------|------------------------------------|------------------------------------------|
| *"Lille"*, *"Paris"*            | GPS reverse-geocoding              | Photos with GPS metadata                 |
| *"dog"*, *"car"*, *"beach"*     | CLIP text–image similarity         | Zero-shot, no training needed            |
| A face photo                    | ArcFace face embeddings            | Returns photos of the same person        |
| Person name (People tab)        | HDBSCAN clusters + your labels     | Cluster is built incrementally           |
| Place marker on the map         | GPS aggregation by place           | Clustered markers, click to filter       |

---

## 📁 Project layout

```
app/              Python indexer + search API + web UI
  src/
    indexer.py    timeline diff / download / recognize / fullres / delete pipeline
    indexer_main.py  dedicated entry point for the `indexer` container (no uvicorn)
    main.py       dedicated entry point for the `app` container (uvicorn only by default)
    faces.py      InsightFace face detection + embeddings
    clip.py       CLIP ViT-B/32 embeddings (ONNX Runtime)
    cluster.py    HDBSCAN people clustering
    geocode.py    offline reverse-geocoding
    store.py      SQLite schema + numpy vector store
    api.py        FastAPI application
    static/       vanilla-JS frontend (incl. Leaflet map)
bridge/           Bun service wrapping the Proton Drive SDK
scripts/          helper scripts (session export, backup, build)
compose.yml       three-container deployment (bridge + indexer + app)
.github/workflows GHCR image publishing
```

---

## ⚙️ Configuration

Environment variables (see `.env.example`):

| Variable              | Default                    | Description                                 |
|-----------------------|----------------------------|---------------------------------------------|
| `DATA_DIR`            | `./data`                   | Persistent data (thumbnails, SQLite, vectors) |
| `PHOTOS_DIR`          | *(empty)*                  | Local Google Takeout export — used to enrich photos with GPS/EXIF without any Proton full-res download |
| `PORT`                | `8080`                     | Web UI port                                 |
| `SYNC_INTERVAL`       | `300`                      | Seconds between timeline diffs              |
| `SYNC_LIMIT`          | `0`                        | Only index the newest N photos (0 = all) — handy for testing |
| `WORKERS`             | `2`                        | Parallel recognition workers                |
| `CLUSTER_INTERVAL`    | `1800`                     | Seconds between people-clustering runs      |
| `GPS_INTERVAL`        | `21600`                    | Seconds between GPS/place enrichment runs   |
| `FACE_SIM_THRESHOLD`  | `0.45`                     | Cosine similarity for auto-tagging faces    |
| `MIN_CLUSTER_SIZE`    | `2`                        | Minimum faces to form a person cluster      |
| `BRIDGE_URL`          | `http://proton-bridge:8090`| Bridge container address                    |
| `MODELS_DIR`          | `DATA_DIR/models`          | Where ML models are stored                  |
| `LOG_LEVEL`           | `INFO`                     | Logging verbosity                           |
| `RUN_INDEXER`         | `0`                        | Set `1` on the `app` container to start the in-process indexer (legacy single-process layout). Default off: the `indexer` container handles the pipeline. |
| `AUTH_ACCESS_TTL`     | `28800`                    | Bearer access-token lifetime (seconds). 8 hours by default. |
| `AUTH_REFRESH_TTL`    | `2592000`                  | Bearer refresh-token lifetime (seconds). 30 days by default. |

### GPS / place enrichment

Proton's API does not expose photo location, but your Google Takeout export keeps GPS in local
`*.supplemental-metadata.json` sidecars. Point `PHOTOS_DIR` at that export and the app will
enrich GPS **automatically** in the background:

- it sha1-hashes the local photo files (cache in `DATA_DIR/gps_sha1_cache.json`, so later runs
  are cheap) and matches against the Proton timeline by content hash — **no full-res download
  ever needed**;
- then reverse-geocodes every photo that has GPS but no place name yet.

Run it manually (e.g. after a fresh import):

```bash
docker compose exec indexer python indexer_main.py --backfill-gps
# after a new Takeout export, force a rehash of the local files:
docker compose exec indexer python indexer_main.py --backfill-gps --rebuild-cache
```

### iPhone (HEIC) photos

Proton serves **no preview** for HEIC/HEIF files. Proton Faces handles them automatically:
the full-resolution file is downloaded once (read-only), decoded locally with `pillow-heif`,
downscaled to a 512px thumbnail, and then processed through the normal pipeline (faces + CLIP).
The full-res bytes are discarded — only the small thumbnail is kept.

---

## 🧭 Roadmap / ideas

- Split people clusters (currently you can merge but not split — manually re-tag works)
- Edit album membership (albums are read-only at present)
- PWA / mobile install

---

## 🔒 Privacy

- **No telemetry, no cloud APIs.** The only network calls go to Proton's servers.
- All ML models run **locally** (ONNX Runtime + CLIP on CPU, no GPU required).
- The bridge is strictly **read-only** against Proton.
- The only files ever kept are the small thumbnails and the index databases in `DATA_DIR`.

---

## 📄 License

[MIT](LICENSE)

## ⚠️ Disclaimer

This project is not affiliated with Proton AG. "Proton", "Proton Drive" and "Proton Photos" are
trademarks of their respective owners. Use at your own risk.
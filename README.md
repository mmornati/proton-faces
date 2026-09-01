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

## 📚 Documentation

The full user guide is published at **[mmornati.github.io/proton-faces](https://mmornati.github.io/proton-faces/)** — search, people, places, admin area, REST API, architecture, every env var.

Highlights:

- [Quickstart](https://mmornati.github.io/proton-faces/getting-started/quickstart/) — 5 minutes from `docker compose up` to your first search
- [Demo mode](https://mmornati.github.io/proton-faces/getting-started/demo-mode/) — run the full app **with zero Proton credentials** (a curated fixture of CC0 photos)
- [People & face tagging](https://mmornati.github.io/proton-faces/user-guide/face-tagging/) — name one face, auto-tag every look-alike
- [REST API](https://mmornati.github.io/proton-faces/reference/api/) — every endpoint
- [Security & privacy](https://mmornati.github.io/proton-faces/reference/security-privacy/) — what's on disk, what's not, how auth works

The docs are rebuilt and published on every push to `main` by the [docs workflow](.github/workflows/docs.yml). To preview locally:

```bash
pip install -r requirements-docs.txt
mkdocs serve         # http://127.0.0.1:8000
```

---

## 🛝 Try the demo

```bash
docker compose --profile demo up -d
```

No Proton account, no session file, no setup. Sign in at http://localhost:8080 with `demo` / `proton-faces`. The app loads a fixture of 82 CC0 photos (32 face portraits + 50 picsum scenes) — every feature of the real app works against it, including face clustering, places, search-by-example, and the admin area. Override the password with `DEMO_ADMIN_PASSWORD=...`. See the [demo-mode guide](https://mmornati.github.io/proton-faces/getting-started/demo-mode/) for details.

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

### Image data flow — local cache vs. live Proton

The **only** thing kept on local disk is a small **512px WebP thumbnail** per photo
(`thumbs/<uid>.webp`, immutable, served by `/api/photos/{uid}/thumb` and reused by
`/api/photos/{uid}/meta`). Full-resolution originals are **never persisted**: every
"View full resolution" click streams the original bytes **live from Proton** through
`/api/photos/{uid}/full` → bridge `/photo/{uid}/full` → Proton Drive SDK, then the
browser. Images stream straight through (no temp file); videos are buffered to a
per-request temp file (`work/<uid>-XXXXXX.full`, deleted as soon as the response ends) so
the HTTP `Range` requests HTML5 `<video>` seeking needs work end-to-end (206 responses).

The indexer additionally uses a throwaway `work/<uid>.download` for photos whose server
preview is missing (HEIC, videos —to extract a poster frame); it is deleted as soon as the
thumbnail exists. Because everything except the 512px thumbnail is fetched on demand, a
slow Proton connection or a dead bridge degrades only the "View full resolution" experience —
thumbing, search, faces, places, and memories all keep working from the local cache.

### Network safety — rate limits, timeouts, capabilities

Proton Drive uses a public HTTPS API. A fresh library sync can produce many
short-lived HTTPS connections per minute from the bridge container. On a
home router or CPE with a small NAT table this can saturate the table and
take other devices on the LAN offline. To make that impossible by default
and tunable when it happens:

- **Outbound rate limit** — `PROTON_BRIDGE_RATE_LIMIT` (default `0`, i.e. **disabled**).
  Set to a positive number of requests-per-second (e.g. `5`) to bound the
  bridge's outbound HTTPS to Proton. `PROTON_BRIDGE_RATE_BURST` (default `2× rate`)
  controls the burst allowance. Honors `Retry-After` on 429 responses.
- **Hard ceiling per full-res download** — `PROTON_BRIDGE_FULL_RES_TIMEOUT_MS`
  (default `300000` = 5 min). Aborts the download and returns 502 if Proton
  hangs (e.g. on an upstream crypto bug). The browser shows a toast instead
  of spinning forever.
- **Manifest verification bypass** — `PROTON_DRIVE_SKIP_MANIFEST_VERIFICATION=1`
  is enabled in the published image because migrated Proton accounts often
  omit the deprecated `AddressKey.PublicKey` field; without this flag the
  SDK hangs on the key-token decrypt (Bun 1.2 WASM OpenPGP issue). Block-level
  SHA256 integrity still applies where available.
- **Container hardening** — `compose.yml` drops all Linux capabilities and sets
  `no-new-privileges` on every service. No container runs with `--network host`
  or `--privileged`; everything goes through the internal `internal` compose
  network and the host's bridge.

If your home router still drops under load, start with `PROTON_BRIDGE_RATE_LIMIT=5`
in `compose.yml` (under `proton-bridge.environment`) and confirm the LAN stays up
while a sync runs.

## Why three containers?

Since [issue #4](https://github.com/mmornati/proton-faces/pull/7) the indexing
pipeline runs in its own container so face detection can't preempt the FastAPI
event loop on the box's CPU cores.

| Container       | Base                              | Role                                                                                                |
|-----------------|-----------------------------------|-----------------------------------------------------------------------------------------------------|
| `proton-bridge` | `oven/bun` + Proton SDK monorepo  | The only component that talks to Proton (auth, thumbnails). Strictly read-only.                      |
| `indexer`       | `python:3.11-slim` + ML models    | Background recognition pipeline (sync / fullres / face detection / CLIP / cluster / GPS). Own entry point, no uvicorn on the host. Also serves an internal status endpoint (see `INDEXER_STATUS_PORT`). |
| `app`           | `python:3.11-slim` + ML models    | FastAPI search API + vanilla-JS web UI on `:8080`. Proxies the indexer's status endpoint for the footer/health modal. |

The `indexer` and `app` containers share the same SQLite index via **WAL mode + a
30 s busy_timeout** and per-row atomic claims (`UPDATE ... WHERE status='new'`),
so they interleave cleanly without any extra coordination. If you'd rather keep
the legacy single-process layout (everything in `app`), set `RUN_INDEXER=1` in
`.env` — the `app` container will then start the indexer threads in-process too.

The published npm `@protontech/drive-sdk` cannot run standalone — its
authentication module is not published — so the bridge is built inside the
Proton Drive SDK monorepo at image build time (pinned tag `cli/v0.8.0`). You
never need to clone it yourself; the Docker build does it automatically.

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
# Inside the containers DATA_DIR is always /data and PHOTOS_DIR is always /takeout.
# To persist data on a specific host disk (e.g. a big volume), set the compose-
# level mount instead — defaults to the named volume "data":
# DATA_MOUNT=/srv/proton-faces/data
# To point indexer/app at a Google Takeout export on the host, set:
# PHOTOS_MOUNT=/srv/photos-takeout
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

> **Three-process layout (issue #4).** `docker compose up` starts three containers:
> `proton-bridge`, `indexer`, and `app` (see [Why three containers?](#why-three-containers)
> above for the rationale). To fall back to the legacy single-process layout (everything
> in `app`), set `RUN_INDEXER=1` in `.env`.

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
| **admin** | ✅ | ✅ | ✅ + admin area (users, backups, schedule, health checks — see below) |

Anyone on `:8080` must log in. Tokens are bearer JWT-free (opaque random
hex), stored in your browser's `localStorage` under `pf.auth`. Access TTL
defaults to 8 hours, refresh to 30 days (configurable via `AUTH_ACCESS_TTL`
and `AUTH_REFRESH_TTL`).

Other admin tasks (after the first admin is created):

```bash
# Add more users (admin-only endpoint also exists at POST /api/admin/users):
scripts/create-admin.sh dad
scripts/create-admin.sh kid --display-name "Kid"
# Non-interactive: ADMIN_PASSWORD=... scripts/create-admin.sh mom
# (the admin endpoint and the inline form in the admin area work too).

# Reset a forgotten password (revokes all that user's sessions):
docker compose exec app python main.py --reset-password mom
```

Per-user favorites live in the `user_favorites` table; the legacy global
`photos.favorited` column is kept only as a one-time backfill target the
first time `--create-admin` runs on an existing library.

---

## 🛠️ Admin area (admin role only)

Log in as an admin and click the **⚙ gear icon** in the top-right of the
header — it only appears for users with the `admin` role. The admin modal
has four tabs:

- **Overview** — server (hostname, Python version, uptime), disk usage of the
  data volume, and last-backup summary.
- **Health checks** — runs seven checks on demand: DB integrity, free disk
  space, backup freshness, backup dir writable, data dir writable, indexer
  liveness, Proton bridge reachability. Each check is an ok/bad pill with a
  short status and detail string.
- **Backups** — list every SQLite snapshot under `DATA_DIR/_backups/`, with
  size and timestamp. Buttons: **Backup now** (creates a fresh
  `index-<UTC-stamp>.sqlite3` via `VACUUM INTO` — consistent, non-blocking
  against the live WAL DB), **Delete** (per row, with confirm), **Prune**
  (keep the newest N per the retention setting). Path traversal is rejected.
- **Schedule** — daily auto-backup inside the app container. **Enabled**,
  **Hour** (0-23 UTC), **Minute** (0-59), **Keep** (1-365). Stored at
  `DATA_DIR/admin_config.json`. The daemon thread wakes every minute and
  runs at most one backup per UTC day to avoid piling up if the app was
  down.
- **Users** — list every account, create new (username ≥ 2 chars, password
  ≥ 8 chars, role `read` / `write` / `admin`), edit display name / role /
  disabled / password, **Logout** (revokes all of that user's sessions),
  **Delete** (refuses to delete the last remaining admin so nobody gets
  locked out).

The user-management section is the same data model exposed by the existing
`/api/admin/users/*` endpoints, now reachable from the UI. No new CLI is
required — everything is in the web admin area.

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
- **Bottom status bar + `?` diagnostics** — every view shows a slim footer with the indexer's live
  state (last sync, pending queue depth, thread liveness). Click the `?` in the header to open the
  full **Status & diagnostics** overlay — server info, runtime counts, and the data behind the
  footer pills (powered by the internal `INDEXER_STATUS_PORT` endpoint; see the config table).

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
    indexer_status.py  tiny in-container HTTP endpoint that exposes the indexer's live state
    main.py       dedicated entry point for the `app` container (uvicorn only by default)
    faces.py      InsightFace face detection + embeddings
    clip.py       CLIP ViT-B/32 embeddings (ONNX Runtime)
    cluster.py    HDBSCAN people clustering
    geocode.py    offline reverse-geocoding
    store.py      SQLite schema + numpy vector store
    auth.py       multi-user account model, bearer-token issuance, role checks
    admin.py      admin-only endpoints + auto-backup worker + health checks
    api.py        FastAPI application
    bridge_client.py  thin HTTP client for the `proton-bridge` service
    config.py     env-driven settings (Settings dataclass)
    static/       vanilla-JS frontend (incl. Leaflet map + admin modal + status overlay)
bridge/           Bun service wrapping the Proton Drive SDK
scripts/          helper scripts (session export, create-admin, backup, build)
compose.yml       three-container deployment (bridge + indexer + app)
.github/workflows GHCR image publishing
```

---

## ⚙️ Configuration

Environment variables (see `.env.example`):

| Variable              | Default                    | Description                                 |
|-----------------------|----------------------------|---------------------------------------------|
| `DATA_MOUNT`          | *(named volume `data`)*   | Compose-level: host path (or `:volume`) the data directory is bind-mounted at. Defaults to Docker's named volume `data`. Set e.g. `DATA_MOUNT=/srv/proton-faces/data` to persist on a big disk. The directory must be writable by UID 1000 (the user all containers run as). |
| `PHOTOS_MOUNT`        | *(unset → `:/dev/null`)*  | Compose-level: optional read-only bind mount of a local Google Takeout export, used to backfill GPS/place data. Inside the indexer/app containers this is always mounted at `/takeout` (and `PHOTOS_DIR` is hardcoded to `/takeout`). To take effect, set e.g. `PHOTOS_MOUNT=/srv/photos-takeout`. |
| `DATA_DIR`            | `/data` (in-container)    | Persistent data (thumbnails, SQLite, vectors). Fixed at `/data` inside the `indexer` and `app` containers; override only for local single-process dev. |
| `PHOTOS_DIR`          | `/takeout` (in-container)  | Optional Google Takeout export mounted at `/takeout` in the indexer/app containers (via `PHOTOS_MOUNT`). Set to anything else only for local dev. |
| `PORT`                | `8080`                     | Web UI port                                 |
| `SYNC_INTERVAL`       | `300`                      | Seconds between timeline diffs              |
| `SYNC_LIMIT`          | `0`                        | Only index the newest N photos (0 = all) — handy for testing |
| `WORKERS`             | `2`                        | Parallel recognition workers                |
| `CLUSTER_INTERVAL`    | `1800`                     | Seconds between people-clustering runs      |
| `GPS_INTERVAL`        | `21600`                    | Seconds between GPS/place enrichment runs   |
| `FACE_SIM_THRESHOLD`  | `0.45`                     | Cosine similarity for auto-tagging faces    |
| `MIN_CLUSTER_SIZE`    | `2`                        | Minimum faces to form a person cluster      |
| `BRIDGE_CACHE_STALE_SEC` | `21600` (6 h)          | Age (seconds) at which the on-disk Proton SDK cache is flagged "stale" by the admin Server-checks panel. Only fires when full-res downloads are also failing (the getFileDownloader hang signature). |
| `BRIDGE_URL`          | `http://proton-bridge:8090`| Bridge container address                    |
| `MODELS_DIR`          | `DATA_DIR/models`          | Where ML models are stored                  |
| `LOG_LEVEL`           | `INFO`                     | Logging verbosity                           |
| `RUN_INDEXER`         | `0`                        | Set `1` on the `app` container to start the in-process indexer (legacy single-process layout). Default off: the `indexer` container handles the pipeline. |
| `INDEXER_STATUS_PORT` | `8091`                     | Internal-only port on the `indexer` container (bound to `0.0.0.0`, no host port published in compose) that exposes its live runtime state to the `app` container for the footer/health modal. Reachable only over the compose network via the `indexer` hostname (`http://indexer:8091`). |
| `INDEXER_STATUS_URL`  | `http://indexer:8091`      | Base URL the `app` container reads to proxy the indexer's status endpoint. Override to `http://127.0.0.1:8091` for local single-process dev with `RUN_INDEXER=1`. |
| `AUTH_ACCESS_TTL`     | `28800`                    | Bearer access-token lifetime (seconds). 8 hours by default. |
| `AUTH_REFRESH_TTL`    | `2592000`                  | Bearer refresh-token lifetime (seconds). 30 days by default. |
| `ADMIN_PASSWORD`      | *(unset → prompt on stdin)*| If set, `python main.py --create-admin USER` and `--reset-password USER` read the password from this env var instead of prompting (so you can run them non-interactively from a script or init container). At least 8 characters or the command rejects the password. |

### GPS / place enrichment

Proton's API does not expose photo location, but your Google Takeout export keeps GPS in local
`*.supplemental-metadata.json` sidecars. Inside the containers the export is mounted at
`/takeout`; on the host you point compose at the export with the **compose-level**
`PHOTOS_MOUNT` variable (see the config table). Once mounted, the app will
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
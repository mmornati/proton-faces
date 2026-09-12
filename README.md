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

## Why it exists

Proton Photos are **end-to-end encrypted** — so nobody but you (and your own machine) can ever
look at them. That also means *you* have to do the searching. Proton Faces turns your encrypted
photo library into a fully searchable archive — people, places, objects, free text — without
ever uploading a single byte back. The index and every ML model run on your own hardware; the
only component that ever talks to Proton is a strictly **read-only** bridge.

---

## Screenshots

![Photos grid](docs/assets/screenshots/photos.png)
![Places map](docs/assets/screenshots/places.png)
![People](docs/assets/screenshots/people.png)
![Face tagging](docs/assets/screenshots/face-tagging.png)
![Free-text search](docs/assets/screenshots/search-dog.png)

---

## Features

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
| 🔐 **2FA (TOTP)** | Optional per-user two-factor auth with any authenticator app | ✅ |
| 🖼️ **Person cover photos** | Pick which face-crop is the "main photo" on each People card | ✅ |
| 🔀 **Suggested merges** | Review-and-select UX for merging look-alike person clusters | ✅ |
| 📱 **Mobile & PWA** | Installable, offline-capable app shell with light/dark themes | ✅ |

---

## Quick start

### 1. Get a Proton session file

`proton-faces` authenticates with a session from the official **Proton Drive CLI** — there is no
OAuth. Install the CLI from the [Proton Drive CLI download page](https://proton.me/download/drive/cli),
sign in once, and export the session for the bridge:

```bash
proton-drive auth login
scripts/export-session.sh    # writes credentials/auth-session.json
```

> The file contains your account tokens — treat it like a password. It is mounted **only** into
> the `proton-bridge` container, and `.gitignore` already excludes `credentials/`. See the
> [session-export guide](https://mmornati.github.io/proton-faces/getting-started/session-export/)
> for GPG-encrypting it at rest.

### 2. Configure

```bash
cp .env.example .env
```

### 3. Start

```bash
docker compose up -d
```

Prebuilt images are published to the GitHub Container Registry, so `docker compose up` pulls
them — no building on the server. To build from source instead, use
`docker compose up -d --build`.

### 4. Create your admin account

```bash
scripts/create-admin.sh admin          # prompts for a password, or:
ADMIN_PASSWORD=... scripts/create-admin.sh admin
```

### 5. Open the app

Browse to **http://localhost:8080** and sign in. The indexer starts immediately and is fully
resumable — the web UI becomes useful right away as results stream in.

> **Three-process layout.** `docker compose up` starts three containers: `proton-bridge`,
> `indexer`, and `app`. To fall back to the legacy single-process layout (everything in `app`),
> set `RUN_INDEXER=1` in `.env`.

---

## 🛝 Try the demo

```bash
docker compose --profile demo up -d
```

No Proton account, no session file, no setup. Sign in at http://localhost:8080 with `demo` / `proton-faces`.
The app loads a fixture of 82 CC0 photos (32 face portraits + 50 picsum scenes) — every feature
of the real app works against it, including face clustering, places, search-by-example, and the
admin area. Override the password with `DEMO_ADMIN_PASSWORD=...`. See the
[demo-mode guide](https://mmornati.github.io/proton-faces/getting-started/demo-mode/) for details.

---

## 🧠 Tech highlights

- **Three containers, one job each.** `proton-bridge` (Bun + Proton Drive SDK) is the *only*
  component that talks to Proton — strictly read-only. `indexer` runs the recognition pipeline
  (sync / fullres / face detection / CLIP / cluster / GPS) on its own cores. `app` serves the
  FastAPI search API and the vanilla-JS web UI on `:8080`.
- **No GPU required.** All ML runs locally on CPU: InsightFace (RetinaFace + ArcFace) for faces,
  CLIP ViT-B/32 via ONNX Runtime for free-text search, HDBSCAN for people clustering, offline
  reverse-geocoding for places.
- **SQLite is the only datastore.** The `indexer` and `app` containers share the index via WAL
  mode + a 30 s busy_timeout and per-row atomic claims — no extra coordination needed.
- **Nothing is ever written back to Proton.** Only a small 512px WebP thumbnail per photo is
  kept on disk; full-resolution originals stream live from Proton on demand and are never
  persisted.
- **Safe by default.** Binary photo endpoints require a bearer token or signed URL, bcrypt cost
  12, refresh tokens rotate, login is rate-limited, and every container drops Linux
  capabilities.

See the [architecture](https://mmornati.github.io/proton-faces/reference/architecture/) and
[security & privacy](https://mmornati.github.io/proton-faces/reference/security-privacy/) docs
for the deep dive.

---

## 📚 Documentation

The full user guide is published at **[mmornati.github.io/proton-faces](https://mmornati.github.io/proton-faces/)** — search, people, places, admin area, REST API, architecture, every env var.

Highlights:

- [Quickstart](https://mmornati.github.io/proton-faces/getting-started/quickstart/) — install to your first real search in ~10 minutes
- [Demo mode](https://mmornati.github.io/proton-faces/getting-started/demo-mode/) — run the full app **with zero Proton credentials**
- [People & face tagging](https://mmornati.github.io/proton-faces/user-guide/face-tagging/) — name one face, auto-tag every look-alike
- [REST API](https://mmornati.github.io/proton-faces/reference/api/) — every endpoint
- [Configuration](https://mmornati.github.io/proton-faces/reference/configuration/) — every environment variable
- [Security & privacy](https://mmornati.github.io/proton-faces/reference/security-privacy/) — what's on disk, what's not, how auth works

The docs are rebuilt and published on every push to `main` by the [docs workflow](.github/workflows/docs.yml). To preview locally:

```bash
pip install -r requirements-docs.txt
mkdocs serve         # http://127.0.0.1:8000
```

---

## 🧭 Roadmap / ideas

- Split people clusters (currently you can merge but not split — manually re-tag works)
- Edit album membership (albums are read-only at present)

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

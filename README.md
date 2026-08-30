# Proton Faces

Private, self-hosted face / object / location search over your Proton Drive **Photos**
library. Everything runs on your own hardware inside Docker — your photos never leave
your machine, and nothing is ever written back to Proton Drive.

It works exactly like the search bar of a photo service (people, places, objects),
but for your Proton Photos, fully offline:

- **People** — automatic face detection + clustering into persons you can name, plus
  "who is this?" search by example photo.
- **Objects & scenes** — free-text search ("dog", "car", "beach", "Lille") using CLIP.
- **Places** — reverse-geocoded location search for photos that carry GPS metadata.
- **Zero download for browsing** — a tiny 512px thumbnail is cached locally for every
  photo; browsing the results never touches Proton again. Full-resolution preview is
  only fetched on demand when you click a photo.

## How it works

```
                         ┌────────────────────────────────────────────┐
                         │                 LOCAL SERVER                │
                         │                                            │
  Proton Photos  ───────▶│   proton-bridge (Bun + Proton Drive SDK)   │
  (E2E encrypted)        │   • timeline diff                          │
                         │   • 512px thumbnail fetch (Type1)          │
                         │   • full-res download on demand            │
                         │                    │                       │
                         │                    ▼                       │
                         │   proton-faces (Python + FastAPI)          │
                         │   • InsightFace (RetinaFace + ArcFace)     │
                         │   • OpenCLIP (ViT-B/32) for object search  │
                         │   • SQLite + numpy vector store            │
                         │   • HDBSCAN people clustering              │
                         │   • vanilla-JS web UI on :8080             │
                         └────────────────────────────────────────────┘
```

Proton Photos are end-to-end encrypted. The **bridge** uses the official
[`@protontech/drive-sdk`](https://github.com/ProtonDriveApps/sdk) (via the Proton Drive
CLI machinery) to authenticate with your existing session, download thumbnails, and
decrypt them locally. Every photo is processed **once** (thumbnail downloaded →
recognition run → thumbnail cached → original file never kept), and the pipeline is
**fully read-only** against Proton — no uploads, no writes, no deletions.

## Why two containers?

| Container       | Base                        | Role                                                        |
|-----------------|-----------------------------|-------------------------------------------------------------|
| `proton-bridge` | `oven/bun` + Proton SDK repo| The only component that talks to Proton (auth, thumbnails)  |
| `proton-faces`  | `python:3.12-slim`          | All machine learning, indexing, search API, web UI          |

The bridge wraps the Proton Drive SDK — the published npm `@protontech/drive-sdk` cannot
run standalone because its authentication module is not published, so the bridge is
built inside the Proton Drive SDK monorepo at image build time (pinned tag `cli/v0.8.0`).
You never need to clone it yourself; it happens automatically in the Docker build.

## Requirements

- A Proton account with Photos enabled
- An existing Proton Drive CLI session (or the ability to log in once)
- A machine with Docker + Docker Compose
  - 4+ CPU cores recommended for reasonable indexing speed (no GPU needed)
  - Enough disk for the thumbnail cache (~30–60 KB × your photo count)

## Quick start

### 1. Get a Proton session file

`proton-faces` authenticates with a session file that the Proton Drive CLI normally keeps
in your keyring / `pass` store. Export it to a file:

```bash
# If you have the CLI session in `pass`:
pass show ch.proton.drive/drive-sdk-cli/auth-session > credentials/auth-session.json
```

> The file contains your account tokens. Keep it private — it is mounted read-only into
> the bridge container and should never be committed to git (see `.gitignore`).

### 2. Configure

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
# DATA_DIR=/srv/proton-faces/data   # where thumbnails + indexes live
# PHOTOS_DIR=                        # optional: local Takeout export for GPS enrichment
```

### 3. Start

```bash
docker compose up -d --build
```

Then open http://localhost:8080.

The indexer starts immediately. The first run processes your whole library and takes
some time (roughly 1–2 s per photo on a modern CPU — a 100k-photo library takes about a
day, fully resumable). The web UI becomes useful right away as results stream in.

## Usage / web UI

- **Search bar** — type free text ("dog", "car", "beach"). Results appear as they are
  indexed.
- **People** — detected faces are auto-clustered; click a person to see their photos,
  click the name to rename them.
- **Search by example** — drop a photo of a face to find every photo containing that
  person.
- Click any result to load the full-resolution photo from Proton on demand.

## Search capabilities

| What you type / do          | Backed by                          | Notes                                        |
|-----------------------------|------------------------------------|----------------------------------------------|
| "Lille", "Paris"            | GPS reverse-geocoding              | Only for photos with GPS metadata            |
| "dog", "car", "beach"       | CLIP text–image similarity         | Zero-shot, no training needed                |
| A face photo                | ArcFace face embeddings            | Returns photos of the same person            |
| Person name (people tab)    | HDBSCAN clusters + your labels     | Cluster is built incrementally               |

## Project layout

```
app/              Python indexer + search API + web UI
  src/
    bridge.py     HTTP client for the bridge container
    indexer.py    timeline diff / download / recognize / delete pipeline
    faces.py      InsightFace face detection + embeddings
    clip.py       OpenCLIP image + text embeddings
    cluster.py    HDBSCAN people clustering
    store.py      SQLite schema + numpy vector store
    api.py        FastAPI application
    static/       vanilla-JS frontend
bridge/           Bun service wrapping the Proton Drive SDK
scripts/          helper scripts (session export, backup, one-off commands)
compose.yml       two-container deployment
```

## Configuration

Environment variables (see `.env.example`):

| Variable                | Default                 | Description                                  |
|-------------------------|-------------------------|----------------------------------------------|
| `DATA_DIR`              | `./data`                | Persistent data (thumbnails, SQLite, vectors)|
| `PHOTOS_DIR`            | *(empty)*               | Local Google Takeout export — used to enrich |
|                         |                         | photos with GPS/EXIF data without any Proton |
|                         |                         | full-res download                            |
| `PORT`                  | `8080`                  | Web UI port                                  |
| `SYNC_INTERVAL`         | `300`                   | Seconds between timeline diffs               |
| `WORKERS`               | `2`                     | Parallel recognition workers                 |
| `BRIDGE_URL`            | `http://proton-bridge:8090` | Bridge container address                  |
| `SYNC_LIMIT`            | `0`                     | Only index the newest N photos (0 = all) — handy for testing |

### GPS / place enrichment

Photos imported from a Google Takeout export keep their GPS metadata in local
`*.supplemental-metadata.json` sidecars (Proton's API does not expose location).
If you still have that export on disk, point `PHOTOS_DIR` at it and run the
one-shot backfill inside the app container (matches by sha1 against the Proton
timeline, so no full-res download is ever needed):

```bash
docker compose exec app python main.py --backfill-gps
```

## Roadmap / ideas

- Face comparison confidence tuning per library
- Album-aware browsing
- Background full-res GPS enrichment (one full download per photo, then delete)

## Privacy

- No telemetry, no cloud APIs. The only network calls go to Proton's servers.
- The ML models run locally (ONNX + OpenCLIP on CPU).
- The bridge is strictly read-only. The only files ever kept are the small thumbnails
  and the index databases in `DATA_DIR`.

## License

[MIT](LICENSE)

## Disclaimer

This project is not affiliated with Proton AG. "Proton", "Proton Drive" and "Proton Photos"
are trademarks of their respective owners. Use at your own risk.
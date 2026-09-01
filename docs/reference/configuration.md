# Configuration

Proton Faces is configured entirely through environment variables. The `app` and `indexer` containers share most of them; the bridge has its own (set in `compose.yml`, mostly Proton SDK env vars).

## Compose-level (host-side)

These are read by `docker-compose.yml` itself, not by the containers:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_MOUNT` | named volume `data` | Host path or `:volume` that backs `DATA_DIR`. On a server with a big disk, set `DATA_MOUNT=/srv/proton-faces/data`. The directory must be writable by UID 1000. |
| `PHOTOS_MOUNT` | `/dev/null` | Optional read-only bind mount of a local Google Takeout export. Inside the indexer/app containers it's always mounted at `/takeout` (and `PHOTOS_DIR` is hardcoded to `/takeout`). Set e.g. `PHOTOS_MOUNT=/srv/photos-takeout` to enable GPS backfill. |
| `PORT` | `8080` | Host port the web UI binds to. |
| `SYNC_INTERVAL` | `300` | Default seconds between timeline diffs. |
| `SYNC_LIMIT` | `0` | Default sync limit (0 = all photos). |
| `WORKERS` | `3` | Default number of recognition workers. |
| `CLUSTER_INTERVAL` | `1800` | Default seconds between people-clustering runs. |
| `GPS_INTERVAL` | `21600` | Default seconds between GPS/place enrichment runs. |
| `FACE_SIM_THRESHOLD` | `0.45` | Default cosine similarity threshold for face auto-tagging. |
| `MIN_CLUSTER_SIZE` | `2` | Default minimum faces to form a person cluster. |
| `LOG_LEVEL` | `INFO` | Default logging verbosity. |
| `DEMO_ADMIN_PASSWORD` | `proton-faces` | Password for the auto-created `demo` admin in demo mode. |
| `RUN_INDEXER` | `0` | Set `1` on the `app` service to start the in-process indexer (legacy single-process layout). |
| `INDEXER_STATUS_PORT` | `8091` | Internal-only port on the `indexer` container for its `/status` endpoint. |
| `BRIDGE_CACHE_STALE_SEC` | `21600` (6 h) | Age (seconds) at which the on-disk Proton SDK cache is flagged "stale" by the admin Server-checks panel. Only fires when full-res downloads are also failing. Lower it to surface stale caches earlier; raise it if your bridge is idle for longer than 6 h. |

## In-container

Set inside `compose.yml` for each service. Most match the compose-level defaults above; the in-container values are set explicitly so the single-process dev path (`python main.py` on the host) works without docker compose.

| Variable | Default in `app`/`indexer` | Description |
|----------|----------------------------|-------------|
| `DATA_DIR` | `/data` (in-container) | Persistent data (thumbnails, SQLite, vectors). Hardcoded to `/data` in both containers; override only for local single-process dev. |
| `PHOTOS_DIR` | `/takeout` (in-container) | Optional Google Takeout export mounted at `/takeout`. |
| `MODELS_DIR` | `/models` (in-container) | Where the ML models live (baked into the image). Override only for testing. |
| `BRIDGE_URL` | `http://proton-bridge:8090` | Bridge container address (internal docker network). |
| `PORT` | `8080` (in-container) | Web UI port inside the `app` container. |
| `INDEXER_STATUS_URL` | `http://indexer:8091` | Where the `app` container reads the indexer's status from. For local single-process dev with `RUN_INDEXER=1`, override to `http://127.0.0.1:8091`. |
| `RUN_INDEXER` | `0` | Set `1` on the `app` container to start the in-process indexer. |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR`. |
| `DEMO_MODE` | unset | `1` enables demo mode (no real Proton account). Set automatically by the `demo` compose profile. |
| `DEMO_ADMIN_PASSWORD` | unset → prompt | Override the demo admin password. |

## Auth

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_ACCESS_TTL` | `28800` (8 hours) | Bearer access-token lifetime in seconds. Set `0` for effectively no expiry during a session (not recommended). |
| `AUTH_REFRESH_TTL` | `2592000` (30 days) | Bearer refresh-token lifetime in seconds. |
| `ADMIN_PASSWORD` | unset → prompt | Pre-set the first admin's password so `--create-admin` runs non-interactively (e.g. from an init container). At least 8 characters or the command rejects it. |

## Performance tuning

A few knobs are worth knowing for large libraries:

```bash
# More parallel recognition workers (capped at #cores - 1)
WORKERS=6

# Less frequent clustering (cheaper if you don't add photos often)
CLUSTER_INTERVAL=3600

# More aggressive face auto-tagging
FACE_SIM_THRESHOLD=0.40

# More frequent sync (faster pick-up of new photos, more API calls)
SYNC_INTERVAL=120
```

The default `WORKERS=3` keeps one core free for the API event loop on a 4-core box. On a 6-core box, `WORKERS=5` is a reasonable ceiling.

## Bridge-only env vars

The `proton-bridge` container reads these (set in `compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8090` | Bridge container port (internal only). |
| `DATA_DIR` | `/data` | Where the bridge caches the session + work files. |
| `PROTON_DRIVE_CREDENTIALS_STORE` | `unsafe_file` | Required by the SDK to use a file-based session. |
| `PROTON_DRIVE_CACHE_DIR` | `/data` | Where the SDK caches encrypted blobs. |
| `PROTON_DRIVE_BASE_URL` | `drive-api.proton.me` | Optional: point at a custom Proton API environment. |

## Local dev (single-process)

For `python main.py` on the host:

```bash
DATA_DIR=./data \
MODELS_DIR=./data/models \
BRIDGE_URL=http://127.0.0.1:8090 \
PHOTOS_DIR= \
LOG_LEVEL=DEBUG \
RUN_INDEXER=1 \
INDEXER_STATUS_URL=http://127.0.0.1:8091 \
ADMIN_PASSWORD=changeme \
python -m main --create-admin admin
```

Then open http://localhost:8080.

## Verifying the config

The `?` overlay in the footer shows every relevant env var (under **Config**). The admin **Overview** tab shows the running version and the values that affect runtime behavior.

## Where to set things

- **In `.env`** for compose-level vars (or pass them inline: `KEY=value docker compose up -d`).
- **In `compose.yml`** for per-service overrides (don't edit `compose.yml` directly — use a `compose.override.yml`).
- **In `docker-compose.local.yml`** for dev-only tweaks (mount source, expose ports, etc.).

---

**Next:** [REST API](api.md) lists every endpoint.

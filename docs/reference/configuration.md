# Configuration

Proton Faces is configured entirely through environment variables. The `app` and `indexer` containers share most of them; the bridge has its own (set in `compose.yml`, mostly Proton SDK env vars).

## Compose-level (host-side)

These are read by `docker-compose.yml` itself, not by the containers:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_MOUNT` | named volume `data` | Host path or `:volume` that backs `DATA_DIR`. On a server with a big disk, set `DATA_MOUNT=/srv/proton-faces/data`. The directory must be writable by UID 1000. |
| `AUTH_SESSION_MOUNT` | `./credentials/auth-session.json` | Host path to the Proton session file (issue #32). The file is bind-mounted **only into the proton-bridge container** at `/data/auth-session.json` — the indexer and `app` containers have no path to it. The mount is writable because the SDK refreshes the file on disk. Override if your session lives elsewhere; the directory must be readable (and writable for refresh) by UID 1000. |
| `PORT` | `8080` | Host port the web UI binds to. |
| `SYNC_INTERVAL` | `300` | Default seconds between timeline diffs. |
| `SYNC_LIMIT` | `0` | Default sync limit (0 = all photos). |
| `WORKERS` | `3` | Default number of recognition workers. |
| `CLUSTER_INTERVAL` | `1800` | Default seconds between people-clustering runs. |
| `CLUSTER_MAX_FACES` | `5000` | Max unassigned faces per clustering run (0 = no cap). |
| `GPS_INTERVAL` | `21600` | Default seconds between GPS/place enrichment runs. |
| `ALBUM_FULL_RESCAN_SEC` | `21600` (6 h) | How often (seconds) the albums sync runs a full photos-table rescan as a repair pass. Between rescans the sync is incremental (issue #95): only albums whose membership changed are recounted. Lower it to converge out-of-band edits faster at the cost of more scanning. |
| `FACE_SIM_THRESHOLD` | `0.45` | Default cosine similarity threshold for face auto-tagging. |
| `MIN_CLUSTER_SIZE` | `2` | Default minimum faces to form a person cluster. |
| `LOG_LEVEL` | `INFO` | Default logging verbosity. |
| `DEMO_ADMIN_PASSWORD` | `proton-faces` | Password for the auto-created `demo` admin in demo mode. |
| `RUN_INDEXER` | `0` | Set `1` on the `app` service to start the in-process indexer (legacy single-process layout). |
| `INDEXER_STATUS_PORT` | `8091` | Internal-only port on the `indexer` container for its `/status` endpoint. |
| `BRIDGE_CACHE_STALE_SEC` | `21600` (6 h) | Age (seconds) at which the on-disk Proton SDK cache is flagged "stale" by the admin Server-checks panel. Only fires when full-res downloads are also failing. Lower it to surface stale caches earlier; raise it if your bridge is idle for longer than 6 h. |
| `ORT_INTRA_OP_THREADS` | `1` | Threads per ONNX Runtime session (CLIP). Default `1` prevents the indexer's workers + CLIP sessions from oversubscribing the CPU; raise only on a box with spare cores. `OMP_NUM_THREADS=1` is set in compose for the OpenMP-backed kernels (insightface). |
| `CLIP_BATCH_SIZE` | `4` | Max photos embedded per CLIP session.run in the indexer when the pending queue is deep (issue #98). With `ORT_INTRA_OP_THREADS=1`, batch>1 reuses per-core work and raises indexer throughput during a bulk backlog at the same CPU quota. Set `1` to disable batching. |
| `CLIP_BATCH_QUEUE_DEPTH` | `8` | Pending-queue depth at which indexer workers switch from batch=1 to `CLIP_BATCH_SIZE` micro-batching. Below it (typical for interactive reclaims) workers embed one photo at a time to keep first-result latency low. |
| `APP_MEM_LIMIT` | `6g` | Memory cap for the `app` container. Each uvicorn worker loads its own CLIP ONNX session (~838 MB RSS measured) + a per-worker matrix cache; 2 workers fit comfortably under 6 GiB. Lower it on small boxes if you also lower `UVICORN_WORKERS`. |
| `APP_CPUS` | `2.0` | CPU cap for the `app` container, so CLIP/search can't starve the other containers on the host. |

## In-container

Set inside `compose.yml` for each service. Most match the compose-level defaults above; the in-container values are set explicitly so the single-process dev path (`python main.py` on the host) works without docker compose.

| Variable | Default in `app`/`indexer` | Description |
|----------|----------------------------|-------------|
| `DATA_DIR` | `/data` (in-container) | Persistent data (thumbnails, SQLite, vectors). Hardcoded to `/data` in both containers; override only for local single-process dev. |
| `MODELS_DIR` | `/models` (in-container) | Where the ML models live (baked into the image). Override only for testing. |
| `BRIDGE_URL` | `http://proton-bridge:8090` | Bridge container address (internal docker network). |
| `PORT` | `8080` (in-container) | Web UI port inside the `app` container. |
| `INDEXER_STATUS_URL` | `http://indexer:8091` | Where the `app` container reads the indexer's status from. For local single-process dev with `RUN_INDEXER=1`, override to `http://127.0.0.1:8091`. |
| `RUN_INDEXER` | `0` | Set `1` on the `app` container to start the in-process indexer. |
| `UVICORN_WORKERS` | `2` | Number of uvicorn workers serving the API. Each worker lazily loads its own CLIP ONNX session (~838 MB RSS) + a per-worker matrix cache, so this is the dominant term in the `app` container's memory footprint. Keep low on memory-constrained hosts; raise on memory-rich ones. Ignored when `RUN_INDEXER=1` (single-process mode). |
| `WARM_MODELS` | `1` | Pre-load the CLIP + InsightFace sessions in every uvicorn worker at startup (lifespan hook, issue #97) so the first `/api/search` and `/api/search/face` after a deploy/restart don't pay a multi-second model load inside a user request. Models are baked into the image and warmed at build time; this re-applies the per-process load in each worker. Set `0` to defer loading to the first request (tests/CLI use `0`). A failed warm-up is logged and non-fatal — the lazy path remains the fallback. |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR`. |
| `DEMO_MODE` | unset | `1` enables demo mode (no real Proton account). Set automatically by the `demo` compose profile. |
| `DEMO_ADMIN_PASSWORD` | unset → prompt | Override the demo admin password. |
| `DEMO_LOGIN_LOGS` | `0` | Log the demo admin password source at WARN on first boot. The password itself is NEVER logged regardless of this flag. Set `1` to log whether the password came from `DEMO_ADMIN_PASSWORD` or the default. |
| `DEMO_ALLOW_PUBLIC_THUMBS` | `0` | Binary media endpoints (`/thumb`, `/full`, `/cover`, `/crop`) are **secure by default**: they require a bearer token or a short-lived signed URL from `/api/sign`. Set `1` to make them world-readable (the `demo` compose profile does this so static `<img>` tags keep working). |

## Auth

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGNING_SECRET` | required (prod) | HMAC key for the short-lived signed URLs on binary media endpoints (`/thumb`, `/full`, `/cover`, `/crop`). **Required outside `DEMO_MODE`** — the app refuses to start without it (`compose.yml` fails the `app` service with `${SIGNING_SECRET:?…}`), so a known default can't be used to forge signed URLs and signed URLs stay valid across all uvicorn workers (default 2, see `UVICORN_WORKERS`). Generate with `openssl rand -hex 32`. In `DEMO_MODE` the `app-demo` compose service sets a fixed `demo-signing-secret` (demo content is public by design; the code also allows an ephemeral per-boot fallback there). |
| `INDEXER_TOKEN` | required (prod) | Shared secret the `app` container sends in the `X-Indexer-Token` header when proxying to the `indexer` container's control API on `:8091` (`/status`, `/trigger-sync`, `/sync-config` — `/healthz` is intentionally auth-free for compose healthchecks). **Required outside `DEMO_MODE`** — the indexer refuses to start without it (`compose.yml` fails both `indexer` and `app` services with `${INDEXER_TOKEN:?…}`), so the control API on the compose internal network cannot be triggered by any container that doesn't already know the secret. The `app` and `indexer` services share the same env var in `compose.yml` so they cannot drift. Generate with `openssl rand -hex 32`. In `DEMO_MODE` the `indexer-demo`/`app-demo` compose services allow an unset token (each falls back to a per-boot ephemeral value and logs a warning); operators that want stable across-restart auth can still set `INDEXER_TOKEN` explicitly. |
| `AUTH_ACCESS_TTL` | `28800` (8 hours) | Bearer access-token lifetime in seconds. Set `0` for effectively no expiry during a session (not recommended). |
| `AUTH_REFRESH_TTL` | `2592000` (30 days) | Bearer refresh-token lifetime in seconds. |
| `AUTH_2FA_PENDING_TTL` | `300` (5 minutes) | How long a pending 2FA token stays valid (seconds) during the two-step login. After this window the user must re-enter username + password. |
| `AUTH_COOKIE_SECURE` | `0` | FP-1: set the `Secure` flag on the HttpOnly refresh-token cookie (`pf_refresh`). Default `0` so local `http://` testing and the demo work; set `1` behind TLS (the `app` compose service does). A `Secure` cookie is only sent over HTTPS, so leaving it `0` behind TLS is a regression of the XSS-exfiltration fix. |
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
| `BRIDGE_HOST` | `0.0.0.0` | Bridge bind address. Set to `127.0.0.1` when running outside Docker to avoid LAN exposure (issue #43). |
| `DATA_DIR` | `/data` | Where the bridge caches the session + work files. |
| `PROTON_DRIVE_CREDENTIALS_STORE` | `unsafe_file` | Session storage backend. `unsafe_file` (default) keeps the plaintext session at `/data/auth-session.json` (the `AUTH_SESSION_MOUNT` bind mount). `pass` selects the SDK's encrypted store: the bridge image ships `pass` + `gnupg`, and its entrypoint generates a container-local GPG key (in the `bridge-gnupg` volume), initializes the store (in the `bridge-pass-store` volume), and migrates an existing plaintext session on first start. See [Session file → Encrypted store](../getting-started/session-export.md#encrypted-store-pass). Live only since PR #129 — earlier compose files hardcoded `unsafe_file`. |
| `PROTON_DRIVE_CACHE_DIR` | `/data` | Where the SDK caches encrypted blobs. |
| `PROTON_DRIVE_BASE_URL` | `drive-api.proton.me` | Optional: point at a custom Proton API environment. |
| `PROTON_BRIDGE_RATE_LIMIT` | `0` (image default; `3` in `compose.yml`) | Sustained outbound requests/second at the *operation* layer: one token per bridge operation start (timeline sync, node listing, album sync, thumbnail batch, full-res download). Honors `Retry-After` on 429 responses. |
| `PROTON_BRIDGE_RATE_BURST` | `Math.max(1, ceil(rate × 2))` | Burst allowance for the operation-layer bucket (`2` in `compose.yml`). |
| `PROTON_BRIDGE_RATE_LIMIT_HTTP` | derived: `RATE_LIMIT × 10` | Requests/second at the SDK's HTTP transport — the patched `HTTPClient` acquires a token before **every** upstream HTTPS call (paginated listings, block downloads, thumbnail requests), not just operation starts (issue #44). Unset/empty derives as `PROTON_BRIDGE_RATE_LIMIT × 10` (default compose: 3 ops/s → 30 HTTP req/s); explicit `0` disables this layer. Also honors `Retry-After` on 429/503. |
| `PROTON_BRIDGE_FULL_RES_TIMEOUT_MS` | `300000` | Full-res download queue-slot clamp and the `/full` MIME-lookup deadline (issue #55). The client can further clamp via the `X-Timeout-Ms` header. |
| `PROTON_BRIDGE_TIMELINE_TIMEOUT_MS` | `1800000` | Deadline for a full `/timeline` sync (pagination + node-key decrypt) (issue #55). |
| `PROTON_BRIDGE_NODES_TIMEOUT_MS` | `300000` | Deadline for the `/nodes` metadata lookup (issue #55). |
| `PROTON_BRIDGE_ALBUMS_TIMEOUT_MS` | `300000` | Deadline for the `/albums` listing (issue #55). |
| `PROTON_BRIDGE_THUMBNAILS_TIMEOUT_MS` | `300000` | Deadline for a `/thumbnails` batch download (issue #55). |

## Local dev (single-process)

For `python main.py` on the host:

```bash
DATA_DIR=./data \
MODELS_DIR=./data/models \
BRIDGE_URL=http://127.0.0.1:8090 \
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

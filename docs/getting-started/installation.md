# Installation

Proton Faces ships as a Docker Compose stack of three services: `proton-bridge`, `indexer`, and `app`. You can also run it as a single process for local development or run the demo without any Proton account.

## Pick a mode

<div class="pf-cards" markdown>

<div class="pf-card" markdown>
### 🐳 Docker Compose (recommended)
Three containers, isolated processes, ML on dedicated cores, single `docker compose up` to start.
</div>

<div class="pf-card" markdown>
### 🎬 [Demo mode](demo-mode.md)
No Proton account, no session file, no configuration. Try every feature on a fixture of CC0 photos.
</div>

<div class="pf-card" markdown>
### 🛠️ Single process
Set `RUN_INDEXER=1` on the `app` container to run the recognition pipeline in-process. Handy for debugging or low-resource boxes.
</div>

<div class="pf-card" markdown>
### 🧑‍💻 Local dev (Python)
Install requirements, run `python main.py`. See [reference/configuration.md](../reference/configuration.md) for the env vars you need.
</div>

</div>

## Docker Compose (production layout)

### 1. Get a Proton session file

First-time user? Install the [Proton Drive CLI](https://proton.me/download/drive/cli), run `proton-drive auth login` once (browser-based, ~2 minutes), then export the session:

```bash
scripts/export-session.sh    # writes credentials/auth-session.json (chmod 600)
```

Already have the CLI session in `pass`? The one-liner works too:

```bash
pass show ch.proton.drive/drive-sdk-cli/auth-session > credentials/auth-session.json
```

See the [Session file guide](session-export.md) for the full recipe, including the macOS Keychain variant, the Windows Credential Manager, and the most secure option — the SDK's GPG-encrypted `pass` store (`PROTON_DRIVE_CREDENTIALS_STORE=pass`).

!!! warning "Keep this file private"
    `auth-session.json` contains your account tokens. It is mounted **into the bridge container only** (issue #32) — the indexer and the internet-facing `app` container have no path to it, so a vulnerability in one of them cannot leak your session. The mount is writable because the SDK rewrites the file on token refresh; see [Session file](session-export.md) for the full layout. The file should never be committed to git (the repo's `.gitignore` already excludes `credentials/`).

### 2. Configure

```bash
cp .env.example .env
# Inside the containers DATA_DIR is always /data and PHOTOS_DIR is always /takeout.
# To persist data on a specific host disk, set the compose-level mount:
# DATA_MOUNT=/srv/proton-faces/data
```

### 3. Start

```bash
docker compose up -d
```

Then open **http://localhost:8080** and sign in with the admin account you created with:

```bash
ADMIN_PASSWORD=changeme scripts/create-admin.sh admin
```

Prebuilt images are published to the GitHub Container Registry, so `docker compose up` pulls them — no building on the server. To build from source instead, use `docker compose build --pull` (or `--build`).

### 4. Watch the indexer catch up

```bash
docker compose logs -f indexer
```

You'll see lines like:

```
INFO indexer: sync: 1234 remote, 87 new, 0 gone
INFO indexer: processed pic-...: 2 faces, clip=ok
INFO indexer: cluster: 5 clusters from 23 faces
```

The web UI becomes useful right away as results stream in.

## Single-process layout (fallback)

```bash
echo "RUN_INDEXER=1" >> .env
docker compose up -d
```

This collapses the indexer into the `app` container. Same UI, same API. Slightly worse latency on small boxes (face detection preempts the API event loop), but fewer moving parts.

## Local Python install

For hacking on the code itself:

```bash
cd app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Bake or download the ML models into ./data/models (see Dockerfile for the steps)
DATA_DIR=./data MODELS_DIR=./data/models python src/main.py
```

Open http://localhost:8080.

## Volumes

| Path | Purpose | Persist by default? |
|---|---|---|
| `data` (named volume) | Thumbnails + SQLite + vector index | Yes |
| `credentials/` | Read-only mount of your Proton session | You manage |
| `/takeout` | Read-only mount of a local Google Takeout export for GPS backfill | Optional |

To persist the data volume on a specific host disk, set `DATA_MOUNT` in `.env`:

```bash
DATA_MOUNT=/srv/proton-faces/data
```

The directory must be writable by UID 1000 (the user all containers run as).

> **Running the image directly (`docker run`).** The images do **not** declare a `VOLUME`, so
> when you run them outside compose you must mount the data directory yourself, e.g.
> `-v proton-faces-data:/data`. Compose does this automatically (it mounts
> `${DATA_MOUNT:-data}:/data` on every service); the note only matters for manual `docker run`
> experiments, where an unmounted `/data` writes to the container's ephemeral filesystem and
> is lost when the container is removed.

## Updating

```bash
docker compose pull
docker compose up -d
```

The data volume is preserved across updates. The schema migrates automatically on `init_db()`.

### Pinning images to a version or digest

`compose.yml` references `ghcr.io/mmornati/proton-faces-*:latest` by default, so `docker compose pull` always fetches the newest published image. For a supply-chain-conscious deployment you can pin to an immutable reference instead:

- **Versioned tag** — every `v*` git tag publishes `ghcr.io/mmornati/proton-faces-{bridge,app}:<tag>` (in addition to `:latest`). Pin a release:

  ```yaml
  image: ghcr.io/mmornati/proton-faces-app:v1.2.3
  ```

- **Digest pin** — the strongest guarantee: the image is identified by its content hash, so a compromised or broken publish can never be pulled silently. Resolve the digest of a known-good tag, then pin it:

  ```bash
  docker buildx imagetools inspect ghcr.io/mmornati/proton-faces-app:v1.2.3
  ```

  ```yaml
  image: ghcr.io/mmornati/proton-faces-app@sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
  ```

  With a digest pin, `docker compose pull` only updates when the digest itself changes — bump it deliberately when you want to upgrade.

## Uninstalling

```bash
docker compose down            # stop containers, keep the data volume
docker compose down --volumes  # also wipe the data volume
```

The bridge container is stateless (the session file lives on the `data` volume too, via `PROTON_DRIVE_CACHE_DIR`); wiping it logs the bridge out.

---

**Next:** the [Quickstart](quickstart.md) walks you through your first search.

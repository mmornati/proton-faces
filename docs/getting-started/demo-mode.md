# Demo mode

Proton Faces ships with a **demo mode** that lets you explore every feature of the app without a Proton account, a session file, or any configuration. The bridge container is skipped entirely; in its place, an in-process `DemoBridge` serves a curated fixture of free CC0/Unsplash photos.

<div class="pf-banner" markdown>
**Use demo mode** to evaluate proton-faces, demo the project to others, write end-to-end tests against the API, or just play with the UI without touching your real library.
</div>

## Quick start

```bash
docker compose --profile demo up -d
```

Open **http://localhost:8080** and sign in with:

- username: `demo`
- password: `proton-faces`

That's it. The two containers (`indexer-demo` and `app-demo`) come up with `DEMO_MODE=1`, skip the Proton bridge, and load the bundled fixture.

Override the password:

```bash
DEMO_ADMIN_PASSWORD=mysecret docker compose --profile demo up -d
```

## What you get

The fixture is a small but **representative** library:

| | Count | Source |
|---|---|---|
| Faces | 32 | randomuser.me portraits (MIT) |
| Scenes / objects | 50 | picsum.photos (Unsplash License) |
| **Total photos** | **82** | |
| Distinct people clusters | ~8 | HDBSCAN over face embeddings |
| Cities with markers on the map | 10 | Lille, Paris, Barcelona, Rome, Berlin, Amsterdam, NYC, Tokyo, Lisbon, Edinburgh |
| Albums | 4 | *Summer 2024*, *Trips*, *Portraits*, *Objects & Scenes* |
| Duplicate pairs | 2 | for the Duplicates view |

Every demo photo has a real sha1 hash, real capture time, and the right media type — the same code paths that run on real Proton photos run on these.

## What changes in demo mode

| Path | Real mode | Demo mode |
|---|---|---|
| `proton-bridge` container | Runs and talks to Proton | Skipped (compose profile) |
| `bridge_client.get_bridge()` | HTTP client to bridge | Returns `DemoBridge` |
| Timeline / nodes / albums | Fetched from Proton's API | Served from `app/src/demo_assets/fixture.json` |
| Thumbnails | Downloaded from Proton | Generated locally from `photos/<uid>.jpg` |
| GPS / place names | From a Google Takeout export | Injected from the fixture |
| Multi-user auth | Manual `--create-admin` | Default `demo` admin auto-created on first boot |
| Everything else (CLIP, faces, clustering, search, admin) | Same | Same |

The ML pipeline — InsightFace, CLIP, HDBSCAN — runs **exactly** as it does in real mode. The only thing that's different is where the photo bytes come from.

## How it's wired

- `app/src/demo.py` defines a `DemoBridge` class that mirrors the surface of the real `BridgeClient`.
- `app/src/bridge_client.py::get_bridge()` returns `DemoBridge` when `DEMO_MODE=1` is set.
- The `demo` compose profile sets `DEMO_MODE=1` on two services (`indexer-demo` and `app-demo`) and excludes `proton-bridge`.

You can read the source: [`app/src/demo.py`](https://github.com/mmornati/proton-faces/blob/main/app/src/demo.py).

## Refreshing the fixture

The fixture is committed to the repo under `app/src/demo_assets/`. To re-fetch it:

```bash
python scripts/fetch-demo-assets.py
```

This pulls new face portraits from randomuser.me and new scenes from picsum.photos. Re-run any time — already-downloaded photos are reused, and the fixture is fully deterministic (RNG seeds are pinned).

To wipe the data volume and start over:

```bash
docker compose --profile demo down --volumes
docker compose --profile demo up -d
```

## Using demo mode with your own code

Demo mode is also a convenient way to run automated UI tests against a real proton-faces instance. The capture script in `scripts/capture_screenshots.py` uses it as a target:

```bash
BASE_URL=http://localhost:8080 DEMO_USERNAME=demo DEMO_PASSWORD=proton-faces \
    python scripts/capture_screenshots.py
```

For API scripting, see the [REST API reference](../reference/api.md).

## Limitations

- The fixture photos are **JPEGs only** — no HEIC, no videos. The HEIC + video fallback path in the fullres loop is not exercised by demo mode.
- All photos are ~512px max-side thumbnails on disk (the "full-resolution" view returns the same bytes).
- Album membership is curated, not real Proton albums.
- No EXIF data besides GPS is populated.

These don't affect the UI behavior — every view in the docs site works against demo mode.

## Credits

Every photo in the fixture is free for use. See [`app/src/demo_assets/CREDITS.md`](https://github.com/mmornati/proton-faces/blob/main/app/src/demo_assets/CREDITS.md) for the full per-photo attribution.

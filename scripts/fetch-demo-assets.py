#!/usr/bin/env python3
"""Fetch the DEMO_MODE photo fixture from CC0 / MIT-licensed sources.

Outputs:
  app/src/demo_assets/photos/<uid>.jpg   — the actual photo bytes
  app/src/demo_assets/fixture.json       — timeline + nodes + albums metadata
  app/src/demo_assets/CREDITS.md         — per-photo license attribution

Sources (no API keys required):
  • Picsum (https://picsum.photos) — Lorem Picsum serves Unsplash photos under
    the Unsplash License (free for commercial and noncommercial use, no
    attribution required; attribution included here for credit).
    https://picsum.photos/
  • Random User (https://randomuser.me) — MIT-licensed avatar portraits used
    here to give the face-recognition pipeline realistic subjects to cluster.
    https://randomuser.me/copyright

This script is idempotent: re-running it will re-download any missing photos
and refresh fixture.json. Already-downloaded photos are reused (by uid).

Run from the repo root:
    python scripts/fetch-demo-assets.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app" / "src" / "demo_assets"
PHOTOS = ASSETS / "photos"
FIXTURE = ASSETS / "fixture.json"
CREDITS = ASSETS / "CREDITS.md"

# Curated city / GPS lookup for the places demo. Maps a human place name to
# (lat, lng). Photos in the fixture get tagged with these so the Places view
# shows clustered markers across multiple cities.
CITIES = [
    ("Lille, France",          50.6292,   3.0573),
    ("Paris, France",          48.8566,   2.3522),
    ("Barcelona, Spain",       41.3851,   2.1734),
    ("Rome, Italy",            41.9028,  12.4964),
    ("Berlin, Germany",        52.5200,  13.4050),
    ("Amsterdam, Netherlands", 52.3676,   4.9041),
    ("New York, NY, USA",      40.7128, -74.0060),
    ("Tokyo, Japan",           35.6762, 139.6503),
    ("Lisbon, Portugal",       38.7223,  -9.1393),
    ("Edinburgh, UK",          55.9533,  -3.1883),
]

# Album membership: each fixture photo is tagged with 0..2 album uids.
ALBUMS = {
    "alb-summer-2024": "Summer 2024",
    "alb-trips": "Trips",
    "alb-portraits": "Portraits",
    "alb-objects": "Objects & Scenes",
}


def _uid(prefix: str, n: int) -> str:
    """Build a stable, short, opaque-looking uid for a demo photo."""
    return f"{prefix}{n:04d}{hashlib.sha1(f'{prefix}-{n}'.encode()).hexdigest()[:6]}"


def _download(url: str, dest: Path, *, retries: int = 4) -> bool:
    """Stream a URL to disk. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "proton-faces-demo-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        fh.write(chunk)
                tmp.replace(dest)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"  ✗ {url}: {last_err}", file=sys.stderr)
    return False


def fetch_randomuser_portraits(count: int = 32) -> list[dict]:
    """Download N face portraits from randomuser.me (MIT license).

    Uses deterministic seeds so re-runs don't churn the fixture.
    """
    out: list[dict] = []
    sexes = ["men", "women"]
    print(f"Fetching {count} face portraits from randomuser.me…")
    rng = random.Random(42)
    used = set()
    n_ok = 0
    while n_ok < count:
        sex = rng.choice(sexes)
        n = rng.randint(0, 99)
        uid = f"face-{sex}-{n:02d}"
        if uid in used:
            continue
        used.add(uid)
        url = f"https://randomuser.me/api/portraits/{sex}/{n}.jpg"
        dest = PHOTOS / f"{uid}.jpg"
        if _download(url, dest):
            out.append({"uid": uid, "src": url, "license": "MIT (randomuser.me)", "kind": "face"})
            n_ok += 1
    return out


def fetch_picsum_landscape(seed: int, w: int = 800, h: int = 600) -> bytes | None:
    """Stream a picsum image at fixed seed/size — Unsplash-mirrored, free use."""
    url = f"https://picsum.photos/seed/pf{seed}/{w}/{h}"
    req = urllib.request.Request(url, headers={"User-Agent": "proton-faces-demo-fetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except Exception as exc:
        print(f"  ✗ picsum seed {seed}: {exc}", file=sys.stderr)
        return None


def fetch_picsum_portrait(seed: int) -> bytes | None:
    return fetch_picsum_landscape(seed, 600, 800)


def fetch_picsum_set(count: int = 50) -> list[dict]:
    """Download a mixed set of landscape + portrait picsum photos.

    ~70% landscape (for the main grid variety), ~30% portrait (for face-shape
    clustering variety).
    """
    out: list[dict] = []
    print(f"Fetching {count} picsum scenes…")
    n_ok = 0
    seed = 100
    while n_ok < count:
        is_portrait = (n_ok % 4 == 3)
        uid = f"pic-{seed:04d}"
        data = fetch_picsum_portrait(seed) if is_portrait else fetch_picsum_landscape(seed)
        if data is None:
            seed += 1
            continue
        (PHOTOS / f"{uid}.jpg").write_bytes(data)
        out.append({
            "uid": uid,
            "src": f"https://picsum.photos/seed/pf{seed}",
            "license": "Unsplash License (picsum.photos)",
            "kind": "scene",
            "portrait": is_portrait,
        })
        n_ok += 1
        seed += 1
    return out


def build_fixture(face_photos: list[dict], scene_photos: list[dict]) -> list[dict]:
    """Compose the timeline/nodes metadata list the demo bridge serves."""
    rng = random.Random(7)
    base_ts = int(time.mktime((2024, 1, 1, 10, 0, 0, 0, 0, 0)))

    items: list[dict] = []

    # Place face photos with a couple of duplicate sha1s so the duplicates view
    # has something to show.
    duplicate_pairs = [
        (face_photos[0]["uid"], face_photos[1]["uid"]) if len(face_photos) >= 2 else None,
        (scene_photos[2]["uid"], scene_photos[5]["uid"]) if len(scene_photos) >= 6 else None,
    ]
    sha1_map: dict[str, str] = {}

    # Faces — all in the "Portraits" album; ~half have GPS to populate Places.
    for i, p in enumerate(face_photos):
        uid = p["uid"]
        ts = base_ts + i * 3600 * 6 + rng.randint(0, 1800)
        # Force sha1 duplicates so the Duplicates view has matches.
        sha1: str | None = None
        for a, b in duplicate_pairs:
            if a == uid:
                sha1 = _sha1_for(b)
                break
            if b == uid:
                sha1 = _sha1_for(a)
                break
        if sha1 is None:
            sha1 = _sha1_for(uid)
        sha1_map[uid] = sha1
        gps = None
        if i % 2 == 0 and rng.random() < 0.7:
            city, lat, lng = rng.choice(CITIES)
            gps = {"lat": lat, "lng": lng, "place": city}
        items.append({
            "uid": uid,
            "name": f"portrait_{uid}.jpg",
            "mediaType": "image/jpeg",
            "captureTime": ts,
            "sha1": sha1,
            "size": 38000 + rng.randint(0, 60000),
            "albums": ["alb-portraits"],
            "gps": gps,
        })

    # Scenes — spread across the other albums; more have GPS.
    scene_albums = ["alb-trips", "alb-summer-2024", "alb-objects", "alb-trips"]
    for i, p in enumerate(scene_photos):
        uid = p["uid"]
        ts = base_ts + (i + 200) * 3600 * 5 + rng.randint(0, 3600)
        sha1: str | None = None
        for a, b in duplicate_pairs:
            if a == uid:
                sha1 = _sha1_for(b)
                break
            if b == uid:
                sha1 = _sha1_for(a)
                break
        if sha1 is None:
            sha1 = _sha1_for(uid)
        sha1_map[uid] = sha1
        gps = None
        if rng.random() < 0.55:
            city, lat, lng = rng.choice(CITIES)
            gps = {"lat": lat, "lng": lng, "place": city}
        albums = rng.sample(scene_albums, k=rng.randint(1, 2))
        items.append({
            "uid": uid,
            "name": f"scene_{uid}.jpg",
            "mediaType": "image/jpeg",
            "captureTime": ts,
            "sha1": sha1,
            "size": 220000 + rng.randint(0, 600000),
            "albums": albums,
            "gps": gps,
        })

    items.sort(key=lambda d: d["captureTime"], reverse=True)
    return items


def _sha1_for(uid: str) -> str:
    return hashlib.sha1(f"pf-demo-{uid}".encode()).hexdigest()


def write_credits(face_photos: list[dict], scene_photos: list[dict]) -> None:
    lines = [
        "# Demo fixture credits",
        "",
        "All photos in `app/src/demo_assets/photos/` are free for use under",
        "the licenses below. They are bundled with proton-faces so the DEMO_MODE",
        "build has a working photo library without any Proton credentials.",
        "",
        "## Sources",
        "",
        "| Source | License | Used for |",
        "|--------|---------|----------|",
        "| [picsum.photos](https://picsum.photos) | [Unsplash License](https://unsplash.com/license) (free for commercial and noncommercial use) | Scene / object photos |",
        "| [randomuser.me](https://randomuser.me) | [MIT](https://randomuser.me/copyright) | Face portraits used by the recognition pipeline |",
        "",
        "## Individual photos",
        "",
    ]
    for p in face_photos:
        lines.append(f"- `{p['uid']}.jpg` — randomuser.me portrait, {p['license']} — {p['src']}")
    for p in scene_photos:
        lines.append(f"- `{p['uid']}.jpg` — picsum.photos scene, {p['license']} — {p['src']}")
    lines.extend([
        "",
        "## Removal",
        "",
        "If you fork the project and don't want bundled demo photos, delete",
        "`app/src/demo_assets/photos/` and `app/src/demo_assets/fixture.json`.",
        "DEMO_MODE will then refuse to start until the assets are restored.",
        "",
    ])
    CREDITS.write_text("\n".join(lines))


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    PHOTOS.mkdir(parents=True, exist_ok=True)

    if FIXTURE.exists():
        print(f"fixture.json already present — only fetching missing photos")
        # Re-parse just enough to learn which uids we already have so we don't
        # re-download scenes that exist on disk.
        existing = json.loads(FIXTURE.read_text())
        face_existing = [u for u in existing if u["uid"].startswith("face-")]
        scene_existing = [u for u in existing if u["uid"].startswith("pic-")]
        face_photos = [{"uid": u["uid"], "src": "already-fetched", "license": "MIT (randomuser.me)", "kind": "face"} for u in face_existing]
        scene_photos = [{"uid": u["uid"], "src": "already-fetched", "license": "Unsplash License (picsum.photos)", "kind": "scene"} for u in scene_existing]
        # Still try to fetch any photos that are missing on disk.
        for u in existing:
            if not (PHOTOS / f"{u['uid']}.jpg").exists():
                print(f"  ! missing on disk: {u['uid']}")
    else:
        face_photos = []
        scene_photos = []

    face_needed = 32 - len(face_photos)
    scene_needed = 50 - len(scene_photos)
    if face_needed > 0:
        face_photos.extend(fetch_randomuser_portraits(face_needed))
    if scene_needed > 0:
        scene_photos.extend(fetch_picsum_set(scene_needed))

    items = build_fixture(face_photos, scene_photos)
    FIXTURE.write_text(json.dumps(items, indent=2))
    write_credits(face_photos, scene_photos)

    print()
    print(f"✓ wrote {len(items)} photos to {PHOTOS}")
    print(f"✓ wrote {FIXTURE.relative_to(ROOT)}")
    print(f"✓ wrote {CREDITS.relative_to(ROOT)}")
    print()
    print("Next: bring up the demo with")
    print("    docker compose --profile demo up -d")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Augment the DEMO_MODE fixture with repeated identities.

The base fixture (scripts/fetch-demo-assets.py) downloads one portrait per
randomuser.me seed, giving 32 different people each appearing in exactly
one photo. With no repeated identities, the clustering algorithm has
nothing to cluster — every face is its own singleton.

This script picks a handful of identities from the existing portrait set
and downloads MULTIPLE COPIES of each, with different uids, so the
clustering pipeline actually has repeated faces to group.

Output: writes additional entries into app/src/demo_assets/photos/ and
app/src/demo_assets/fixture.json.

Run from the repo root:
    python scripts/augment-demo-fixture.py

Idempotent: re-running won't add duplicates.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app" / "src" / "demo_assets"
PHOTOS = ASSETS / "photos"
FIXTURE = ASSETS / "fixture.json"
CREDITS = ASSETS / "CREDITS.md"

# How many copies of each identity we add. With min_cluster_size=2 in
# settings, we need >= 2 copies for clustering to merge them.
COPIES_PER_IDENTITY = 5

# Pick a handful of identities to repeat. These are randomuser.me seeds
# that we already have on disk (face-men-03.jpg, face-women-77.jpg, etc.).
# We pick 5 men and 5 women so the People view shows a realistic mix.
IDENTITIES = [
    ("face-men-03",   "men",   3),
    ("face-men-17",   "men",  17),
    ("face-men-29",   "men",  29),
    ("face-men-71",   "men",  71),
    ("face-men-86",   "men",  86),
    ("face-women-19", "women", 19),
    ("face-women-28", "women", 28),
    ("face-women-77", "women", 77),
    ("face-women-13", "women", 13),
    ("face-women-05", "women",  5),
]


def _uid(identity: str, copy_idx: int) -> str:
    """Stable uid: <identity>-c<copy_idx>. Example: face-men-03-c1."""
    return f"{identity}-c{copy_idx}"


def _sha1_for(uid: str) -> str:
    return hashlib.sha1(f"pf-demo-{uid}".encode()).hexdigest()


def _download(url: str, timeout: int = 30) -> bytes | None:
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "proton-faces-demo-augment/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    print(f"  ✗ {url}: {last_err}", file=sys.stderr)
    return None


def _make_variations(src_bytes: bytes, identity: str, n_copies: int) -> list[bytes]:
    """Make n variations of the same face image: same identity, different
    JPEG quality / tiny pixel shift / slight crop. This gives the face
    detector multiple distinct-but-similar inputs that cluster correctly.

    Each variation is the same face from randomuser.me, so the InsightFace
    embedding for each is nearly identical → clustering merges them into
    one person.
    """
    try:
        img = Image.open(io.BytesIO(src_bytes)).convert("RGB")
    except Exception as exc:
        print(f"  ✗ could not decode {identity}: {exc}", file=sys.stderr)
        return []

    out = []
    rng = random.Random(hash(identity) & 0xFFFFFFFF)
    for i in range(n_copies):
        # Small random crop / resize to vary pixel values slightly without
        # changing the identity. Most "copies" stay at native size.
        variant = img.copy()
        if i > 0:
            # Tiny scale: 0.95 .. 1.05
            scale = 0.95 + rng.random() * 0.10
            w, h = variant.size
            nw, nh = max(32, int(w * scale)), max(32, int(h * scale))
            variant = variant.resize((nw, nh), Image.LANCZOS)
        # Vary JPEG quality so the bytes differ but the face stays the same.
        quality = 75 + rng.randint(0, 20)
        buf = io.BytesIO()
        variant.save(buf, format="JPEG", quality=quality)
        out.append(buf.getvalue())
    return out


def main() -> int:
    if not FIXTURE.exists():
        print(f"Run scripts/fetch-demo-assets.py first to create {FIXTURE}",
              file=sys.stderr)
        return 2

    ASSETS.mkdir(parents=True, exist_ok=True)
    PHOTOS.mkdir(parents=True, exist_ok=True)

    fixture = json.loads(FIXTURE.read_text())
    existing_uids = {item["uid"] for item in fixture}

    # Group existing fixture entries by identity (uid without copy suffix).
    by_identity: dict[str, list[dict]] = {}
    for item in fixture:
        uid = item["uid"]
        if uid.startswith("face-") and "-c" in uid:
            base = uid.rsplit("-c", 1)[0]
            by_identity.setdefault(base, []).append(item)

    new_entries: list[dict] = []
    base_ts = int(time.mktime((2024, 6, 1, 12, 0, 0, 0, 0, 0)))
    rng = random.Random(123)

    for uid_base, sex, n in IDENTITIES:
        src_path = PHOTOS / f"{uid_base}.jpg"
        if not src_path.exists():
            # Fetch the portrait from randomuser.me if it's not already on disk.
            url = f"https://randomuser.me/api/portraits/{sex}/{n}.jpg"
            data = _download(url)
            if not data:
                continue
            src_path.write_bytes(data)

        # Skip copies that already exist.
        copies_to_make = []
        for i in range(1, COPIES_PER_IDENTITY):
            new_uid = _uid(uid_base, i)
            if new_uid in existing_uids:
                continue
            copies_to_make.append((i, new_uid))

        if not copies_to_make:
            print(f"  · {uid_base}: {COPIES_PER_IDENTITY - 1} copies already present, skipping")
            continue

        # Make variations of the original face image.
        src_bytes = src_path.read_bytes()
        variations = _make_variations(src_bytes, uid_base, len(copies_to_make))
        if len(variations) != len(copies_to_make):
            print(f"  ✗ {uid_base}: failed to make variations", file=sys.stderr)
            continue

        for (copy_idx, new_uid), jpeg_bytes in zip(copies_to_make, variations):
            dest = PHOTOS / f"{new_uid}.jpg"
            dest.write_bytes(jpeg_bytes)

            ts = base_ts + rng.randint(0, 60 * 60 * 24 * 30)
            entry = {
                "uid": new_uid,
                "name": f"portrait_{new_uid}.jpg",
                "mediaType": "image/jpeg",
                "captureTime": ts,
                "sha1": _sha1_for(new_uid),
                "size": len(jpeg_bytes),
                "albums": ["alb-portraits"],
                "is_demo_repeat": True,
                "demo_identity": uid_base,
            }
            new_entries.append(entry)

        print(f"  + {uid_base}: added {len(copies_to_make)} repeat copies")

    if not new_entries:
        print("\nNothing to add — all repeat copies already present.")
        return 0

    fixture.extend(new_entries)
    fixture.sort(key=lambda d: d["captureTime"], reverse=True)
    FIXTURE.write_text(json.dumps(fixture, indent=2))
    print(f"\n✓ added {len(new_entries)} entries to {FIXTURE.relative_to(ROOT)}")
    print(f"  total photos: {len(fixture)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Rebuild the DEMO_MODE face fixture from curated Wikimedia Commons portraits.

The original fixture downloaded 32 avatar portraits from randomuser.me, whose
individual photographers cannot be identified (only "MIT (randomuser.me)"
is claimable). This script replaces those 32 faces with Wikimedia Commons
portraits that have an identifiable photographer and a permissive license
(CC0 / CC BY / CC BY-SA), keeping the existing uid scheme
(`face-{sex}-{n:02d}`) and re-generating the repeat "copies" used by the
clustering pipeline.

Inputs:
  /tmp/all_faces/{i:02d}.jpg      — 500px Commons thumbs already downloaded
  /tmp/selection.json             — {men:[...32...], women:[...]} with fields
                                    i, artist, title, license
  app/src/demo_assets/fixture.json — existing fixture (scenes are preserved)

Outputs:
  app/src/demo_assets/photos/face-*.jpg  (32 base + 40 repeat copies)
  app/src/demo_assets/fixture.json
  app/src/demo_assets/CREDITS.md   (face rows rewritten with artist/license)

Run from the repo root:
    python scripts/rebuild-faces-from-commons.py
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import time
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

ALL_FACES = Path("/tmp/all_faces")
SELECTION = Path("/tmp/selection.json")

# Max dimension for normalized face JPEGs (keeps full_photo downloads light and
# faces comfortably large for the detector). InsightFace det_size is 640.
MAX_DIM = 800

# Target base uids, matching the existing fixture (17 men + 15 women).
MEN_UIDS = ["face-men-03", "face-men-05", "face-men-11", "face-men-12",
            "face-men-17", "face-men-29", "face-men-45", "face-men-48",
            "face-men-70", "face-men-71", "face-men-75", "face-men-86",
            "face-men-89", "face-men-90", "face-men-91", "face-men-97",
            "face-men-98"]
WOMEN_UIDS = ["face-women-00", "face-women-04", "face-women-05", "face-women-10",
              "face-women-13", "face-women-19", "face-women-28", "face-women-31",
              "face-women-35", "face-women-43", "face-women-68", "face-women-73",
              "face-women-75", "face-women-77", "face-women-80"]

# Repeat identities (5 men + 5 women) — must be present as base uids above.
IDENTITIES = [
    "face-men-03", "face-men-17", "face-men-29", "face-men-71", "face-men-86",
    "face-women-19", "face-women-28", "face-women-77", "face-women-13", "face-women-05",
]
COPIES_PER_IDENTITY = 5  # base + 4 copies

# City / GPS lookup (mirrors scripts/fetch-demo-assets.py).
CITIES = [
    ("Lille, France", 50.6292, 3.0573), ("Paris, France", 48.8566, 2.3522),
    ("Barcelona, Spain", 41.3851, 2.1734), ("Rome, Italy", 41.9028, 12.4964),
    ("Berlin, Germany", 52.5200, 13.4050), ("Amsterdam, Netherlands", 52.3676, 4.9041),
    ("New York, NY, USA", 40.7128, -74.0060), ("Tokyo, Japan", 35.6762, 139.6503),
    ("Lisbon, Portugal", 38.7223, -9.1393), ("Edinburgh, UK", 55.9533, -3.1883),
]

COMMONS_PAGE = "https://commons.wikimedia.org/wiki/"


def _sha1_for(uid: str) -> str:
    return hashlib.sha1(f"pf-demo-{uid}".encode()).hexdigest()


def normalize_face(src: Path) -> bytes:
    """Downscale (if needed) and re-encode a portrait as RGB JPEG q90."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_DIM / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_copies(src_bytes: bytes, identity: str, n: int) -> list[bytes]:
    """n variations of the same face (tiny scale / quality tweaks) so the
    embeddings stay near-identical and clustering merges them."""
    img = Image.open(io.BytesIO(src_bytes)).convert("RGB")
    rng = random.Random(hash(identity) & 0xFFFFFFFF)
    out = []
    for i in range(n):
        variant = img.copy()
        if i > 0:
            scale = 0.95 + rng.random() * 0.10
            w, h = variant.size
            variant = variant.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.LANCZOS)
        quality = 75 + rng.randint(0, 20)
        buf = io.BytesIO()
        variant.save(buf, format="JPEG", quality=quality)
        out.append(buf.getvalue())
    return out


def main() -> int:
    selection = json.loads(SELECTION.read_text())
    men_sel = selection["men"]
    women_sel = selection["women"]
    assert len(men_sel) == 17 and len(women_sel) == 15, (len(men_sel), len(women_sel))

    # Map each selected Commons image to its target uid.
    plan: dict[str, dict] = {}
    for uid, e in zip(MEN_UIDS, men_sel):
        plan[uid] = e
    for uid, e in zip(WOMEN_UIDS, women_sel):
        plan[uid] = e
    assert set(plan) == set(MEN_UIDS) | set(WOMEN_UIDS)

    # 1. Write the 32 base face photos.
    face_meta: dict[str, dict] = {}
    for uid, e in plan.items():
        src = ALL_FACES / f"{e['i']:02d}.jpg"
        if not src.exists():
            print(f"  ✗ missing {src}", file=sys.stderr)
            return 2
        jpeg = normalize_face(src)
        (PHOTOS / f"{uid}.jpg").write_bytes(jpeg)
        face_meta[uid] = {
            "uid": uid, "artist": e["artist"], "title": e["title"],
            "license": e["license"], "bytes": jpeg,
        }
        print(f"  + {uid}.jpg  <-  {src.name}  ({e['artist']})")

    # 2. Read existing fixture, keep scenes verbatim.
    existing = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else []
    scene_photos = [e for e in existing if e["uid"].startswith("pic-")]
    assert len(scene_photos) == 50, f"expected 50 scenes, got {len(scene_photos)}"

    # 3. Regenerate the 40 repeat copies from the new base faces.
    rng = random.Random(123)
    base_ts = int(time.mktime((2024, 6, 1, 12, 0, 0, 0, 0, 0)))
    copy_entries = []
    for identity in IDENTITIES:
        src_bytes = (PHOTOS / f"{identity}.jpg").read_bytes()
        for i in range(1, COPIES_PER_IDENTITY):
            new_uid = f"{identity}-c{i}"
            jpeg = make_copies(src_bytes, identity, 1)[0]
            (PHOTOS / f"{new_uid}.jpg").write_bytes(jpeg)
            copy_entries.append({
                "uid": new_uid,
                "name": f"portrait_{new_uid}.jpg",
                "mediaType": "image/jpeg",
                "captureTime": base_ts + rng.randint(0, 60 * 60 * 24 * 30),
                "sha1": _sha1_for(new_uid),
                "size": len(jpeg),
                "albums": ["alb-portraits"],
                "is_demo_repeat": True,
                "demo_identity": identity,
            })

    # 4. Build the 32 base face entries (with a forced duplicate sha1 pair so
    #    the Duplicates view has content).
    rng2 = random.Random(7)
    base_ts2 = int(time.mktime((2024, 1, 1, 10, 0, 0, 0, 0, 0)))
    dup_face = _sha1_for(MEN_UIDS[0])
    face_entries = []
    for i, uid in enumerate(MEN_UIDS + WOMEN_UIDS):
        sha1 = _sha1_for(uid)
        if uid == MEN_UIDS[1]:
            sha1 = dup_face  # force face[0] and face[1] to share a content hash
        gps = None
        if i % 2 == 0 and rng2.random() < 0.7:
            city, lat, lng = rng2.choice(CITIES)
            gps = {"lat": lat, "lng": lng, "place": city}
        meta = face_meta[uid]
        face_entries.append({
            "uid": uid,
            "name": f"portrait_{uid}.jpg",
            "mediaType": "image/jpeg",
            "captureTime": base_ts2 + i * 3600 * 6 + rng2.randint(0, 1800),
            "sha1": sha1,
            "size": len(meta["bytes"]),
            "albums": ["alb-portraits"],
            "gps": gps,
        })

    # 5. Force one scene duplicate pair too (scene[2] and scene[5]).
    if len(scene_photos) >= 6:
        scene_photos[5]["sha1"] = scene_photos[2]["sha1"]

    items = face_entries + copy_entries + scene_photos
    items.sort(key=lambda d: d["captureTime"], reverse=True)
    FIXTURE.write_text(json.dumps(items, indent=2))
    print(f"  ✓ fixture: {len(items)} entries ({len(face_entries)} faces, "
          f"{len(copy_entries)} copies, {len(scene_photos)} scenes)")

    # 6. Rewrite CREDITS.md. Keep existing scene (pic-*) attribution lines
    #    verbatim (already rewritten to real photographers), replace the
    #    face section and the randomuser Sources row.
    existing_credits = CREDITS.read_text() if CREDITS.exists() else ""
    scene_lines = [ln for ln in existing_credits.splitlines() if ln.startswith("- `pic-")]
    if not scene_lines:
        scene_lines = [f"- `{e['uid']}.jpg` — picsum.photos scene (Unsplash License) — https://picsum.photos"
                       for e in scene_photos]

    face_rows = []
    for uid in MEN_UIDS + WOMEN_UIDS:
        e = face_meta[uid]
        page = COMMONS_PAGE + e["title"]
        face_rows.append(f"- `{uid}.jpg` — {e['artist']}, “{e['title'].replace('File:', '')}”, {e['license']} — {page}")
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
        "| [picsum.photos](https://picsum.photos) → [Unsplash](https://unsplash.com) | [Unsplash License](https://unsplash.com/license) (free for commercial and noncommercial use, no attribution required) | Scene / object photos, attributed to the original photographer |",
        "| [Wikimedia Commons](https://commons.wikimedia.org) | [CC0 / CC BY / CC BY-SA](https://commons.wikimedia.org/wiki/Commons:Licensing) | Face portraits used by the recognition pipeline |",
        "",
        "## Individual photos",
        "",
        "### Faces (Wikimedia Commons, photographer and license per file)",
        "",
    ]
    lines.extend(face_rows)
    lines.extend([
        "",
        "### Scenes (picsum.photos → Unsplash)",
        "",
    ])
    lines.extend(scene_lines)
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
    print(f"  ✓ credits: {CREDITS.relative_to(ROOT)}")
    print("\nNext: wipe the demo data volume and restart the demo stack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
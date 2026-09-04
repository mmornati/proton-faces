#!/usr/bin/env python3
"""Find Wikimedia Commons portrait photos with named photographers and
permissive licenses, validated to contain a single detectable face.

Runs inside the demo container (has insightface + /models). Writes:
  /data/commons_all.json          — every candidate (full title, artist, license, url, thumb)
  /data/commons_candidates.json   — validated single-face candidates (appended)

Usage (inside container):
    python scripts/find-commons-faces.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, "/app/src")

import numpy as np
from PIL import Image

CATEGORIES = [
    "Portrait photographs of men",
    "Portrait photographs of women",
]

ALLOWED_LICENSES = {
    "CC0", "Public domain", "CC BY", "CC BY 4.0", "CC BY 2.0", "CC BY-SA 4.0",
    "CC BY-SA 3.0", "CC BY-SA 2.0", "CC BY 2.5", "CC BY-SA",
}

UA = {"User-Agent": "proton-faces-demo/1.0 (attribution research)"}

ALL_PATH = "/data/commons_all.json"
RESULT_PATH = "/data/commons_candidates.json"


def _http_request(req: urllib.request.Request, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                wait = 4 * (attempt + 1)
                print(f"  ... retry {attempt + 1} after {wait}s ({exc.code})", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  ... net retry {attempt + 1} ({exc})", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    return b""


def api(params: dict) -> dict:
    params = {**params, "format": "json"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    return json.loads(_http_request(req))


def category_members(cat: str, limit: int = 300) -> list[dict]:
    out = []
    cont = None
    while len(out) < limit:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{cat}",
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": "480",
        }
        if cont:
            params["gcmcontinue"] = cont
        data = api(params)
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            out.append(p)
        time.sleep(2)
        if "continue" not in data:
            break
        cont = data["continue"].get("gcmcontinue")
    return out


def clean_artist(raw: str) -> str:
    txt = re.sub(r"<[^>]+>", "", raw or "")
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = txt.split("Author", 1)[-1].strip(" :")
    if not txt or txt.lower() in {"unknown", "anonymous", "unknown author", "anonymous author"}:
        return ""
    return txt[:120]


def load_prev() -> set:
    try:
        with open(RESULT_PATH) as fh:
            return {d["title"] for d in json.load(fh)}
    except Exception:
        return set()


def main() -> int:
    from faces import detect_faces
    from io import BytesIO

    done = load_prev()

    # Phase 1: collect all candidates (API only), persist full list.
    if os.path.exists(ALL_PATH):
        seen = {d["title"]: d for d in json.load(open(ALL_PATH))}
        print(f"# loaded {len(seen)} candidates from cache", file=sys.stderr)
    else:
        seen = {}
        for cat in CATEGORIES:
            print(f"# {cat}", file=sys.stderr)
            for p in category_members(cat):
                title = p["title"]
                ii = p.get("imageinfo", [{}])[0]
                em = ii.get("extmetadata", {})
                lic = em.get("LicenseShortName", {}).get("value", "")
                artist = clean_artist(em.get("Artist", {}).get("value", ""))
                if lic not in ALLOWED_LICENSES:
                    continue
                if not artist:
                    continue
                thumb = ii.get("thumburl")
                url = ii.get("descriptionurl", "")
                if not thumb:
                    continue
                seen.setdefault(title, {
                    "title": title, "license": lic, "artist": artist,
                    "url": url, "thumb": thumb,
                })
            print(f"  {cat}: {len(seen)} candidates so far", file=sys.stderr)
        with open(ALL_PATH, "w") as fh:
            json.dump(list(seen.values()), fh, indent=1)
        print(f"# persisted {len(seen)} candidates", file=sys.stderr)

    # Phase 2: validate faces, append incrementally.
    results = []
    pending = [c for c in seen.values() if c["title"] not in done]
    print(f"# {len(seen)} candidates, {len(pending)} to validate", file=sys.stderr)

    with open(RESULT_PATH, "a") as rfh:
        for i, c in enumerate(pending):
            try:
                req = urllib.request.Request(c["thumb"], headers={"User-Agent": "Mozilla/5.0"})
                data = _http_request(req)
                img = Image.open(BytesIO(data)).convert("RGB")
                arr = np.asarray(img)[:, :, ::-1]  # RGB -> BGR
                faces = detect_faces(arr)
                time.sleep(1.0)
            except Exception as exc:
                print(f"  ! {title}: {exc}", file=sys.stderr)
                continue
            if len(faces) != 1:
                continue
            f = faces[0]
            x1, y1, x2, y2 = f["bbox"]
            w, h = x2 - x1, y2 - y1
            c["face_conf"] = round(f["confidence"], 3)
            c["face_frac"] = round((w * h) / (img.width * img.height), 3)
            rfh.write(json.dumps(c) + "\n")
            rfh.flush()
            print(f"  ok {c['title'][:60]} conf={c['face_conf']} frac={c['face_frac']} "
                  f"({img.width}x{img.height})", file=sys.stderr)

    # Rewrite result file as a clean JSON array.
    with open(RESULT_PATH) as fh:
        lines = [l for l in fh if l.strip()]
    arr = [json.loads(l) for l in lines]
    arr.sort(key=lambda c: -c["face_frac"])
    with open(RESULT_PATH, "w") as fh:
        json.dump(arr, fh, indent=1)
    print(f"# total validated: {len(arr)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Demo bridge for proton-faces — no Proton account required.

When `DEMO_MODE=1` is set, `bridge_client.get_bridge()` returns a `DemoBridge`
instead of an HTTP client. `DemoBridge` exposes the same surface as the real
bridge and serves a curated fixture of free CC0 photos shipped in
`app/src/demo_assets/`. This lets anyone run the full app — recognition, search,
people clustering, places, face tagging — without authenticating against a real
Proton account.

Fixture layout:
  app/src/demo_assets/photos/<uid>.jpg   — the actual photo bytes (~80 photos)
  app/src/demo_assets/fixture.json       — timeline + nodes + albums metadata
  app/src/demo_Assets/CREDITS.md         — per-photo license attribution

The fixture is intentionally small (~80 photos, a few MB) so it processes in
under a minute on a modern laptop and the whole directory fits in the Git repo.

Why this exists: proton-faces is a fully self-hosted app, but the real
onboarding path requires a Proton Drive account. Demo mode makes the first run
zero-friction: `docker compose --profile demo up -d`, open the UI, see a working
photo library, no credentials needed.

Lifecycle: `DemoBridge` is a stateless object; the data lives on disk and in
`fixture.json`. Methods are thread-safe (the data is read-only after first
load). `_init_default_admin` ensures a `demo` admin user exists on first start
so the login screen is reachable.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Iterable

from PIL import Image

from config import settings

log = logging.getLogger("demo")

_ASSETS_DIR = Path(__file__).parent / "demo_assets"
_PHOTOS_DIR = _ASSETS_DIR / "photos"
_FIXTURE_PATH = _ASSETS_DIR / "fixture.json"


# --- Fixture loading -------------------------------------------------------

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _build_fixture(photos: list[dict]) -> dict:
    """Return a dict shaped exactly like a Proton bridge response."""
    timeline = []
    nodes = {}
    albums: dict[str, dict] = {}
    for p in photos:
        uid = p["uid"]
        node = {
            "uid": uid,
            "name": p["name"],
            "mediaType": p["mediaType"],
            "captureTime": p["captureTime"],
            "sha1": p["sha1"],
            "size": p["size"],
            "albums": p.get("albums", []),
        }
        if "gps" in p:
            node["gps"] = p["gps"]
        timeline.append(node)
        nodes[uid] = node
        for a in p.get("albums", []):
            albums.setdefault(a, {"uid": a, "name": a})
    return {"timeline": timeline, "nodes": nodes, "albums": list(albums.values())}


def _load_fixture() -> dict:
    """Load the fixture (cached on first call)."""
    global _FIXTURE_CACHE
    if _FIXTURE_CACHE is not None:
        return _FIXTURE_CACHE
    with open(_FIXTURE_PATH, "r", encoding="utf-8") as fh:
        photos = json.load(fh)
    _FIXTURE_CACHE = _build_fixture(photos)
    log.info("demo: loaded %d photos from fixture", len(photos))
    return _FIXTURE_CACHE


_FIXTURE_CACHE: dict | None = None


# --- Response object that mimics httpx.Response for full_photo -----------

class _BytesResponse:
    """Mimics the bits of `httpx.Response` that `indexer._fullres_loop` uses.

    `iter_bytes`, `raise_for_status`, `close`, `status_code`, `headers`.
    Sufficient for the indexer to consume; full API compatibility isn't required.
    """
    def __init__(self, data: bytes, media_type: str = "image/jpeg") -> None:
        self._data = data
        self._closed = False
        self.status_code = 200
        self.headers = {"content-type": media_type, "content-length": str(len(data))}

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise RuntimeError(f"demo full_photo {self.status_code}")

    def iter_bytes(self, chunk_size: int = 65536) -> Iterable[bytes]:
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]

    def close(self) -> None:
        self._closed = True


# --- DemoBridge ------------------------------------------------------------

class DemoBridge:
    """A fake proton-bridge that serves curated fixture data from disk."""

    def __init__(self) -> None:
        if not _PHOTOS_DIR.exists() or not _FIXTURE_PATH.exists():
            raise RuntimeError(
                f"Demo assets missing. Expected {_PHOTOS_DIR} and {_FIXTURE_PATH}. "
                "Run scripts/fetch-demo-assets.py to populate them."
            )
        self._fixture = _load_fixture()

    # ----- health / sync -----

    def health(self) -> dict:
        return {"ok": True, "loggedIn": True, "demo": True}

    def timeline(self, limit: int = 0) -> list[dict]:
        items = list(self._fixture["timeline"])
        items.sort(key=lambda n: n.get("captureTime") or 0, reverse=True)
        if limit and limit > 0:
            items = items[:limit]
        return items

    def timeline_ids(self) -> list[dict]:
        return [
            {"uid": n["uid"], "captureTime": n.get("captureTime")}
            for n in self._fixture["timeline"]
        ]

    def nodes(self, uids: list[str]) -> list[dict]:
        out = []
        for uid in uids:
            n = self._fixture["nodes"].get(uid)
            if n is not None:
                out.append(n)
        return out

    def albums(self) -> dict:
        return {"albums": self._fixture["albums"]}

    # ----- thumbnail download (writes into DATA_DIR/work) -----

    def thumbnails(self, uids: list[str]) -> dict:
        """Convert each fixture JPEG to WebP and drop it into work_dir.

        The indexer's downloader expects `ok=True` once the work file is on disk
        regardless of the response shape, so we return a per-uid dict in the
        same shape as the real bridge.
        """
        results: list[dict] = []
        work_dir = settings.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        for uid in uids:
            src = _PHOTOS_DIR / f"{uid}.jpg"
            if not src.exists():
                results.append({"uid": uid, "ok": False, "error": "fixture photo missing"})
                continue
            dest = work_dir / f"{uid}.webp"
            try:
                with Image.open(src) as img:
                    img.thumbnail((512, 512))
                    img = img.convert("RGB")
                    img.save(dest, format="WEBP", quality=82, method=6)
            except Exception as exc:  # pragma: no cover
                log.warning("demo: thumbnail encode failed for %s: %s", uid, exc)
                results.append({"uid": uid, "ok": False, "error": str(exc)[:200]})
                continue
            results.append({"uid": uid, "ok": True})
        return {"results": results}

    # ----- full-resolution stream (for HEIC / video fallback) -----

    def full_photo(self, uid: str, range_header: str | None = None) -> _BytesResponse:
        """Return the raw JPEG bytes for a fixture photo.

        Used by the fullres fallback path (HEIC / video). Demo photos are all
        JPEG so we just hand the bytes back.
        """
        src = _PHOTOS_DIR / f"{uid}.jpg"
        if not src.exists():
            return _BytesResponse(b"", "image/jpeg")  # empty body, 200 OK; indexer treats as no preview
        return _BytesResponse(src.read_bytes(), "image/jpeg")

    def close(self) -> None:
        pass  # stateless

    # ----- bridge SDK cache management (admin tooling) -----
    #
    # DemoBridge is stateless so it has no SDK caches to report on or
    # clear. Mirror the surface so the admin check + clear button work in
    # demo mode without branching everywhere.

    def cache_status(self) -> dict:
        return {"ok": True, "files": [], "uptimeSec": 0, "demo": True}

    def clear_cache(self) -> dict:
        return {"ok": True, "removed": [], "demo": True}


# --- Auto-create a default admin on first start ----------------------------

def ensure_default_admin() -> None:
    """Create a default `demo` admin user when DEMO_MODE=1 and no admin exists.

    Idempotent: skips if a `demo` user is already there, or if any other admin
    user exists. Logs the credentials once so they're easy to spot in the
    container logs.
    """
    import store
    from auth import hash_password

    store.init_db()

    if store.get_user_by_username("demo") is not None:
        return
    # Any admin already? Leave the user to create their own.
    for u in store.list_users():
        if u["role"] == "admin" and not u["disabled"]:
            return

    password = os.environ.get("DEMO_ADMIN_PASSWORD", "proton-faces")
    store.create_user(
        username="demo",
        password_hash=hash_password(password),
        role="admin",
        display_name="Demo Admin",
    )
    # DEMO_LOGIN_LOGS controls whether the password is logged at WARN.
    # Default ON in pure-demo mode; OFF in DEMO_HARDENING_MODE / public demo.
    from auth import demo_login_logs
    if demo_login_logs():
        log.warning("=" * 60)
        log.warning("DEMO_MODE: created default admin user")
        log.warning("  username: demo")
        log.warning("  password: %s", password)
        log.warning("  (set DEMO_ADMIN_PASSWORD=... to override)")
        log.warning("=" * 60)
    else:
        log.warning("DEMO_MODE: created default admin user (username=demo, "
                    "password hidden — set DEMO_LOGIN_LOGS=1 to log it)")


# --- Inject demo-only metadata after sync --------------------------------

def apply_demo_gps() -> int:
    """Write GPS/place from the demo fixture into SQLite for each photo.

    In real Proton usage GPS comes from a Google Takeout export (see
    ``backfill_gps``). The demo fixture embeds GPS directly so the Places view
    works without any external data. Idempotent.
    """
    import sqlite3

    fixture = _load_fixture()
    updated = 0
    with sqlite3.connect(settings.db_path, timeout=30) as conn:
        for node in fixture["timeline"]:
            gps = node.get("gps")
            if not gps:
                continue
            place = gps.get("place")
            cur = conn.execute(
                "UPDATE photos SET gps_lat=?, gps_lng=?, place=? "
                "WHERE uid=? AND (gps_lat IS NULL OR gps_lng IS NULL)",
                (gps["lat"], gps["lng"], place, node["uid"]),
            )
            if cur.rowcount:
                updated += 1
        conn.commit()
    if updated:
        log.info("demo: applied GPS for %d fixture photos", updated)
    return updated

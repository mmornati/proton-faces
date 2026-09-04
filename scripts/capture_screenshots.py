#!/usr/bin/env python3
"""Capture screenshots + short screencasts of every major proton-faces UI view.

Drives a headless Chromium against a running instance (typically the demo
container on http://localhost:8080), logs in, and walks every view — saving
PNG screenshots into docs/assets/screenshots/ and MP4 screencasts into
docs/assets/screencasts/.

Prerequisites:
  • The proton-faces demo (or real) instance is up at BASE_URL.
  • A user exists with username=DEMO_USERNAME password=DEMO_PASSWORD
    (defaults: demo / proton-faces).
  • The indexer has had time to populate: photos with `status='done'` and
    ideally some people clusters + GPS enrichment.

Usage:
  python scripts/capture_screenshots.py              # capture everything
  python scripts/capture_screenshots.py --only login,photos,people
  python scripts/capture_screenshots.py --videos    # also capture screencasts
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

ROOT = Path(__file__).resolve().parent.parent
SHOTS_DIR = ROOT / "docs" / "assets" / "screenshots"
VIDS_DIR = ROOT / "docs" / "assets" / "screencasts"
TMP_VIDS = Path("/tmp/pf-videos")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
USER = os.environ.get("DEMO_USERNAME", "demo")
PASS = os.environ.get("DEMO_PASSWORD", "proton-faces")
VIEWPORT = {"width": 1440, "height": 900}
VIDEO_SIZE = {"width": 1440, "height": 900}


# --- helpers ---------------------------------------------------------------

def login(page) -> None:
    """Drive the login modal."""
    page.wait_for_selector("#login-username", timeout=15000)
    page.fill("#login-username", USER)
    page.fill("#login-password", PASS)
    page.click("#login-submit")
    # Login triggers a location.reload(); wait for the nav to appear.
    page.wait_for_selector("nav a[data-view='photos']", timeout=20000)
    page.wait_for_function(
        "() => document.querySelector('nav a[data-view=\"photos\"]') !== null",
        timeout=20000,
    )


def click_nav(page, view: str) -> None:
    page.evaluate(
        f"""
        (() => {{
            const link = document.querySelector(`nav a[data-view="{view}"]`);
            if (link) link.click();
        }})()
        """
    )


def capture_screenshot(page, name: str) -> Path:
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SHOTS_DIR / f"{name}.png"
    # Give thumbnail tiles a real chance to finish loading before shooting,
    # so the docs screenshots don't show "missing" (still-loading) photos.
    try:
        wait_for_images(page)
    except PWTimeout:
        pass  # never let one slow image abort the capture
    page.wait_for_timeout(300)
    page.screenshot(path=str(out), full_page=False)
    print(f"  ✓ {out.relative_to(ROOT)}")
    return out


def wait_for_cards(page, sel: str, min_count: int = 4, timeout_ms: int = 20000) -> None:
    page.wait_for_function(
        f"() => document.querySelectorAll('{sel}').length >= {min_count}",
        timeout=timeout_ms,
    )


def wait_for_images(page, timeout_ms: int = 30000) -> None:
    """Force every <img> in the DOM to load, then wait until all are done.

    The UI sets ``loading="lazy"`` on thumbnail tiles, so images below the
    fold never fetch on their own and stay blank in a screenshot. We flip them
    to eager (idempotent) so they start loading immediately, then block until
    every image is complete — otherwise the docs screenshots show "missing"
    (still-loading / never-requested) photos.
    """
    page.wait_for_function(
        """() => {
            const imgs = Array.from(document.images);
            for (const i of imgs) { i.loading = "eager"; i.decoding = "sync"; }
            return imgs.every(i => i.complete === true || !i.src);
        }""",
        timeout=timeout_ms,
    )


def shot_login(page) -> None:
    page.evaluate("() => localStorage.clear()")
    page.goto(BASE_URL + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#loginmodal.open, #login-username", timeout=15000)
    page.wait_for_timeout(400)
    capture_screenshot(page, "login")
    # Re-login so subsequent captures still work.
    login(page)


def shot_photos(page) -> None:
    click_nav(page, "photos")
    wait_for_cards(page, "#view .card", min_count=8)
    page.wait_for_timeout(1500)
    capture_screenshot(page, "photos")


def shot_favorites(page) -> None:
    click_nav(page, "favorites")
    page.wait_for_timeout(800)
    capture_screenshot(page, "favorites")


def shot_archive(page) -> None:
    click_nav(page, "archive")
    page.wait_for_timeout(800)
    capture_screenshot(page, "archive")


def shot_people(page) -> None:
    click_nav(page, "people")
    wait_for_cards(page, "#view .person", min_count=2)
    page.wait_for_timeout(1000)
    capture_screenshot(page, "people")


def shot_person_detail(page) -> None:
    """Click the first person card and capture their photo grid + map button."""
    page.evaluate("""
        (() => {
            const el = document.querySelector('#view .person');
            if (el) el.click();
        })()
    """)
    page.wait_for_timeout(1000)
    capture_screenshot(page, "person-detail")


def shot_search(page) -> None:
    """Type 'dog' into the search bar and capture results."""
    click_nav(page, "photos")
    page.wait_for_timeout(400)
    page.fill("#q", "dog")
    page.locator("#q").first.press("Enter")
    page.wait_for_timeout(2500)
    capture_screenshot(page, "search-dog")
    # Clear search box for the next shots.
    page.fill("#q", "")
    page.wait_for_timeout(500)


def shot_face_search(page) -> None:
    """Open the 'Search by example' modal."""
    page.click("#btn-face")
    page.wait_for_selector("#facemodal", timeout=5000)
    page.wait_for_timeout(500)
    capture_screenshot(page, "face-search-upload")
    page.click("#faceclose")
    page.wait_for_timeout(300)


def shot_places(page) -> None:
    click_nav(page, "places")
    # Leaflet markers are images, so wait for at least one .leaflet-marker-icon.
    page.wait_for_function(
        "() => document.querySelectorAll('.leaflet-marker-icon, #view .place').length >= 2",
        timeout=20000,
    )
    page.wait_for_timeout(2500)  # tiles need a moment
    capture_screenshot(page, "places")


def shot_albums(page) -> None:
    click_nav(page, "albums")
    page.wait_for_timeout(1000)
    capture_screenshot(page, "albums")


def shot_tags(page) -> None:
    click_nav(page, "tags")
    page.wait_for_timeout(1000)
    capture_screenshot(page, "tags")


def shot_duplicates(page) -> None:
    click_nav(page, "duplicates")
    page.wait_for_timeout(1500)
    capture_screenshot(page, "duplicates")


def shot_unassigned(page) -> None:
    click_nav(page, "unassigned")
    page.wait_for_timeout(1200)
    capture_screenshot(page, "unassigned")


def shot_photo_detail(page) -> None:
    """Open a photo's detail panel."""
    click_nav(page, "photos")
    wait_for_cards(page, "#view .card", min_count=2)
    page.wait_for_timeout(500)
    page.evaluate("() => { const c = document.querySelector('#view .card'); if (c) c.click(); }")
    page.wait_for_timeout(1200)
    capture_screenshot(page, "photo-detail")
    # Close so the next shot starts clean.
    page.evaluate("() => { const b = document.querySelector('#dclose'); if (b) b.click(); }")
    page.wait_for_timeout(400)


def shot_face_tagging(page) -> None:
    """Open a photo with detected faces — capture the clickable face boxes."""
    click_nav(page, "people")
    wait_for_cards(page, "#view .person", min_count=2)
    page.evaluate("() => { const el = document.querySelector('#view .person'); if (el) el.click(); }")
    page.wait_for_timeout(800)
    page.evaluate("() => { const c = document.querySelector('#view .card'); if (c) c.click(); }")
    page.wait_for_timeout(1500)
    capture_screenshot(page, "face-tagging")
    page.evaluate("() => { const b = document.querySelector('#dclose'); if (b) b.click(); }")
    page.wait_for_timeout(400)


def shot_admin(page) -> None:
    """Open the admin gear-icon modal and capture the Overview tab."""
    page.click("#btn-admin")
    page.wait_for_selector("#adminmodal, .admin", timeout=5000)
    page.wait_for_timeout(1500)
    capture_screenshot(page, "admin-overview")


def shot_admin_users(page) -> None:
    """Switch to the Users tab inside the admin modal."""
    page.evaluate("""
        (() => {
            const tabs = [...document.querySelectorAll('.tab, [role="tab"], button, .tablink')];
            const u = tabs.find(t => /^users$/i.test((t.textContent || '').trim()));
            if (u) u.click();
        })()
    """)
    page.wait_for_timeout(800)
    capture_screenshot(page, "admin-users")


def shot_admin_backups(page) -> None:
    page.evaluate("""
        (() => {
            const tabs = [...document.querySelectorAll('.tab, [role="tab"], button, .tablink')];
            const u = tabs.find(t => /backups/i.test(t.textContent || ''));
            if (u) u.click();
        })()
    """)
    page.wait_for_timeout(800)
    capture_screenshot(page, "admin-backups")


def shot_status(page) -> None:
    """Open the '?' status & diagnostics overlay."""
    # The status modal is opened via the help button in the header.
    page.evaluate("""
        (() => {
            const btn = document.querySelector('#helpbtn, [data-action="status"], .status-btn');
            if (btn) btn.click();
        })()
    """)
    page.wait_for_timeout(1200)
    capture_screenshot(page, "status-modal")


SCREENSHOTS = {
    "login": shot_login,
    "photos": shot_photos,
    "favorites": shot_favorites,
    "archive": shot_archive,
    "people": shot_people,
    "person-detail": shot_person_detail,
    "photo-detail": shot_photo_detail,
    "search": shot_search,
    "face-search": shot_face_search,
    "face-tagging": shot_face_tagging,
    "places": shot_places,
    "albums": shot_albums,
    "tags": shot_tags,
    "duplicates": shot_duplicates,
    "unassigned": shot_unassigned,
    "admin": shot_admin,
    "admin-users": shot_admin_users,
    "admin-backups": shot_admin_backups,
    "status": shot_status,
}


# --- screencast capture ---------------------------------------------------

SCREENCASTS = {
    "search-typing": lambda p: (
        click_nav(p, "photos"),
        p.wait_for_timeout(400),
        p.fill("#q", "dog"),
        p.wait_for_timeout(1500),
        p.fill("#q", "beach"),
        p.wait_for_timeout(1500),
        p.fill("#q", "Lille"),
        p.wait_for_timeout(1500),
    ),
    "name-a-face": lambda p: (
        click_nav(p, "people"),
        p.wait_for_timeout(800),
        p.evaluate("() => { const el = document.querySelector('#view .person'); if (el) el.click(); }"),
        p.wait_for_timeout(800),
        p.evaluate("() => { const c = document.querySelector('#view .card'); if (c) c.click(); }"),
        p.wait_for_timeout(800),
    ),
    "places": lambda p: (
        click_nav(p, "places"),
        p.wait_for_timeout(2500),
        p.evaluate("() => { const m = document.querySelector('.leaflet-marker-icon'); if (m) m.click(); }"),
        p.wait_for_timeout(1500),
    ),
    "admin-backup": lambda p: (
        p.click("#btn-admin"),
        p.wait_for_timeout(800),
        p.evaluate("""() => {
            const t = [...document.querySelectorAll('.tab, [role=tab], button')].find(
                e => /backups/i.test(e.textContent || ''));
            if (t) t.click();
        }"""),
        p.wait_for_timeout(800),
        p.evaluate("""() => {
            const b = [...document.querySelectorAll('button')].find(
                e => /backup now/i.test(e.textContent || ''));
            if (b) b.click();
        }"""),
        p.wait_for_timeout(2500),
    ),
    "duplicate-hide": lambda p: (
        click_nav(p, "duplicates"),
        p.wait_for_timeout(1500),
        p.evaluate("""() => {
            const b = [...document.querySelectorAll('button')].find(
                e => /hide/i.test(e.textContent || ''));
            if (b) b.click();
        }"""),
        p.wait_for_timeout(1500),
    ),
}


def record_screencasts(p, only: list[str] | None = None) -> None:
    VIDS_DIR.mkdir(parents=True, exist_ok=True)
    if TMP_VIDS.exists():
        shutil.rmtree(TMP_VIDS)
    TMP_VIDS.mkdir(parents=True, exist_ok=True)
    targets = only or list(SCREENCASTS.keys())
    ffmpeg = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")
    for name in targets:
        fn = SCREENCASTS.get(name)
        if not fn:
            continue
        recdir = TMP_VIDS / name
        recdir.mkdir(exist_ok=True)
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport=VIDEO_SIZE,
            record_video_dir=str(recdir),
            record_video_size=VIDEO_SIZE,
            reduced_motion="reduce",
            color_scheme="dark",
        )
        page = ctx.new_page()
        page.goto(BASE_URL + "/", wait_until="domcontentloaded")
        try:
            login(page)
            fn(page)
            try:
                wait_for_images(page)
            except PWTimeout:
                pass
            page.wait_for_timeout(2000)
        except Exception as exc:
            print(f"  ✗ {name}: {exc}", file=sys.stderr)
        ctx.close()
        browser.close()

        webms = list(recdir.glob("*.webm"))
        if not webms:
            print(f"  ! {name}: no video recorded")
            continue
        webm = webms[0]
        mp4 = VIDS_DIR / f"{name}.mp4"
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-i", str(webm),
                    "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-an",
                    str(mp4),
                ],
                check=True,
            )
            print(f"  ✓ {mp4.relative_to(ROOT)} ({mp4.stat().st_size // 1024} KB)")
        except FileNotFoundError:
            print("  ! ffmpeg not found — leaving webm in place")
            webm.rename(VIDS_DIR / f"{name}.webm")
        except subprocess.CalledProcessError as exc:
            print(f"  ✗ ffmpeg failed for {name}: {exc}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated subset of views to capture")
    ap.add_argument("--videos", action="store_true",
                    help="also record short MP4 screencasts (requires ffmpeg)")
    args = ap.parse_args()

    targets = (args.only.split(",") if args.only else list(SCREENSHOTS.keys()))
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport=VIEWPORT,
            reduced_motion="reduce",
            color_scheme="dark",
        )
        page = ctx.new_page()
        page.goto(BASE_URL + "/", wait_until="domcontentloaded")
        login(page)

        for name in targets:
            fn = SCREENSHOTS.get(name)
            if not fn:
                print(f"  ! skipping unknown view '{name}'", file=sys.stderr)
                continue
            try:
                fn(page)
            except Exception as exc:
                print(f"  ✗ {name}: {exc}", file=sys.stderr)

        ctx.close()
        browser.close()

        if args.videos:
            print()
            print("Recording screencasts…")
            record_screencasts(p,
                               only=args.only.split(",") if args.only else None)

    print()
    print(f"Done. Screenshots in {SHOTS_DIR.relative_to(ROOT)}")
    if args.videos:
        print(f"      Screencasts in {VIDS_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

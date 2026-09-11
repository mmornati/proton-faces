#!/usr/bin/env python3
"""Capture a looping GIF of the proton-faces mobile (PWA) experience.

Drives a headless Chromium at an iPhone-sized viewport against a running
instance (typically the demo container on http://localhost:8080), walks the
main views with tap-driven navigation, screenshots each step, composites every
frame inside an iPhone frame, and assembles a looping GIF for the docs.

Screenshots (not Playwright's record_video) are used because record_video
produces blank frames under mobile emulation.

Prerequisites:
  • The proton-faces demo (or real) instance is up at BASE_URL.
  • A user exists with username=DEMO_USERNAME / DEMO_PASSWORD
    (defaults: demo / proton-faces).
  • ffmpeg on PATH (or FFMPEG env var).

Usage:
  python scripts/capture_mobile_gif.py
  BASE_URL=http://localhost:8080 python scripts/capture_mobile_gif.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
GIF_DIR = ROOT / "docs" / "assets" / "screencasts"
TMP = Path("/tmp/pf-mobile-gif")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
USER = os.environ.get("DEMO_USERNAME", "demo")
PASS = os.environ.get("DEMO_PASSWORD", "proton-faces")
FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")

# iPhone 15-ish logical viewport. device_scale_factor=3 gives crisp 3x pixels;
# the GIF is downscaled to 2x for a reasonable file size.
VIEWPORT = {"width": 390, "height": 844}
DPR = 3
GIF_SCALE = 2  # final GIF pixels = logical * GIF_SCALE (780x1688)

# Bezel geometry (in final GIF pixels, i.e. logical * GIF_SCALE).
BEZEL = 26
CORNER = 64
ISLAND_W, ISLAND_H = 250, 66
ISLAND_TOP = 30

# How long each screenshot is held (in GIF frames) and how many frames the
# crossfade between two screenshots spans.
HOLD_FRAMES = 9
FADE_FRAMES = 4
FPS = 12
# Final GIF width; the iPhone frame is downscaled to this for a compact file.
GIF_WIDTH = 360


def login(page) -> None:
    page.wait_for_selector("#login-username", timeout=15000)
    page.fill("#login-username", USER)
    page.fill("#login-password", PASS)
    page.click("#login-submit")
    page.wait_for_selector("nav a[data-view='photos']", timeout=20000)


def tap_nav(page, view: str) -> None:
    page.evaluate(
        f"""
        (() => {{
            const link = document.querySelector(`nav a[data-view="{view}"]`);
            if (link) link.click();
        }})()
        """
    )


def wait_for_images(page, timeout_ms: int = 30000) -> None:
    """Force lazy <img>s to load so the GIF never shows blank tiles."""
    page.wait_for_function(
        """() => {
            const imgs = Array.from(document.images);
            for (const i of imgs) { i.loading = "eager"; i.decoding = "sync"; }
            return imgs.every(i => i.complete === true || !i.src);
        }""",
        timeout=timeout_ms,
    )


def settle(page, ms: int = 1200) -> None:
    try:
        wait_for_images(page)
    except PWTimeout:
        pass
    page.wait_for_timeout(ms)


def walk(page, shots: list[Path]) -> None:
    """The tap-driven mobile tour: login → photos → search → people → places → albums."""
    page.goto(BASE_URL + "/", wait_until="domcontentloaded")
    page.wait_for_timeout(1500)  # let the login screen settle
    page.screenshot(path=str(TMP / "s-login.png"))
    shots.append(TMP / "s-login.png")

    login(page)
    settle(page, 1500)
    page.screenshot(path=str(TMP / "s-photos.png"))
    shots.append(TMP / "s-photos.png")

    # Search: type a query, watch results re-rank.
    page.fill("#q", "dog")
    page.wait_for_timeout(2200)
    page.screenshot(path=str(TMP / "s-search-dog.png"))
    shots.append(TMP / "s-search-dog.png")
    page.fill("#q", "beach")
    page.wait_for_timeout(2200)
    page.screenshot(path=str(TMP / "s-search-beach.png"))
    shots.append(TMP / "s-search-beach.png")
    page.fill("#q", "")
    page.wait_for_timeout(600)

    # People grid.
    tap_nav(page, "people")
    settle(page, 1500)
    page.screenshot(path=str(TMP / "s-people.png"))
    shots.append(TMP / "s-people.png")

    # Places map.
    tap_nav(page, "places")
    page.wait_for_timeout(2800)
    page.screenshot(path=str(TMP / "s-places.png"))
    shots.append(TMP / "s-places.png")

    # Albums.
    tap_nav(page, "albums")
    settle(page, 1500)
    page.screenshot(path=str(TMP / "s-albums.png"))
    shots.append(TMP / "s-albums.png")

    # Back to the photos grid to close the loop.
    tap_nav(page, "photos")
    settle(page, 1500)
    page.screenshot(path=str(TMP / "s-photos-end.png"))
    shots.append(TMP / "s-photos-end.png")


def make_iphone_frame() -> Path:
    """Draw an iPhone frame PNG (screen area transparent) with PIL."""
    from PIL import Image, ImageDraw

    w, h = VIEWPORT["width"] * GIF_SCALE, VIEWPORT["height"] * GIF_SCALE
    fw, fh = w + BEZEL * 2, h + BEZEL * 2
    frame = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    d = ImageDraw.Draw(frame)

    # Bezel: near-black graphite with a subtle titanium edge.
    d.rounded_rectangle([0, 0, fw - 1, fh - 1], radius=CORNER, fill=(24, 24, 26, 255))
    d.rounded_rectangle(
        [BEZEL - 2, BEZEL - 2, fw - BEZEL + 1, fh - BEZEL + 1],
        radius=CORNER - 2,
        outline=(58, 58, 62, 255),
        width=2,
    )

    # Punch out the screen area so the app content shows through.
    d.rounded_rectangle(
        [BEZEL, BEZEL, BEZEL + w - 1, BEZEL + h - 1],
        radius=CORNER - BEZEL,
        fill=(0, 0, 0, 0),
    )

    # Side buttons.
    btn_w, btn_h, btn_r = 4, 90, 2
    d.rounded_rectangle([0, 260, btn_w, 260 + btn_h], radius=btn_r, fill=(24, 24, 26, 255))
    d.rounded_rectangle([0, 400, btn_w, 400 + btn_h], radius=btn_r, fill=(24, 24, 26, 255))
    d.rounded_rectangle([fw - btn_w, 300, fw, 300 + btn_h], radius=btn_r, fill=(24, 24, 26, 255))

    # Dynamic island.
    ix = (fw - ISLAND_W) // 2
    d.rounded_rectangle(
        [ix, ISLAND_TOP, ix + ISLAND_W, ISLAND_TOP + ISLAND_H], radius=ISLAND_H // 2, fill=(0, 0, 0, 255)
    )

    out = TMP / "iphone-frame.png"
    frame.save(out)
    return out


def composite(shot: Path, phone) -> Path:
    """Paste one app screenshot into the iPhone frame; return the composited PNG."""
    from PIL import Image

    w, h = VIEWPORT["width"] * GIF_SCALE, VIEWPORT["height"] * GIF_SCALE
    app = Image.open(shot).convert("RGB").resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGBA", phone.size, (0, 0, 0, 0))
    canvas.paste(app, (BEZEL, BEZEL))
    canvas = Image.alpha_composite(canvas, phone)
    out = TMP / "composited" / f"c-{shot.stem}.png"
    canvas.convert("RGB").save(out)
    return out


def build_frames(shots: list[Path], phone) -> list[Path]:
    """Expand the screenshots into a frame sequence with crossfades."""
    from PIL import Image

    frames: list[Path] = []
    comps = [composite(s, phone) for s in shots]
    imgs = [Image.open(c).convert("RGB") for c in comps]

    def emit(img: Image.Image, tag: str) -> None:
        out = TMP / "frames" / f"f-{len(frames):05d}.png"
        img.save(out)
        frames.append(out)

    for i, img in enumerate(imgs):
        for _ in range(HOLD_FRAMES):
            emit(img, f"hold{i}")
        if i + 1 < len(imgs):
            nxt = imgs[i + 1]
            for k in range(1, FADE_FRAMES + 1):
                t = k / (FADE_FRAMES + 1)
                blended = Image.blend(img, nxt, t)
                emit(blended, f"fade{i}-{k}")
    return frames


def assemble_gif(frames: list[Path], out: Path) -> None:
    """Assemble the frame sequence into a looping GIF via ffmpeg."""
    scale = f"scale={GIF_WIDTH}:-1"
    subprocess.run(
        [
            FFMPEG, "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", str(TMP / "frames" / "f-%05d.png"),
            "-vf", f"{scale},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=none",
            "-loop", "0",
            str(out),
        ],
        check=True,
    )


def main() -> int:
    global FPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=FPS, help="GIF frame rate (default 12)")
    args = ap.parse_args()
    FPS = args.fps

    GIF_DIR.mkdir(parents=True, exist_ok=True)
    if TMP.exists():
        shutil.rmtree(TMP)
    (TMP / "composited").mkdir(parents=True)
    (TMP / "frames").mkdir(parents=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=DPR,
            is_mobile=True,
            has_touch=True,
            reduced_motion="reduce",
            color_scheme="dark",
        )
        page = ctx.new_page()
        shots: list[Path] = []
        walk(page, shots)
        ctx.close()
        browser.close()

    print(f"  captured {len(shots)} screenshots")
    from PIL import Image

    phone = Image.open(make_iphone_frame()).convert("RGBA")
    frames = build_frames(shots, phone)
    print(f"  built {len(frames)} frames")
    out = GIF_DIR / "mobile-browse.gif"
    assemble_gif(frames, out)
    print(f"  ✓ {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB, {len(frames)} frames @ {FPS}fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

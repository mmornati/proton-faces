"""Offline reverse geocoding for GPS coordinates (city / place names).

Uses the `reverse_geocoder` package (bundled GeoNames cities1000 dataset, no
network required after install). Wraps it defensively: if the package or its
dataset is unavailable, returns None instead of crashing.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("geocode")

_lock = threading.Lock()
_rg = None


def _load():
    global _rg
    if _rg is not None:
        return _rg
    with _lock:
        if _rg is not None:
            return _rg
        try:
            import reverse_geocoder as rg

            rg.search((0.0, 0.0))  # warm up + force dataset load
            _rg = rg
            log.info("reverse_geocoder ready")
        except Exception as exc:  # pragma: no cover
            log.warning("reverse_geocoder unavailable: %s", exc)
            _rg = False  # cached negative
        return _rg


def reverse_geocode(lat: float, lng: float) -> str | None:
    """Return a human place name for a coordinate, e.g. 'Lille, Hauts-de-France'."""
    rg = _load()
    if not rg or lat is None or lng is None:
        return None
    try:
        result = rg.search((lat, lng))
        if result and result[0]:
            r = result[0]
            parts = [p for p in (r.get("name"), r.get("admin1")) if p]
            return ", ".join(parts) if parts else None
    except Exception as exc:  # pragma: no cover
        log.debug("geocode failed for (%s,%s): %s", lat, lng, exc)
    return None


def reverse_geocode_many(points: list[tuple[float, float]]) -> dict[tuple[float, float], str | None]:
    """Batch reverse geocoding with a coordinate-level cache.

    Accepts a list of (lat, lng) tuples (coords may repeat across photos of the
    same city) and returns {coord: place}. Coords are rounded to 4 decimals
    (~11m) so nearby photos share a single lookup and result.
    """
    rg = _load()
    if not rg:
        return {p: None for p in points}

    rounded = [(round(p[0], 4), round(p[1], 4)) for p in points]
    results: dict[tuple[float, float], str | None] = {}
    todo: list[tuple[float, float]] = []
    for coord in rounded:
        if coord not in results:
            results[coord] = None
            todo.append(coord)

    try:
        batch = rg.search(todo) if todo else []
        for coord, result in zip(todo, batch):
            if not result:
                continue
            parts = [p for p in (result.get("name"), result.get("admin1")) if p]
            results[coord] = ", ".join(parts) if parts else None
    except Exception as exc:  # pragma: no cover
        log.warning("batch geocode failed (%d coords): %s", len(todo), exc)

    # Map original (unrounded) points onto their rounded result.
    return {p: results[round(p[0], 4), round(p[1], 4)] for p in points}
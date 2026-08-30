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
"""HTTP client for the proton-bridge container.

When `DEMO_MODE=1` is set, `get_bridge()` returns a `DemoBridge` instance
instead — see `demo.py` for the details. The switch is transparent to callers:
the real bridge and the demo expose the same method surface.
"""
from __future__ import annotations

import json
import os

import httpx
from config import settings


class BridgeError(Exception):
    pass


class BridgeTransientError(BridgeError):
    """Raised when the bridge returned a transient error (429/502/503/etc.)
    that the caller should retry after a backoff rather than treating as a
    permanent failure.

    `retry_after_sec` is the wait time suggested by the upstream
    `Retry-After` header (clamped to [0, 600] seconds). 0 if the server
    didn't include a header.
    """

    def __init__(self, status_code: int, message: str, retry_after_sec: float = 0):
        super().__init__(f"{status_code} {message} (retry_after={retry_after_sec}s)")
        self.status_code = status_code
        self.retry_after_sec = retry_after_sec


def _parse_retry_after(value: str | None) -> float:
    """Parse an HTTP Retry-After header value. Returns seconds (clamped 0-600).

    Supports both delta-seconds ("120") and HTTP-date formats; the date form
    is rare in practice for Proton Drive but we handle it for completeness.
    """
    if not value:
        return 0
    try:
        sec = float(value)
    except ValueError:
        # HTTP-date form: "Wed, 21 Oct 2026 07:28:00 GMT"
        try:
            from email.utils import parsedate_to_datetime
            target = parsedate_to_datetime(value)
            if target is None:
                return 0
            import datetime as _dt
            now = _dt.datetime.now(target.tzinfo)
            sec = (target - now).total_seconds()
        except Exception:
            return 0
    return max(0.0, min(600.0, sec))


class BridgeClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.bridge_url).rstrip("/")
        self._client = httpx.Client(timeout=120.0)

    def health(self) -> dict:
        r = self._client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def _ndjson_items(self, r: httpx.Response) -> list[dict]:
        items: list[dict] = []
        for line in r.iter_lines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
        return items

    def timeline(self, limit: int = 0) -> list[dict]:
        """Return the photo timeline as a list of photo nodes.

        limit > 0 restricts to the `limit` most recent photos (useful for
        incremental testing); 0 fetches everything.

        The bridge streams the response as newline-delimited JSON (one node per
        line); comment lines starting with '#' are progress/keep-alive markers
        and are skipped. We read the stream incrementally so the long fetch
        never times out on the client side either.
        """
        params = {"limit": limit} if limit > 0 else None
        with self._client.stream(
            "GET",
            f"{self.base_url}/timeline",
            params=params,
            timeout=httpx.Timeout(3600.0, connect=30.0),
        ) as r:
            r.raise_for_status()
            items = self._ndjson_items(r)
        if not isinstance(items, list):
            raise BridgeError(f"timeline returned unexpected payload: {items!r}")
        return items

    def timeline_ids(self) -> list[dict]:
        """Return only {uid, captureTime} for every photo in the timeline.

        Cheap: no per-node metadata decryption, so it completes in seconds even
        on a large library. Used to diff against the local index without
        re-fetching (and re-decrypting) full metadata for every photo.
        """
        with self._client.stream(
            "GET",
            f"{self.base_url}/timeline/ids",
            timeout=httpx.Timeout(3600.0, connect=30.0),
        ) as r:
            r.raise_for_status()
            return self._ndjson_items(r)

    def nodes(self, uids: list[str]) -> list[dict]:
        """Fetch full metadata for specific photo uids (NDJSON streamed)."""
        if not uids:
            return []
        with self._client.stream(
            "POST",
            f"{self.base_url}/nodes",
            json={"uids": uids},
            timeout=httpx.Timeout(3600.0, connect=30.0),
        ) as r:
            r.raise_for_status()
            return self._ndjson_items(r)

    def albums(self) -> dict:
        """Return all albums as {uid, name} pairs (plain JSON)."""
        r = self._client.get(f"{self.base_url}/albums", timeout=httpx.Timeout(300.0, connect=30.0))
        r.raise_for_status()
        return r.json()

    def thumbnails(self, uids: list[str]) -> dict:
        """Ask the bridge to download Type1 thumbnails into DATA_DIR/work/.

        The bridge is synchronous: by the time it responds, every `ok` uid has
        its WebP written on the shared volume.
        """
        r = self._client.post(f"{self.base_url}/thumbnails", json={"uids": uids})
        r.raise_for_status()
        return r.json()

    def full_photo(self, uid: str, range_header: str | None = None,
                   timeout_ms: int | None = None) -> httpx.Response:
        """Stream a full-resolution photo (read-only, on demand).

        ``range_header`` (e.g. ``bytes=0-``) is forwarded so HTTP Range
        seeking works end-to-end for videos.

        ``timeout_ms`` bounds how long the bridge holds a download queue slot
        for this request. When the caller has its own hard timeout (the API's
        ``_FULL_TIMEOUT_SEC``), it should pass the same value so the bridge
        aborts its SDK download shortly after the client gives up, instead of
        pinning the slot until the bridge's own ceiling. ``None`` leaves the
        bridge's FULL_RES_TIMEOUT_MS in place (used by the indexer, which
        needs the whole file).

        For transient errors (429 / 502 / 503) we raise
        `BridgeTransientError` carrying the parsed Retry-After value so the
        caller can back off without flagging the photo as `error`. Other
        non-2xx statuses still raise `httpx.HTTPStatusError` via the caller's
        `resp.raise_for_status()` so behavior is preserved for permanent
        failures (e.g. 401, 404).

        Timeouts: 30s for the response headers (long enough for Proton's
        normal handshake, short enough that a hung bridge doesn't pile up
        open connections); no overall read timeout so the body can stream.
        """
        # Streaming: return the raw response so the caller can iterate the body
        # as it arrives (full-res downloads can be slow; don't buffer them).
        headers = {"Range": range_header} if range_header else {}
        if timeout_ms:
            headers["X-Timeout-Ms"] = str(timeout_ms)
        req = self._client.build_request(
            "GET",
            f"{self.base_url}/photo/{uid}/full",
            headers=headers,
            timeout=httpx.Timeout(30.0, connect=10.0, read=None, write=None),
        )
        resp = self._client.send(req, stream=True)
        # Eagerly check status so the caller doesn't have to drain the stream
        # before learning the bridge is rate-limiting them. We only re-raise
        # transient statuses; permanent ones are still surfaced via
        # resp.raise_for_status() in the caller.
        if resp.status_code in (429, 502, 503):
            retry_after = _parse_retry_after(resp.headers.get("Retry-After") or resp.headers.get("retry-after"))
            resp.close()
            msg = resp.headers.get("X-Error-Message", "")
            raise BridgeTransientError(resp.status_code, msg or "upstream transient error", retry_after)
        return resp

    def close(self) -> None:
        self._client.close()

    # ----- bridge SDK cache management (admin tooling) -----

    def cache_status(self) -> dict:
        """Ask the bridge for the on-disk SDK cache state.

        Returns `{"ok": True, "files": [{"name": "cache-crypto.sqlite", "size": N, "mtime": ts}, ...],
                   "uptimeSec": N}` so the admin "stale cache" check can flag
        a hung getFileDownloader without scraping bridge logs.

        Short timeout (5s) — this is called from the admin checks UI; we
        don't want it to hang the whole checks panel if the bridge is wedged.
        """
        r = self._client.get(f"{self.base_url}/cache", timeout=httpx.Timeout(5.0, connect=5.0))
        r.raise_for_status()
        return r.json()

    def clear_cache(self) -> dict:
        """Tell the bridge to unlink its SDK caches and exit.

        compose's `restart: unless-stopped` policy respawns the bridge with
        a fresh cache, fixing the "stale cache after a Proton incident"
        hang (see docs/reference/troubleshooting.md).

        The response should arrive before the bridge exits, but we use a
        short read timeout in case the bridge is wedged at the moment of
        the call — we still want a clean error rather than an indefinite
        hang on the admin click.
        """
        r = self._client.post(
            f"{self.base_url}/cache/clear",
            timeout=httpx.Timeout(5.0, connect=5.0),
        )
        r.raise_for_status()
        return r.json()


_bridge: BridgeClient | "DemoBridge" | None = None  # type: ignore[name-defined]


def get_bridge():
    global _bridge
    if _bridge is None:
        if os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on"):
            from demo import DemoBridge

            _bridge = DemoBridge()
        else:
            _bridge = BridgeClient()
    return _bridge
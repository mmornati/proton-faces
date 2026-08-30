"""HTTP client for the proton-bridge container."""
from __future__ import annotations

import json

import httpx
from config import settings


class BridgeError(Exception):
    pass


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

    def thumbnails(self, uids: list[str]) -> dict:
        """Ask the bridge to download Type1 thumbnails into DATA_DIR/work/.

        The bridge is synchronous: by the time it responds, every `ok` uid has
        its WebP written on the shared volume.
        """
        r = self._client.post(f"{self.base_url}/thumbnails", json={"uids": uids})
        r.raise_for_status()
        return r.json()

    def full_photo(self, uid: str) -> httpx.Response:
        """Stream a full-resolution photo (read-only, on demand)."""
        # Streaming: return the raw response so the caller can iterate the body
        # as it arrives (full-res downloads can be slow; don't buffer them).
        req = self._client.build_request(
            "GET",
            f"{self.base_url}/photo/{uid}/full",
            timeout=httpx.Timeout(1800.0, connect=30.0),
        )
        return self._client.send(req, stream=True)

    def close(self) -> None:
        self._client.close()


_bridge: BridgeClient | None = None


def get_bridge() -> BridgeClient:
    global _bridge
    if _bridge is None:
        _bridge = BridgeClient()
    return _bridge
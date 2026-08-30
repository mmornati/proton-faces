"""HTTP client for the proton-bridge container."""
from __future__ import annotations

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

    def timeline(self, limit: int = 0) -> list[dict]:
        """Return the photo timeline as a list of photo nodes.

        limit > 0 restricts to the `limit` most recent photos (useful for
        incremental testing); 0 fetches everything.
        """
        params = {"limit": limit} if limit > 0 else None
        r = self._client.get(f"{self.base_url}/timeline", params=params)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise BridgeError(f"timeline returned unexpected payload: {data!r}")
        return data

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
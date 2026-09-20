"""ASGI middleware that rejects oversized request bodies before they are parsed.

Every JSON route reads its body with ``Body(...)``; without a ceiling an
unauthenticated client can POST hundreds of MB to ``/api/auth/login`` and
have FastAPI buffer + ``json.loads`` all of it. Two checks:

* ``Content-Length`` above the limit → immediate 413, nothing is read.
* No ``Content-Length`` (chunked) → bytes are counted as the app reads them
  and a 413 is raised once the limit is crossed (Starlette's exception
  middleware turns it into a response because nothing has been sent yet).

Paths listed in ``exempt_prefixes`` (the multipart face-search upload,
which enforces its own byte cap) bypass both checks.
"""
from __future__ import annotations

from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int, exempt_prefixes: tuple[str, ...] = ()) -> None:
        self.app = app
        self.max_bytes = max(1, int(max_bytes))
        self.exempt_prefixes = exempt_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if any(path.startswith(p) for p in self.exempt_prefixes):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send, status=400, detail="invalid content-length")
                return

        seen = 0
        limit = self.max_bytes

        async def counting_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise HTTPException(413, "request body too large")
            return message

        await self.app(scope, counting_receive, send)

    @staticmethod
    async def _reject(send: Send, status: int = 413, detail: str = "request body too large") -> None:
        body = ('{"detail":"%s"}' % detail).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})

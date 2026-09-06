"""ASGI middleware that gzips JSON/UI responses, skipping binary media.

Self-contained (stdlib zlib only) so it behaves identically on the starlette
0.4x pinned by CI and the newer starlette in local dev — the upstream
GZipMiddleware signature changed between the two.
"""
from __future__ import annotations

import zlib

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_COMPRESSIBLE_PREFIXES = ("application/json", "text/")
_COMPRESSIBLE_EXACT = ("image/svg+xml",)


class CompressionMiddleware:
    def __init__(self, app: ASGIApp, minimum_size: int = 500, compresslevel: int = 9) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if "gzip" not in headers.get("Accept-Encoding", ""):
            await self.app(scope, receive, send)
            return
        await _GzipResponder(self.app, self.minimum_size, self.compresslevel)(scope, receive, send)


class _GzipResponder:
    def __init__(self, app: ASGIApp, minimum_size: int, compresslevel: int) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel
        self.send: Send | None = None
        self.initial_message: Message | None = None
        self.started = False
        self._skip = False
        self._compressor: zlib._Compress | None = None

    @property
    def compressor(self) -> zlib._Compress:
        if self._compressor is None:
            self._compressor = zlib.compressobj(self.compresslevel, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        return self._compressor

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.send = send
        await self.app(scope, receive, self._send)

    async def _send(self, message: Message) -> None:
        assert self.send is not None
        message_type = message["type"]
        if message_type == "http.response.start":
            # Don't send the initial message until we've decided whether to
            # rewrite the headers (Content-Encoding / Content-Length / Vary).
            self.initial_message = message
            headers = Headers(raw=message["headers"])
            self._skip = (
                "content-encoding" in headers
                or message["status"] == 206
                or not self._compressible(headers.get("content-type", ""))
            )
        elif message_type == "http.response.body":
            if self._skip:
                if not self.started:
                    self.started = True
                    await self.send(self.initial_message)
                await self.send(message)
            elif not self.started:
                self.started = True
                body = message.get("body", b"")
                more_body = message.get("more_body", False)
                if len(body) < self.minimum_size and not more_body:
                    # Don't apply compression to small outgoing responses.
                    await self.send(self.initial_message)
                    await self.send(message)
                elif not more_body:
                    body = self._compress(body, more_body=False)
                    headers = MutableHeaders(raw=self.initial_message["headers"])
                    headers.add_vary_header("Accept-Encoding")
                    if body != message["body"]:
                        headers["Content-Encoding"] = "gzip"
                        headers["Content-Length"] = str(len(body))
                        message["body"] = body
                    await self.send(self.initial_message)
                    await self.send(message)
                else:
                    # Initial body in a streaming response.
                    body = self._compress(body, more_body=True)
                    headers = MutableHeaders(raw=self.initial_message["headers"])
                    headers.add_vary_header("Accept-Encoding")
                    if body != message["body"]:
                        headers["Content-Encoding"] = "gzip"
                        del headers["Content-Length"]
                        message["body"] = body
                    await self.send(self.initial_message)
                    await self.send(message)
            else:
                # Remaining body in a streaming response.
                message["body"] = self._compress(
                    message.get("body", b""), more_body=message.get("more_body", False)
                )
                await self.send(message)
        elif message_type == "http.response.pathsend":
            # Don't apply gzip to pathsend responses.
            if self.initial_message is not None:
                await self.send(self.initial_message)
            await self.send(message)

    def _compressible(self, content_type: str) -> bool:
        media_type = content_type.partition(";")[0].strip().lower()
        return media_type.startswith(_COMPRESSIBLE_PREFIXES) or media_type in _COMPRESSIBLE_EXACT

    def _compress(self, body: bytes, *, more_body: bool) -> bytes:
        if more_body:
            return self.compressor.compress(body) + self.compressor.flush(zlib.Z_SYNC_FLUSH)
        return self.compressor.compress(body) + self.compressor.flush()

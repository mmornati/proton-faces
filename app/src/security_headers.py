"""ASGI middleware that adds security headers to every HTTP response.

Defense-in-depth for the single-file UI: XSS containment (CSP), clickjacking
protection (X-Frame-Options + frame-ancestors), MIME-sniffing prevention
(X-Content-Type-Options), referrer-leakage control (Referrer-Policy), a
restrictive Permissions-Policy, and HSTS when the deployment declares TLS.

The CSP is deliberately pragmatic: the SPA ships as one index.html with
inline <script>/<style> blocks, so script-src/style-src need 'unsafe-inline'.
Tightening script-src to a nonce or an external file is a follow-up (issue #47).
Leaflet is vendored (static/vendor), so no CDN origin is allow-listed; map
tiles come from OpenStreetMap and are the only external origin.
"""
from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_CSP = (
    "default-src 'self'; "
    "img-src 'self' data: blob: https://*.tile.openstreetmap.org; "
    "media-src 'self' blob:; "
    "style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)

_HEADERS = {
    "content-security-policy": _CSP,
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}

# One year, subdomains included. Only emitted when AUTH_COOKIE_SECURE=1 —
# the operator's declaration that the app is served over TLS — so a plain
# http:// dev or demo install never pins browsers to https.
_HSTS = "max-age=31536000; includeSubDomains"


def _hsts_enabled() -> bool:
    from config import settings

    return bool(settings.auth_cookie_secure)


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        await self.app(scope, receive, self._send_with_headers(send))

    def _send_with_headers(self, send: Send) -> Send:
        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                for name, value in _HEADERS.items():
                    if name not in headers:
                        headers[name] = value
                if _hsts_enabled() and "strict-transport-security" not in headers:
                    headers["strict-transport-security"] = _HSTS
                message["headers"] = headers.raw
            await send(message)

        return send_wrapper

"""Local multi-user authentication.

Proton does not publish an OAuth/OIDC provider for third-party services, so
this app implements its own (small) identity layer: family-member accounts
(username + bcrypt password) stored in the same SQLite index as the photos.
The Proton credentials stay where they always were — inside the bridge
container — and the single Proton session still drives the library. This
module just makes sure the web UI is gated to the humans allowed to use it.

Tokens are opaque random strings stored in ``auth_tokens`` and presented by
the client as ``Authorization: Bearer <token>``. Access tokens live ~8 h,
refresh tokens ~30 d. On restart nothing is lost — both kinds survive in
SQLite.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass

import bcrypt
from fastapi import Depends, HTTPException, Request, status

import store

log = logging.getLogger("auth")


# Role ordering: read ⊂ write ⊂ admin. Higher roles inherit all lower-role
# permissions; a `write` user can do everything a `read` user can, plus
# editing metadata. Anything below the requested rank returns 403.
ROLE_RANK = {"read": 1, "write": 2, "admin": 3}


@dataclass
class CurrentUser:
    """The authenticated user, attached to the request via ``request.state.user``."""
    id: int
    username: str
    display_name: str
    role: str

    def has_role(self, min_role: str) -> bool:
        return ROLE_RANK.get(self.role, 0) >= ROLE_RANK.get(min_role, 0)


# --- TTL config (env-overridable, default 8h access / 30d refresh) ---------

def access_ttl() -> int:
    return int(os.environ.get("AUTH_ACCESS_TTL", "28800"))     # 8 hours


def refresh_ttl() -> int:
    return int(os.environ.get("AUTH_REFRESH_TTL", "2592000"))  # 30 days


# --- password helpers ------------------------------------------------------

def hash_password(plain: str) -> str:
    """Bcrypt cost 12 (~250 ms on a modern CPU — slow enough to deter brute force)."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


# --- FastAPI dependencies --------------------------------------------------

def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(None, 1)[1].strip()
    return token or None


# --- signed URLs for binary endpoints -------------------------------------
# /thumb, /full, /cover, /crop are consumed by <img>/<video>/<source> tags
# that can't attach an Authorization header. Two release modes:
#
#   1. DEMO_ALLOW_PUBLIC_THUMBS=1 (default for the demo profile)
#      - the suffix endpoints stay world-readable so a static <img> tag
#        works without any JS round-trip.
#      - signed URLs are still issued and accepted (so the prod binary
#        can opt-in to signed-only at any time without breaking demos).
#
#   2. DEMO_ALLOW_PUBLIC_THUMBS=0 (recommended for prod)
#      - the suffix endpoints require either a valid bearer token OR a
#        valid short-lived signed URL (?sig=...&exp=...). Front-end
#        must call /api/sign once per page load and append ?sig=&exp=
#        to every <img src=...> URL it renders.

def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _hardening_overrides(key: str, demo_default: bool) -> bool:
    """Resolve a demo flag, with DEMO_HARDENING_MODE as the master switch.

    When DEMO_HARDENING_MODE=1, return the SAFE value (overriding whatever
    the env says) for flags that should be locked-down in public demos.
    The legacy opt-in flags (DEMO_ALLOW_PUBLIC_THUMBS=1) win when
    DEMO_HARDENING_MODE is explicitly 0.
    """
    from config import demo_hardening_mode  # local import: avoid cycle
    if demo_hardening_mode():
        # When hardening mode is on, force these flags to their safe values.
        if key in (
            "DEMO_ALLOW_PUBLIC_THUMBS",
            "DEMO_LOGIN_LOGS",
        ):
            return False
        if key in (
            "DEMO_DISABLE_ADMIN_USER_MANAGEMENT",
            "DEMO_DISABLE_BACKUPS",
        ):
            return True
    return _env_bool(key, demo_default)


def allow_public_thumbs() -> bool:
    """True when the suffix endpoints (/thumb,/full,/cover,/crop) are world-readable."""
    return _hardening_overrides("DEMO_ALLOW_PUBLIC_THUMBS", True)


def demo_disable_admin_user_management() -> bool:
    """True to hide /api/admin/users from /docs + 404 the endpoints.

    Recommended ON for public demos so an attacker can't enumerate the admin
    area from OpenAPI. The endpoints still work; they just don't appear in
    the auto-generated docs.
    """
    return _hardening_overrides("DEMO_DISABLE_ADMIN_USER_MANAGEMENT", False)


def demo_disable_backups() -> bool:
    """True to 404 every /api/admin/backup* endpoint.

    Backups contain a full VACUUM INTO of the SQLite index, which includes
    every user, every face embedding, every photo UID. Disable in public
    demos where the threat model allows an admin compromise.
    """
    return _hardening_overrides("DEMO_DISABLE_BACKUPS", False)


def demo_login_logs() -> bool:
    """True to log demo admin credentials at WARN on first boot.

    Default OFF in prod; demo.py opts in via DEMO_LOGIN_LOGS=1 only when
    the deploy template sets it. The WARN line is convenient for ops but
    leaks the admin password to anyone with `docker compose logs` access.
    """
    return _hardening_overrides("DEMO_LOGIN_LOGS", True)


def demo_hardening_mode() -> bool:
    """Aggregate switch — see config.demo_hardening_mode()."""
    from config import demo_hardening_mode as _dhm
    return _dhm()


def _signing_secret() -> bytes:
    """Process-local secret used to sign short-lived URLs for binary assets.

    We pull from SIGNING_SECRET env (set by the operator / deploy template) and
    fall back to a per-boot random secret so a leaked secret only compromises
    URLs signed during that boot. Set SIGNING_SECRET explicitly in production
    so URLs survive a restart; rotate by setting a new value + invalidating
    existing cookies/sessions.
    """
    s = os.environ.get("SIGNING_SECRET", "").strip()
    if s:
        return s.encode("utf-8")
    # Per-boot fallback. We log a warning so this is never silent in prod.
    if not hasattr(_signing_secret, "_ephemeral"):
        import secrets
        _signing_secret._ephemeral = secrets.token_bytes(32)  # type: ignore[attr-defined]
        log.warning(
            "SIGNING_SECRET not set; using a per-boot ephemeral secret. "
            "Set SIGNING_SECRET in production so signed URLs survive restarts."
        )
    return _signing_secret._ephemeral  # type: ignore[attr-defined]


def make_signed_token(path: str, ttl_seconds: int = 300) -> tuple[str, int]:
    """Return (sig, exp) for a path. Path is the URL path WITHOUT query string."""
    exp = int(time.time()) + ttl_seconds
    msg = f"{path}|{exp}".encode("utf-8")
    sig = hmac.new(_signing_secret(), msg, hashlib.sha256).hexdigest()
    return sig, exp


def verify_signed_token(path: str, sig: str | None, exp: int | None) -> bool:
    """Constant-time check of a (sig, exp) pair against path."""
    if not sig or not exp:
        return False
    if exp < int(time.time()):
        return False
    msg = f"{path}|{exp}".encode("utf-8")
    expected = hmac.new(_signing_secret(), msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


# Paths that don't require authentication. Used by `require_user` to short-circuit
# before enforcing the bearer-token check, since FastAPI's route-level
# `dependencies=[]` doesn't override app-level ones.
_AUTH_FREE_PATHS = frozenset({
    "/api/auth/login",
    "/api/auth/refresh",
    "/api/auth/limits",
    "/api/health",
    "/api/status",
})

# Path-segment suffixes for routes that return binary responses (image/video
# bytes) consumed by <img>/<video>/<source> tags. Those browser primitives
# can't attach an Authorization header, so they have to be world-readable —
# the server is already gated by the login modal and (in front of Traefik)
# the auth-aware reverse proxy in real deployments.
_AUTH_FREE_BINARY_SUFFIXES = frozenset({
    "/thumb",
    "/full",
    "/cover",
    "/crop",
})


def _is_auth_free(path: str) -> bool:
    if path in _AUTH_FREE_PATHS:
        return True
    if not allow_public_thumbs():
        # In prod mode, the binary endpoints are NOT auth-free; they require
        # either a bearer token (handled by the route) or a valid signed URL
        # (validated by the route via verify_signed_token).
        return False
    # Demo mode (or explicitly opted-in): the binary endpoints stay world-readable
    # so static <img> tags work without JS.
    return any(path.endswith(s) for s in _AUTH_FREE_BINARY_SUFFIXES)


def signed_or_token(request: Request) -> CurrentUser | None:
    """Allow access to a binary endpoint if EITHER a valid bearer token OR a
    valid signed URL (?sig=&exp=) is presented.

    Used by the /thumb /full /cover /crop routes so they can be locked down in
    prod (DEMO_ALLOW_PUBLIC_THUMBS=0) while staying consumable by <img> tags via
    short-lived signed URLs issued by /api/sign.
    """
    sig = request.query_params.get("sig")
    exp_raw = request.query_params.get("exp")
    exp = int(exp_raw) if exp_raw and exp_raw.isdigit() else None
    if sig and verify_signed_token(request.url.path, sig, exp):
        return None  # Auth-free access via signed URL.
    # Fall back to bearer-token auth.
    return require_user(request)


def require_user(request: Request) -> CurrentUser | None:
    """Resolve the bearer token to a CurrentUser; raise 401 otherwise.

    Auth-free paths (`/api/auth/*`, `/api/health`, `/api/status`, plus the
    binary `/thumb`, `/full`, `/cover`, `/crop` tail endpoints in demo mode)
    short-circuit and return None — those routes don't need a user. All other
    routes receive a fully-populated CurrentUser or raise 401.

    In prod mode (DEMO_ALLOW_PUBLIC_THUMBS=0), the binary endpoints are NOT
    auth-free: the global dep runs the bearer check here, but a valid signed
    URL (?sig=&exp=) lets `signed_or_token` short-circuit per-route. To make
    the bearer check here not 401 spuriously when only a signed URL is
    presented, we treat the binary endpoints as still "soft auth-free" at this
    layer — the per-route dependency enforces either signed URL or bearer.
    """
    if _is_auth_free(request.url.path):
        return None
    # In prod mode the binary endpoints also need a guard: either a valid
    # signed URL OR a bearer token. Short-circuit here when a valid signed URL
    # is presented; otherwise fall through to the bearer-token check.
    if not allow_public_thumbs() and any(
        request.url.path.endswith(s) for s in _AUTH_FREE_BINARY_SUFFIXES
    ):
        sig = request.query_params.get("sig")
        exp_raw = request.query_params.get("exp")
        exp = int(exp_raw) if exp_raw and exp_raw.isdigit() else None
        if sig and verify_signed_token(request.url.path, sig, exp):
            return None
    token = _extract_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    row = store.lookup_token(token)
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    if row["disabled"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="user disabled")
    if row["expires_at"] < __import__("time").time():
        store.revoke_token(token)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token expired")
    if row["kind"] != "access":
        # Refresh tokens are not accepted on regular API routes — they only
        # work at /api/auth/refresh.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="access token required")
    user = CurrentUser(
        id=row["user_id"],
        username=row["username"],
        display_name=row["display_name"] or row["username"],
        role=row["role"],
    )
    request.state.user = user
    return user


def require_role(min_role: str):
    """Build a dependency that enforces ``min_role`` (read|write|admin)."""
    if min_role not in ROLE_RANK:
        raise ValueError(f"unknown role: {min_role}")

    def _dep(user: CurrentUser | None = Depends(require_user)) -> CurrentUser:
        if user is None:
            # Public endpoint — should never reach here (only auth-free
            # paths skip the check), but keep the error clean if a future
            # refactor moves things around.
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
        if not user.has_role(min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"requires role >= {min_role}",
            )
        return user

    _dep.__name__ = f"require_role_{min_role}"
    return _dep


# --- login / refresh / logout (called from api.py) ------------------------

def login(username: str, password: str, *, user_agent: str | None = None,
           ip: str | None = None) -> tuple[str, str, CurrentUser]:
    """Verify credentials, mint access + refresh tokens, return (access, refresh, user).

    On bad credentials: raises 401 (caller decides the response shape).
    """
    row = store.get_user_by_username(username)
    if row is None or row["disabled"] or not verify_password(password, row["password_hash"]):
        # Constant-ish: always hash the dummy to keep wall time comparable when
        # the username doesn't exist (defense against username enumeration).
        if row is None:
            bcrypt.checkpw(b"probe", bcrypt.hashpw(b"probe", bcrypt.gensalt(rounds=4)))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    access = store.issue_token(row["id"], "access", access_ttl(),
                                user_agent=user_agent, ip=ip)
    refresh = store.issue_token(row["id"], "refresh", refresh_ttl(),
                                 user_agent=user_agent, ip=ip)
    store.touch_last_login(row["id"])
    user = CurrentUser(id=row["id"], username=row["username"],
                       display_name=row["display_name"] or row["username"],
                       role=row["role"])
    return access, refresh, user


def refresh(refresh_token: str, *, user_agent: str | None = None,
            ip: str | None = None) -> tuple[str, str, CurrentUser]:
    """Mint a new (access, refresh) pair, **rotating** the refresh token.

    Security rationale (P-02 from the live pen test, 2026-09-01):
    the previous implementation let the same refresh token mint new
    access tokens indefinitely until expiry. That meant a stolen
    refresh token worked for the full 30-day TTL. With rotation, every
    successful /api/auth/refresh call revokes the old refresh token and
    mints a new one. If a stolen refresh token is used by the attacker,
    the legitimate user's next refresh call sees "invalid refresh token"
    and is forced to re-login — surfacing the theft.
    """
    row = store.lookup_token(refresh_token)
    if row is None or row["kind"] != "refresh" or row["disabled"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid refresh token")
    if row["expires_at"] < __import__("time").time():
        store.revoke_token(refresh_token)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="refresh expired")
    # ROTATE: revoke the old refresh token before minting a new pair.
    store.revoke_token(refresh_token)
    access = store.issue_token(row["user_id"], "access", access_ttl(),
                                user_agent=user_agent, ip=ip)
    new_refresh = store.issue_token(row["user_id"], "refresh", refresh_ttl(),
                                     user_agent=user_agent, ip=ip)
    user = CurrentUser(id=row["user_id"], username=row["username"],
                       display_name=row["display_name"] or row["username"],
                       role=row["role"])
    return access, new_refresh, user

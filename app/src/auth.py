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

import logging
import os
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
    for suffix in _AUTH_FREE_BINARY_SUFFIXES:
        if (
            len(path) > len(suffix)
            and path.endswith(suffix)
            and path[-len(suffix) - 1] == "/"
        ):
            return True
    return False


def require_user(request: Request) -> CurrentUser | None:
    """Resolve the bearer token to a CurrentUser; raise 401 otherwise.

    Auth-free paths (`/api/auth/*`, `/api/health`, `/api/status`, plus the
    binary `/thumb`, `/full`, `/cover`, `/crop` tail endpoints) short-circuit
    and return None — those routes don't need a user. All other routes
    receive a fully-populated CurrentUser or raise 401.
    """
    if _is_auth_free(request.url.path):
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
            ip: str | None = None) -> tuple[str, CurrentUser]:
    row = store.lookup_token(refresh_token)
    if row is None or row["kind"] != "refresh" or row["disabled"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid refresh token")
    if row["expires_at"] < __import__("time").time():
        store.revoke_token(refresh_token)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="refresh expired")
    access = store.issue_token(row["user_id"], "access", access_ttl(),
                                user_agent=user_agent, ip=ip)
    user = CurrentUser(id=row["user_id"], username=row["username"],
                       display_name=row["display_name"] or row["username"],
                       role=row["role"])
    return access, user

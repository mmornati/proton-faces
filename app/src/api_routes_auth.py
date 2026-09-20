"""Auth + sign + 2FA route handlers.

Handlers are defined as module-level functions so they can be re-exported
from ``api.py`` (tests call them as ``api.api_login(...)`` etc.). The
``router`` object registers them on the FastAPI app via ``include_router``
in ``api.py``. Public routes set ``dependencies=[]`` explicitly so they
stay public even if route ordering changes (the global ``require_user``
dep on the FastAPI app would otherwise block them).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

import api  # noqa: E402  (intentional: see api_common docstring)
import store
from auth import (
    MAX_PASSWORD_BYTES,
    CurrentUser,
    decrypt_totp_secret,
    encrypt_totp_secret,
    generate_totp_secret,
    hash_password,
    login,
    make_signed_token,
    match_totp_code,
    password_too_long,
    require_user,
    totp_uri,
    verify_2fa,
    verify_password,
)
from auth import access_ttl as auth_access_ttl
from auth import refresh as refresh_tokens

log = logging.getLogger("api")

router = APIRouter()

# One HMAC per path; a grid page signs at most a few hundred URLs. The cap
# keeps /api/sign from being a CPU amplifier for an authenticated caller.
SIGN_MAX_PATHS = 1000


@router.post("/api/auth/login", dependencies=[])
def api_login(request: Request, body: dict = Body(...)):
    """Exchange username+password for an access+refresh token pair."""
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if not username or not password:
        raise HTTPException(400, "username and password required")
    ua = request.headers.get("user-agent")
    ip = request.client.host if request.client else None
    access, refresh, user, needs_2fa, pending = login(username, password, user_agent=ua, ip=ip)
    if needs_2fa:
        return {
            "2fa_required": True,
            "pending_token": pending,
            "user": {
                "id": user.id,
                "username": user.username,
                "display_name": user.display_name,
                "role": user.role,
            },
        }
    resp = JSONResponse({
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_in": auth_access_ttl(),
        "user": {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
        },
    })
    api._set_refresh_cookie(resp, refresh)
    return resp


@router.post("/api/auth/2fa/verify", dependencies=[])
def api_2fa_verify(request: Request, body: dict = Body(...)):
    """Second step of a 2FA login: redeem a pending token + 6-digit code."""
    pending = (body.get("pending_token") or "").strip()
    code = (body.get("code") or "").strip()
    if not pending or not code:
        raise HTTPException(400, "pending_token and code required")
    ua = request.headers.get("user-agent")
    ip = request.client.host if request.client else None
    access, refresh, user = verify_2fa(pending, code, user_agent=ua, ip=ip)
    resp = JSONResponse({
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_in": auth_access_ttl(),
        "user": {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
        },
    })
    api._set_refresh_cookie(resp, refresh)
    return resp


@router.post("/api/auth/2fa/setup")
def api_2fa_setup(request: Request, user: CurrentUser = Depends(require_user)):
    """Start 2FA enrollment: generate a fresh TOTP secret."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    secret_b32 = generate_totp_secret()
    store.set_totp_secret(user.id, encrypt_totp_secret(secret_b32))
    store.set_totp_enabled(user.id, False)
    return {
        "secret": secret_b32,
        "otpauth_uri": totp_uri(secret_b32, user.username),
        "issuer": "proton-faces",
    }


@router.post("/api/auth/2fa/confirm")
def api_2fa_confirm(request: Request, body: dict = Body(...),
                    user: CurrentUser = Depends(require_user)):
    """Activate 2FA after the user confirms a valid code from their app."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    code = (body.get("code") or "").strip()
    if not code:
        raise HTTPException(400, "code required")
    secret_enc = store.get_totp_secret(user.id)
    if not secret_enc:
        raise HTTPException(400, "no 2FA setup in progress — call /api/auth/2fa/setup first")
    try:
        secret_b32 = decrypt_totp_secret(secret_enc)
    except Exception:
        raise HTTPException(400, "invalid 2FA setup state")
    counter = match_totp_code(secret_b32, code, last_counter=store.get_totp_last_counter(user.id))
    if counter is None:
        raise HTTPException(400, "invalid code — check the time on your device and try again")
    store.set_totp_last_counter(user.id, counter)
    store.set_totp_enabled(user.id, True)
    return {"ok": True, "totp_enabled": True}


@router.post("/api/auth/2fa/disable")
def api_2fa_disable(request: Request, body: dict = Body(default={}),
                    user: CurrentUser = Depends(require_user)):
    """Turn off 2FA for the current user."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    code = (body.get("code") or "").strip()
    password = body.get("password") or ""
    secret_enc = store.get_totp_secret(user.id)
    if not secret_enc:
        raise HTTPException(400, "2FA is not enabled")
    if code:
        try:
            secret_b32 = decrypt_totp_secret(secret_enc)
        except Exception:
            raise HTTPException(400, "invalid 2FA state")
        counter = match_totp_code(secret_b32, code, last_counter=store.get_totp_last_counter(user.id))
        if counter is None:
            raise HTTPException(400, "invalid code")
    elif password:
        row = store.get_user_by_username(user.username)
        if row is None or not verify_password(password, row["password_hash"]):
            raise HTTPException(400, "invalid password")
    else:
        raise HTTPException(400, "code or password required to disable 2FA")
    store.set_totp_secret(user.id, None)
    store.set_totp_enabled(user.id, False)
    store.revoke_pending_2fa_for_user(user.id)
    return {"ok": True, "totp_enabled": False}


@router.post("/api/auth/refresh", dependencies=[])
def api_refresh(request: Request, body: dict = Body(default={})):
    """Issue a new (access, refresh) pair. The refresh token is rotated."""
    rt = (request.cookies.get(api._REFRESH_COOKIE) or "").strip()
    if not rt:
        rt = (body.get("refresh_token") or "").strip()
    if not rt:
        raise HTTPException(400, "refresh_token required")
    ua = request.headers.get("user-agent")
    ip = request.client.host if request.client else None
    access, new_refresh, user = refresh_tokens(rt, user_agent=ua, ip=ip)
    resp = JSONResponse({
        "access_token": access,
        "refresh_token": new_refresh,
        "token_type": "Bearer",
        "expires_in": auth_access_ttl(),
        "user": {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
        },
    })
    api._set_refresh_cookie(resp, new_refresh)
    return resp


@router.post("/api/auth/logout")
def api_logout(request: Request, user: CurrentUser = Depends(require_user)):
    """Invalidate the bearer token used for this request."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.split(None, 1)[1].strip() if auth_header.lower().startswith("bearer ") else None
    if token:
        store.revoke_token(token)
    rt = (request.cookies.get(api._REFRESH_COOKIE) or "").strip()
    if rt:
        store.revoke_token(rt)
    resp = JSONResponse({"ok": True})
    api._clear_refresh_cookie(resp)
    return resp


@router.get("/api/auth/me")
def api_me(user: CurrentUser = Depends(require_user)):
    """Return the current user's profile."""
    row = store.get_user_by_id(user.id)
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "totp_enabled": bool(row and row["totp_enabled"]),
    }


@router.post("/api/auth/password")
def api_change_password(request: Request, body: dict = Body(...),
                        user: CurrentUser = Depends(require_user)):
    """Self-service password change."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    current = body.get("current_password")
    new = body.get("new_password")
    if not isinstance(current, str) or not isinstance(new, str):
        raise HTTPException(400, "current_password and new_password are required")
    row = store.get_user_by_username(user.username)
    if row is None or not verify_password(current, row["password_hash"]):
        raise HTTPException(401, "current password is incorrect")
    if len(new) < 8:
        raise HTTPException(400, "new password must be at least 8 characters")
    if password_too_long(new):
        raise HTTPException(400, f"new password must be at most {MAX_PASSWORD_BYTES} bytes")
    store.update_user(user.id, password_hash=hash_password(new))
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.split(None, 1)[1].strip() if auth_header.lower().startswith("bearer ") else None
    revoked = store.revoke_all_tokens_except(user.id, token) if token else store.revoke_all_tokens(user.id)
    return {"ok": True, "revoked_sessions": revoked}


@router.get("/api/auth/limits", dependencies=[])
def api_limits():
    """Public — UI uses this to render the login screen with the right labels."""
    return {"min_username": 2, "min_password": 8, "max_password": MAX_PASSWORD_BYTES}


@router.post("/api/sign")
def api_sign(request: Request,
              body: dict = Body(...),
              user: CurrentUser = Depends(require_user)):
    """Issue short-lived signed URLs for binary endpoints."""
    paths = body.get("paths")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(400, "paths must be a non-empty list")
    if len(paths) > SIGN_MAX_PATHS:
        raise HTTPException(400, f"too many paths (max {SIGN_MAX_PATHS})")
    ttl = 300
    if isinstance(body.get("ttl"), int):
        ttl = max(30, min(3600, body["ttl"]))
    out = []
    for p in paths:
        if not isinstance(p, str) or not p.startswith("/api/"):
            raise HTTPException(400, f"invalid path: {p!r}")
        if not any(p.endswith(s) for s in ("/thumb", "/full", "/cover", "/crop")):
            raise HTTPException(400, f"path not signable: {p!r}")
        sig, exp = make_signed_token(p, ttl_seconds=ttl)
        out.append({"path": p, "sig": sig, "exp": exp})
    return {"urls": out, "ttl": ttl}

"""Admin route handlers: users, backups, sync, schedule, checks, bridge cache.

Handlers are defined as module-level functions so they can be re-exported
from ``api.py`` (tests call them as ``api.api_admin_users(...)`` etc.).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, HTTPException

import admin
import api  # noqa: E402  (intentional: see api_common docstring)
from auth import (
    MAX_PASSWORD_BYTES,
    ROLE_RANK,
    CurrentUser,
    demo_disable_backups,
    hash_password,
    password_too_long,
    require_role,
)
from store import (
    create_user,
    delete_empty_people,
    delete_user,
    get_user_by_id,
    get_user_by_username,
    list_users,
    revoke_all_tokens,
    revoke_pending_2fa_for_user,
    set_totp_enabled,
    set_totp_secret,
    update_user,
)

log = logging.getLogger("api")

router = APIRouter()


@router.get("/api/admin/users")
def api_admin_users(_: CurrentUser = Depends(require_role("admin"))):
    """List all users (admin only)."""
    if api.demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    return {"users": [api._user_row_public(r) for r in list_users()]}


@router.post("/api/admin/users")
def api_admin_create_user(body: dict = Body(...),
                           _: CurrentUser = Depends(require_role("admin"))):
    """Create a new user (admin only)."""
    if api.demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    username = (body.get("username") or "").strip()
    display_name = (body.get("display_name") or username).strip() or username
    password = body.get("password") or ""
    role = (body.get("role") or "read").strip().lower()
    if len(username) < 2:
        raise HTTPException(400, "username must be at least 2 characters")
    if len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    if password_too_long(password):
        raise HTTPException(400, f"password must be at most {MAX_PASSWORD_BYTES} bytes")
    if role not in ROLE_RANK:
        raise HTTPException(400, f"role must be one of {sorted(ROLE_RANK)}")
    if get_user_by_username(username) is not None:
        raise HTTPException(409, "username already exists")
    user_id = create_user(username=username, password_hash=hash_password(password),
                           role=role, display_name=display_name)
    row = get_user_by_id(user_id)
    return {"user": api._user_row_public(row)}


@router.patch("/api/admin/users/{user_id}")
def api_admin_patch_user(user_id: int, body: dict = Body(...),
                          actor: CurrentUser = Depends(require_role("admin"))):
    """Update a user (admin only)."""
    if api.demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    if get_user_by_id(user_id) is None:
        raise HTTPException(404, "user not found")
    display_name = body.get("display_name")
    role = body.get("role")
    disabled = body.get("disabled")
    password = body.get("password")
    if role is not None and role not in ROLE_RANK:
        raise HTTPException(400, f"role must be one of {sorted(ROLE_RANK)}")
    if password is not None and len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    if password is not None and password_too_long(password):
        raise HTTPException(400, f"password must be at most {MAX_PASSWORD_BYTES} bytes")
    password_hash = hash_password(password) if password else None
    # Never demote or disable the last active admin: there would be no
    # in-app way back (only the --create-admin shell path).
    if (role is not None and role != "admin") or disabled:
        target = get_user_by_id(user_id)
        if target["role"] == "admin" and not target["disabled"]:
            others = [u for u in list_users() if u["role"] == "admin" and not u["disabled"] and u["id"] != user_id]
            if not others:
                raise HTTPException(400, "cannot demote or disable the last admin")
    try:
        update_user(user_id, display_name=display_name, role=role,
                    disabled=disabled, password_hash=password_hash)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    revoked = 0
    if password and not (user_id == actor.id):
        revoked = revoke_all_tokens(user_id)
    row = get_user_by_id(user_id)
    return {"user": api._user_row_public(row), "tokens_revoked": revoked}


@router.delete("/api/admin/users/{user_id}")
def api_admin_delete_user(user_id: int,
                           actor: CurrentUser = Depends(require_role("admin"))):
    """Remove a user. The last remaining admin cannot delete themselves."""
    if api.demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    row = get_user_by_id(user_id)
    if row is None:
        raise HTTPException(404, "user not found")
    if row["id"] == actor.id:
        admins = [u for u in list_users() if u["role"] == "admin" and not u["disabled"] and u["id"] != actor.id]
        if not admins:
            raise HTTPException(400, "cannot delete the last admin")
    delete_user(user_id)
    return {"ok": True}


@router.post("/api/admin/users/{user_id}/logout")
def api_admin_force_logout(user_id: int,
                            _: CurrentUser = Depends(require_role("admin"))):
    """Sign a user out of every device."""
    if api.demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    n = revoke_all_tokens(user_id)
    return {"ok": True, "revoked": n}


@router.post("/api/admin/users/{user_id}/2fa/disable")
def api_admin_disable_2fa(user_id: int,
                           _: CurrentUser = Depends(require_role("admin"))):
    """Force-disable 2FA for a user (e.g. lost authenticator device)."""
    if api.demo_disable_admin_user_management():
        raise HTTPException(404, "not found")
    if get_user_by_id(user_id) is None:
        raise HTTPException(404, "user not found")
    set_totp_secret(user_id, None)
    set_totp_enabled(user_id, False)
    revoke_pending_2fa_for_user(user_id)
    return {"ok": True, "totp_enabled": False}


@router.get("/api/admin/overview")
def api_admin_overview(_: CurrentUser = Depends(require_role("admin"))):
    """Server overview (admin only)."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    return admin.overview()


@router.post("/api/admin/backup")
def api_admin_backup(_: CurrentUser = Depends(require_role("admin"))):
    """Trigger a manual snapshot now."""
    if demo_disable_backups():
        raise HTTPException(404, "not found")
    try:
        res = admin.snapshot_backup()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    return {"name": res["name"], "size": res["size"], "ts": res["ts"]}


@router.get("/api/admin/backups")
def api_admin_backups(_: CurrentUser = Depends(require_role("admin"))):
    """List available backups."""
    if demo_disable_backups():
        raise HTTPException(404, "not found")
    return admin.list_backups()


@router.delete("/api/admin/backups/{name}")
def api_admin_delete_backup(name: str, _: CurrentUser = Depends(require_role("admin"))):
    """Delete a backup by name."""
    if demo_disable_backups():
        raise HTTPException(404, "not found")
    try:
        return admin.delete_backup(name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))


@router.post("/api/admin/db/compact")
def api_admin_db_compact(_: CurrentUser = Depends(require_role("admin"))):
    """VACUUM the index to reclaim free pages. Run while the indexer is idle."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    import sqlite3

    from store import compact_database
    try:
        return {"ok": True, **compact_database()}
    except sqlite3.OperationalError as exc:
        raise HTTPException(409, f"database busy, retry when the indexer is idle: {exc}")


@router.post("/api/admin/backups/prune")
def api_admin_prune_backups(body: dict = Body(default={}),
                             _: CurrentUser = Depends(require_role("admin"))):
    """Prune old backups."""
    if demo_disable_backups():
        raise HTTPException(404, "not found")
    try:
        res = admin.prune_backups(body.get("keep"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return {"ok": res.get("ok", True),
            "removed": res.get("removed", []),
            "removed_count": len(res.get("removed", [])),
            "kept": res.get("kept", 0)}


@router.get("/api/admin/schedule")
def api_admin_get_schedule(_: CurrentUser = Depends(require_role("admin"))):
    """Get the current sync schedule."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    return admin.get_schedule()


@router.put("/api/admin/schedule")
def api_admin_set_schedule(body: dict = Body(...),
                            _: CurrentUser = Depends(require_role("admin"))):
    """Set the sync schedule."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    try:
        return admin.set_schedule(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/api/admin/sync")
def api_admin_get_sync(_: CurrentUser = Depends(require_role("admin"))):
    """Admin view of the indexer's sync state + live sync configuration."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    status = api._indexer_proxy_json("GET", "/status")
    cfg = api._indexer_proxy_json("GET", "/sync-config")
    return {
        "last_sync": status.get("last_sync"),
        "last_sync_error": status.get("last_sync_error"),
        "threads": status.get("threads", {}),
        "tip_interval": api.settings.sync_interval,
        "config": cfg,
    }


@router.post("/api/admin/sync/trigger")
def api_admin_sync_trigger(_: CurrentUser = Depends(require_role("admin"))):
    """Ask the indexer to run a full scan on its next sync-loop iteration."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    return api._indexer_proxy_json("POST", "/trigger-sync")


@router.put("/api/admin/sync")
def api_admin_set_sync(body: dict = Body(...),
                        _: CurrentUser = Depends(require_role("admin"))):
    """Persist the live sync configuration on the indexer."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    return api._indexer_proxy_json("PUT", "/sync-config", body)


@router.post("/api/admin/checks")
def api_admin_checks(_: CurrentUser = Depends(require_role("admin"))):
    """Run health checks."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    return admin.run_checks(recent_full_res_failures=api._recent_full_res_failures())


@router.post("/api/admin/people/gc-empty")
def api_admin_gc_empty_people(_: CurrentUser = Depends(require_role("admin"))):
    """Delete anonymous people rows with no faces / no photos (ghost rows)."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    deleted = delete_empty_people()
    api._invalidate_people_cache()
    return {"deleted": deleted, "ok": True}


@router.post("/api/admin/people/prune-small")
def api_admin_prune_small_people(body: dict = Body(default={}),
                                 _: CurrentUser = Depends(require_role("admin"))):
    """Retroactively apply MIN_CLUSTER_SIZE: drop anonymous clusters below it.

    Body: ``{"min_photos": N, "dry_run": bool}``. ``min_photos`` defaults to
    the configured MIN_CLUSTER_SIZE. Faces go back to the unassigned pool
    and are regrouped by the next cluster run; named people are untouched.
    ``dry_run`` only reports how many rows would go.
    """
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    from store import delete_people_bulk, small_unnamed_people_ids
    try:
        min_photos = int(body.get("min_photos", api.settings.min_cluster_size))
    except (TypeError, ValueError):
        raise HTTPException(400, "min_photos must be an integer")
    min_photos = max(1, min(min_photos, 1000))
    ids = small_unnamed_people_ids(min_photos)
    if body.get("dry_run"):
        return {"ok": True, "dry_run": True, "min_photos": min_photos, "candidate_count": len(ids)}
    api._drop_people_crops(ids)
    deleted = delete_people_bulk(ids)
    api._invalidate_dups_cache()
    api._invalidate_people_cache()
    return {"ok": True, "min_photos": min_photos, "deleted": deleted}


@router.get("/api/admin/bridge/cache")
def api_admin_bridge_cache(_: CurrentUser = Depends(require_role("admin"))):
    """Get the bridge SDK cache status."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    try:
        return api.get_bridge().cache_status()
    except Exception as exc:
        raise HTTPException(502, f"bridge cache lookup failed: {exc}")


@router.post("/api/admin/bridge/cache/clear")
def api_admin_bridge_cache_clear(_: CurrentUser = Depends(require_role("admin"))):
    """Tell the bridge to clear its SDK cache and restart itself."""
    if api.demo_disable_admin_area():
        raise HTTPException(404, "not found")
    try:
        return api.get_bridge().clear_cache()
    except Exception as exc:
        raise HTTPException(502, f"bridge cache clear failed: {exc}")

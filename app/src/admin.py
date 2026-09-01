"""Server-ops helpers for the admin area: backup, disk checks, scheduled runs.

Everything here is gated behind ``require_role("admin")`` in api.py — this
module only implements the operations. There is intentionally no extra auth
token: the app's existing role-based account system is the gate.

Backups use ``VACUUM INTO`` for a consistent, non-blocking snapshot of the
live WAL-mode SQLite index (thumbnails / CLIP cache are re-derivable, so
only ``index.sqlite3`` is archived). Scheduled auto-backups run in a daemon
thread inside the ``app`` container, with the schedule persisted in
``DATA_DIR/admin_config.json`` so the admin UI can edit it without touching
the host.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from config import settings

log = logging.getLogger("admin")


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in (
        "1", "true", "yes", "on",
    )


_SCHEDULE_FILE = "admin_config.json"

# Defaults when admin_config.json is missing or incomplete.
_DEFAULT_SCHEDULE = {
    "enabled": False,
    "hour": 3,
    "minute": 0,
    "keep": 10,
    "last_backup_at": None,
}

# Fail the disk check when free space drops below this fraction of total.
_FREE_FRAC_THRESHOLD = 0.10


def _schedule_path() -> Path:
    return settings.data_dir / _SCHEDULE_FILE


# --- backups --------------------------------------------------------------

def snapshot_backup() -> dict:
    """VACUUM INTO a consistent snapshot of the index DB; return {name, size, ts}."""
    settings.backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = settings.backup_dir / f"index-{stamp}.sqlite3"
    if not settings.db_path.exists():
        raise FileNotFoundError("index DB not found yet")
    # VACUUM INTO produces a consistent snapshot from a WAL-mode db without
    # blocking writers; the source connection is read-only.
    src = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    try:
        src.execute(f'VACUUM INTO "{dest}"')
    finally:
        src.close()
    sz = dest.stat().st_size
    try:
        with _schedule_path().open("r") as f:
            sched = json.load(f)
    except (OSError, json.JSONDecodeError):
        sched = {}
    sched["last_backup_at"] = time.time()
    _schedule_path().write_text(json.dumps(sched, indent=2))
    return {"name": dest.name, "size": sz, "ts": dest.stat().st_mtime}


def _backup_rows() -> list[dict]:
    """Sorted newest-first metadata for every index-*.sqlite3 backup."""
    if not settings.backup_dir.exists():
        return []
    rows = []
    for p in settings.backup_dir.iterdir():
        if not p.is_file() or not p.name.startswith("index-") or not p.name.endswith(".sqlite3"):
            continue
        try:
            rows.append({"name": p.name, "size": p.stat().st_size,
                         "mtime": p.stat().st_mtime, "ts": p.stat().st_mtime})
        except OSError:
            continue
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def list_backups() -> list[dict]:
    """Return a list of backups [{name, size_bytes, ts}, ...] newest first."""
    return [{"name": r["name"], "size_bytes": r["size"], "ts": r["ts"]}
            for r in _backup_rows()]


def delete_backup(name: str) -> dict:
    """Remove one backup; guard against path traversal."""
    if not name or Path(name).name != name:
        raise ValueError("invalid backup name")
    target = settings.backup_dir / name
    if not target.is_file():
        raise FileNotFoundError("backup not found")
    target.unlink()
    return {"ok": True, "name": name}


def prune_backups(keep: int | None = None) -> dict:
    """Delete oldest backups beyond the retention limit."""
    schedule = get_schedule()
    if keep is None:
        keep = int(schedule.get("keep", _DEFAULT_SCHEDULE["keep"]))
    keep = max(1, int(keep))
    rows = _backup_rows()
    removed = []
    for r in rows[keep:]:
        try:
            delete_backup(r["name"])
            removed.append(r["name"])
        except (OSError, ValueError, FileNotFoundError):
            log.warning("failed to prune backup %s", r["name"])
    return {"ok": True, "removed": removed, "kept": min(keep, len(rows))}


# --- schedule -------------------------------------------------------------

def get_schedule() -> dict:
    """Read admin_config.json, merge with defaults, fold in last-backup freshness."""
    try:
        raw = json.loads(_schedule_path().read_text())
    except (OSError, json.JSONDecodeError):
        raw = {}
    sched = dict(_DEFAULT_SCHEDULE)
    sched.update({k: v for k, v in raw.items() if k in _DEFAULT_SCHEDULE})
    # If last_backup_at isn't persisted, fall back to the filesystem newest.
    if not sched.get("last_backup_at"):
        rows = _backup_rows()
        sched["last_backup_at"] = rows[0]["ts"] if rows else None
    return sched


def set_schedule(body: dict) -> dict:
    """Validate and persist admin_config.json."""
    sched = get_schedule()
    sched["enabled"] = bool(body.get("enabled", sched["enabled"]))
    try:
        h = int(body.get("hour", sched["hour"]))
        sched["hour"] = max(0, min(23, h))
    except (TypeError, ValueError):
        pass
    try:
        m = int(body.get("minute", sched["minute"]))
        sched["minute"] = max(0, min(59, m))
    except (TypeError, ValueError):
        pass
    try:
        sched["keep"] = max(1, min(365, int(body.get("keep", sched["keep"]))))
    except (TypeError, ValueError):
        pass
    # Preserve the last_backup_at field when re-saving.
    out = {k: sched[k] for k in _DEFAULT_SCHEDULE}
    _schedule_path().write_text(json.dumps(out, indent=2))
    return out


def _now_hm() -> tuple[int, int]:
    t = time.localtime()
    return t.tm_hour, t.tm_min


def _backup_due(sched: dict) -> bool:
    if not sched.get("enabled"):
        return False
    hh, mm = _now_hm()
    if hh != int(sched.get("hour", 0)) or mm != int(sched.get("minute", 0)):
        return False
    last = sched.get("last_backup_at")
    if not last:
        return True
    try:
        last_d = datetime.fromtimestamp(float(last), tz=timezone.utc).date()
        return last_d != datetime.now(tz=timezone.utc).date()
    except Exception:
        return True


def _worker_loop() -> None:
    while True:
        try:
            sched = get_schedule()
            if _backup_due(sched):
                log.info("scheduled backup running (hh:mm=%s:%s)",
                         sched.get("hour"), sched.get("minute"))
                res = snapshot_backup()
                log.info("backup done: %s (%d bytes)", res["name"], res["size"])
                try:
                    pr = prune_backups(sched.get("keep", _DEFAULT_SCHEDULE["keep"]))
                    log.info("prune: kept=%d removed=%d", pr["kept"], len(pr["removed"]))
                except Exception:
                    log.exception("backup prune failed")
        except Exception:
            log.exception("scheduled backup failed")
        # Sleep in small steps so a shutdown is prompt-ish.
        for _ in range(6):
            time.sleep(10)


def start_backup_worker() -> threading.Thread:
    """Start the daemon thread that enforces the auto-backup schedule."""
    t = threading.Thread(target=_worker_loop, name="backup-worker", daemon=True)
    t.start()
    log.info("backup worker started (schedule=%s)", get_schedule())
    return t


# --- checks ---------------------------------------------------------------

def _db_integrity() -> dict:
    try:
        conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            quick = conn.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            conn.close()
        ok = integrity == "ok" and quick == "ok"
        return {"name": "Database integrity", "ok": ok,
                "status": "ok" if ok else "fail",
                "detail": "integrity_check/quick_check pass" if ok else f"integrity={integrity!r} quick={quick!r}"}
    except Exception as exc:
        return {"name": "Database integrity", "ok": False, "status": "fail", "detail": str(exc)}


def _disk_space() -> dict:
    try:
        total, used, free = shutil.disk_usage(settings.data_dir)
        frac = free / total
        ok = frac >= _FREE_FRAC_THRESHOLD
        detail = (f"{free/1e9:.1f} GB free of {total/1e9:.1f} GB "
                  f"({frac*100:.0f}%)")
        if not ok:
            detail += " — below 10% threshold"
        return {"name": "Disk space", "ok": ok, "status": "ok" if ok else "low",
                "detail": detail}
    except Exception as exc:
        return {"name": "Disk space", "ok": False, "status": "fail", "detail": str(exc)}


def _backup_freshness() -> dict:
    rows = _backup_rows()
    if not rows:
        return {"name": "Backup freshness", "ok": False, "status": "stale",
                "detail": "no backups yet"}
    age_h = (time.time() - rows[0]["ts"]) / 3600
    ok = age_h < 48
    status = "ok" if ok else "stale"
    return {"name": "Backup freshness", "ok": ok, "status": status,
            "detail": f"latest backup {age_h:.1f} h ago"}


def _backup_writable() -> dict:
    try:
        settings.backup_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.backup_dir / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        return {"name": "Backup dir writable", "ok": True, "status": "ok",
                "detail": str(settings.backup_dir)}
    except Exception as exc:
        return {"name": "Backup dir writable", "ok": False, "status": "fail",
                "detail": str(exc)}


def _data_writable() -> dict:
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.data_dir / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        return {"name": "Data dir writable", "ok": True, "status": "ok",
                "detail": str(settings.data_dir)}
    except Exception as exc:
        return {"name": "Data dir writable", "ok": False, "status": "fail",
                "detail": str(exc)}


def _indexer_liveness() -> dict:
    try:
        from indexer import get_indexer_state  # type: ignore
    except Exception as exc:
        return {"name": "Indexer", "ok": False, "status": "fail",
                "detail": f"import error: {exc}"}
    try:
        state = get_indexer_state() or {}
    except Exception as exc:
        return {"name": "Indexer", "ok": False, "status": "fail",
                "detail": str(exc)}
    if state.get("remote"):
        return {"name": "Indexer", "ok": True, "status": "remote",
                "detail": "running in dedicated indexer container"}
    err = state.get("last_sync_error")
    started = state.get("started_at")
    if not started:
        return {"name": "Indexer", "ok": False, "status": "idle",
                "detail": "no sync has run yet"}
    if err:
        return {"name": "Indexer", "ok": False, "status": "error",
                "detail": f"last sync error: {err}"}
    last = state.get("last_sync") or 0
    age = time.time() - last if last else None
    detail = f"last sync {age:.0f}s ago" if age is not None else "never synced"
    ok = age is not None and age < 3600
    return {"name": "Indexer", "ok": ok, "status": "ok" if ok else "stale",
            "detail": detail}


def _bridge_reachability() -> dict:
    try:
        from bridge_client import BridgeClient  # type: ignore
        bc = BridgeClient(settings.bridge_url)
        h = bc.health()
        ok = bool(h and h.get("ok"))
        detail = "bridge healthy" if ok else f"bridge response: {h}"
        return {"name": "Bridge reachable", "ok": ok,
                "status": "ok" if ok else "down", "detail": detail}
    except Exception as exc:
        return {"name": "Bridge reachable", "ok": False, "status": "down",
                "detail": str(exc)}


def _bridge_cache_health(recent_full_res_failures: int = 0) -> dict:
    """Inspect the on-disk SDK cache via the bridge's GET /cache endpoint.

    Flags `stale` when:
      - the bridge is reachable,
      - the newest cache file is older than `settings.bridge_cache_stale_sec`,
      - AND the `/api/photos/{uid}/full` endpoint has been failing recently
        (the getFileDownloader-waitForCondition2 hang signature).

    Cache age alone is NOT actionable — an idle bridge may not touch its
    caches for hours. Combining with the full-res failure count avoids
    false positives while still catching the real "stale cache after a
    Proton incident" condition.

    On any bridge/cache lookup error we return `ok=False` with the error
    in `detail` so the admin UI surfaces the failure rather than silently
    skipping the check.
    """
    from bridge_client import BridgeClient  # type: ignore
    name = "Bridge cache"
    try:
        bc = BridgeClient(settings.bridge_url)
        status = bc.cache_status()
    except Exception as exc:
        return {"name": name, "ok": False, "status": "down",
                "detail": f"cache lookup failed: {exc}"}

    files = status.get("files") or []
    if not files:
        return {"name": name, "ok": True, "status": "empty",
                "detail": "no SDK cache files reported by bridge"}

    try:
        now = time.time()
        newest_mtime = max(f.get("mtime") or 0 for f in files)
        age = int(now - newest_mtime)
        ages_str = ", ".join(
            f"{f['name']}={int(now - (f.get('mtime') or 0))}s"
            for f in files
        )
    except Exception as exc:
        return {"name": name, "ok": False, "status": "error",
                "detail": f"malformed cache report: {exc}"}

    stale = age > settings.bridge_cache_stale_sec and recent_full_res_failures > 0
    detail = f"newest {age}s old; recent /full failures={recent_full_res_failures}; {ages_str}"
    return {
        "name": name,
        "ok": not stale,
        "status": "stale" if stale else "ok",
        "detail": detail,
    }


def run_checks(recent_full_res_failures: int = 0) -> dict:
    """Run every server check; return the report.

    `recent_full_res_failures` is supplied by the API layer (count of /full
    proxy failures in the last 15 minutes). It feeds the bridge-cache
    staleness check so an idle bridge isn't flagged just because its
    caches happen to be old.
    """
    checks = [
        _db_integrity(),
        _disk_space(),
        _backup_freshness(),
        _backup_writable(),
        _data_writable(),
        _indexer_liveness(),
        _bridge_reachability(),
        _bridge_cache_health(recent_full_res_failures=recent_full_res_failures),
    ]
    passed = sum(1 for c in checks if c["ok"])
    return {"checks": checks, "passed": passed, "total": len(checks), "ts": time.time()}


def overview() -> dict:
    """Aggregated admin landing view: server, disk, last backup, schedule, count."""
    sched = get_schedule()
    rows = _backup_rows()
    disk: dict = {}
    try:
        total, used, free = shutil.disk_usage(settings.data_dir)
        # P-05: when EXPOSE_OPERATIONAL_DETAILS=0 (the safe default in
        # public deployments), redact the exact byte counts and the
        # on-host data path. We keep the free_frac ratio because admins
        # need it to know if the disk is full.
        if _env_bool("EXPOSE_OPERATIONAL_DETAILS", False):
            disk = {
                "path": str(settings.data_dir),
                "total": total,
                "used": used,
                "free": free,
                "free_frac": free / total if total else 0,
                "data_bytes": _dir_size_bytes(settings.data_dir),
            }
        else:
            disk = {"path": None, "free_frac": free / total if total else 0,
                    "data_bytes": _dir_size_bytes(settings.data_dir)}
    except OSError:
        disk = {"path": None, "free_frac": 0, "data_bytes": 0}
    # P-05: redact hostname + full platform string when not in
    # operational-details mode. Admins still see Python version + uptime.
    if _env_bool("EXPOSE_OPERATIONAL_DETAILS", False):
        server = {
            "hostname": platform.node(),
            "app_version": os.environ.get("APP_VERSION", "dev"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "uptime_sec": _uptime_seconds(),
        }
    else:
        server = {
            "hostname": None,
            "app_version": os.environ.get("APP_VERSION", "dev"),
            "python": platform.python_version(),
            "platform": None,
            "uptime_sec": _uptime_seconds(),
        }
    backup = {
        "last_name": rows[0]["name"] if rows else None,
        "last_ts": rows[0]["ts"] if rows else None,
        "last_size": rows[0]["size"] if rows else 0,
        "count": len(rows),
        "total_size": sum(r["size"] for r in rows),
    }
    return {
        "server": server,
        "disk": disk,
        "backup": backup,
        "schedule": sched,
        "backups": list_backups(),
    }


_BOOT_TIME = time.time()


def _uptime_seconds() -> float:
    return max(0.0, time.time() - _BOOT_TIME)


def _dir_size_bytes(path: Path) -> int:
    """Best-effort recursive directory size in bytes."""
    if not path.exists():
        return 0
    total = 0
    try:
        for entry in path.iterdir():
            try:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += _dir_size_bytes(entry)
            except OSError:
                continue
    except OSError:
        return total
    return total

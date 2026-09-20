
import pytest

import admin
import disk_usage


class TestBackups:
    def test_snapshot_backup_creates_file(self, tmp_db):
        out = admin.snapshot_backup()
        assert out["name"].startswith("index-")
        assert out["name"].endswith(".sqlite3")
        assert out["size"] > 0
        assert (admin.settings.backup_dir / out["name"]).exists()

    def test_snapshot_requires_db(self, app_settings):
        with pytest.raises(FileNotFoundError):
            admin.snapshot_backup()

    def test_list_and_delete_backup(self, tmp_db):
        admin.snapshot_backup()
        backups = admin.list_backups()
        assert len(backups) == 1
        assert backups[0]["name"].startswith("index-")
        assert backups[0]["size_bytes"] > 0
        assert admin.delete_backup(backups[0]["name"]) == {"ok": True, "name": backups[0]["name"]}
        assert admin.list_backups() == []

    def test_delete_backup_rejects_traversal(self, tmp_db):
        with pytest.raises(ValueError):
            admin.delete_backup("../evil.sqlite3")

    def test_delete_backup_missing(self, tmp_db):
        with pytest.raises(FileNotFoundError):
            admin.delete_backup("index-00000000-000000.sqlite3")

    def test_prune_backups_keeps_newest(self, tmp_db, monkeypatch):
        stamps = iter(["20240101-000001", "20240101-000002", "20240101-000003",
                       "20240101-000004", "20240101-000005"])
        monkeypatch.setattr(admin.time, "strftime", lambda fmt, *a: next(stamps))
        for _ in range(5):
            admin.snapshot_backup()
        out = admin.prune_backups(keep=2)
        assert len(out["removed"]) == 3
        assert len(admin.list_backups()) == 2

    def test_prune_keep_defaults_to_schedule(self, tmp_db, monkeypatch):
        stamps = iter(["20240101-000001", "20240101-000002", "20240101-000003"])
        monkeypatch.setattr(admin.time, "strftime", lambda fmt, *a: next(stamps))
        for _ in range(3):
            admin.snapshot_backup()
        admin.set_schedule({"keep": 1})
        out = admin.prune_backups()
        assert len(out["removed"]) == 2


class TestSchedule:
    def test_default_schedule(self, app_settings):
        sched = admin.get_schedule()
        assert sched["enabled"] is False
        assert sched["keep"] == 10
        assert sched["last_backup_at"] is None

    def test_set_schedule_persists_and_merges(self, app_settings):
        admin.set_schedule({"enabled": True, "hour": 4, "minute": 30})
        sched = admin.get_schedule()
        assert sched["enabled"] is True
        assert sched["hour"] == 4
        assert sched["minute"] == 30
        assert sched["keep"] == 10  # default retained

    def test_set_schedule_rejects_invalid_values(self, app_settings):
        for body in ({"hour": 99}, {"hour": -1}, {"minute": 60}, {"minute": -5},
                     {"keep": 0}, {"keep": 9999}, {"hour": "3"}, {"minute": 3.5},
                     {"enabled": "false"}):
            with pytest.raises(ValueError):
                admin.set_schedule(body)
        # nothing was written by the failed calls
        assert admin.get_schedule()["hour"] == 3
        assert admin.get_schedule()["keep"] == 10

    def test_set_schedule_rejects_lists_all_bad_fields(self, app_settings):
        with pytest.raises(ValueError, match="hour, minute, keep"):
            admin.set_schedule({"hour": 99, "minute": -1, "keep": 0})

    def test_set_schedule_partial_update_merges(self, app_settings):
        admin.set_schedule({"enabled": True, "hour": 4, "minute": 30})
        sched = admin.get_schedule()
        assert sched["enabled"] is True
        assert sched["hour"] == 4
        assert sched["minute"] == 30
        assert sched["keep"] == 10  # default retained

    def test_last_backup_at_falls_back_to_filesystem(self, tmp_db):
        admin.snapshot_backup()
        sched = admin.get_schedule()
        assert sched["last_backup_at"] is not None


def _utc(y, mo, d, h, mi, s=0):
    from datetime import datetime, timezone
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)


class TestBackupDue:
    def _sched(self, **kw):
        s = {"enabled": True, "hour": 3, "minute": 0, "keep": 10, "last_backup_at": None}
        s.update(kw)
        return s

    def test_disabled_never_due(self):
        assert admin._backup_due(self._sched(enabled=False), _utc(2024, 1, 1, 3, 0)) is False

    def test_before_slot_not_due(self):
        assert admin._backup_due(self._sched(), _utc(2024, 1, 1, 2, 59)) is False

    def test_at_slot_without_last_backup_due(self):
        assert admin._backup_due(self._sched(), _utc(2024, 1, 1, 3, 0)) is True

    def test_after_slot_without_last_backup_due(self):
        # worker was down at 03:00, wakes at 03:05 -> catch-up fires once
        assert admin._backup_due(self._sched(), _utc(2024, 1, 1, 3, 5)) is True

    def test_after_slot_with_last_backup_before_slot_due(self):
        # last backup yesterday 02:00 -> today's 03:00 slot is still owed
        last = _utc(2024, 1, 1, 2, 0).timestamp()
        assert admin._backup_due(self._sched(last_backup_at=last), _utc(2024, 1, 1, 3, 5)) is True

    def test_after_slot_with_last_backup_at_slot_not_due(self):
        # already ran today's slot -> no double fire
        last = _utc(2024, 1, 1, 3, 0).timestamp()
        assert admin._backup_due(self._sched(last_backup_at=last), _utc(2024, 1, 1, 3, 5)) is False

    def test_after_slot_with_last_backup_yesterday_not_due(self):
        # ran yesterday's slot -> today's slot is owed (catch-up across days)
        last = _utc(2024, 1, 1, 3, 0).timestamp()
        assert admin._backup_due(self._sched(last_backup_at=last), _utc(2024, 1, 2, 3, 5)) is True

    def test_multi_day_gap_fires_once(self):
        # last backup 3 days ago -> still exactly one fire owed
        last = _utc(2024, 1, 1, 3, 0).timestamp()
        assert admin._backup_due(self._sched(last_backup_at=last), _utc(2024, 1, 4, 3, 5)) is True

    def test_corrupt_last_backup_treated_as_due(self):
        assert admin._backup_due(self._sched(last_backup_at="garbage"), _utc(2024, 1, 1, 3, 5)) is True


def _ok_check(name="x"):
    return {"name": name, "ok": True, "status": "ok", "detail": ""}


class TestOverview:
    def test_overview_uses_shared_cache(self, tmp_db, monkeypatch):
        """Issue #103: overview() must reuse the shared 1 h cache instead of
        walking the whole data dir on every admin page load."""
        disk_usage._dirsize_cache = {}
        top_walks: list[None] = []
        real_walk = disk_usage.dir_size_bytes

        def spy_walk(path):
            if path == admin.settings.data_dir:
                top_walks.append(None)
            return real_walk(path)

        monkeypatch.setattr(disk_usage, "dir_size_bytes", spy_walk)
        out = admin.overview()
        assert "disk" in out
        assert "data_bytes" in out["disk"]
        # exactly one top-level walk for the data dir (cached on the second call)
        assert len(top_walks) == 1
        admin.overview()
        assert len(top_walks) == 1

    def test_overview_redacted_hides_path(self, tmp_db, monkeypatch):
        monkeypatch.delenv("EXPOSE_OPERATIONAL_DETAILS", raising=False)
        out = admin.overview()
        assert out["disk"]["path"] is None
        assert out["server"]["hostname"] is None


class TestRunChecks:
    def test_run_checks_shape(self, tmp_db, monkeypatch):
        admin.snapshot_backup()
        monkeypatch.setattr(admin, "_indexer_liveness", _ok_check)
        monkeypatch.setattr(admin, "_bridge_reachability", _ok_check)
        monkeypatch.setattr(
            admin, "_bridge_cache_health",
            lambda recent_full_res_failures=0: _ok_check(),
        )
        out = admin.run_checks()
        assert out["total"] == 8
        assert out["passed"] == out["total"]
        assert len(out["checks"]) == 8

    def test_run_checks_reports_failures(self, tmp_db, monkeypatch):
        admin.snapshot_backup()
        monkeypatch.setattr(
            admin, "_indexer_liveness",
            lambda: {"name": "x", "ok": False, "status": "down", "detail": "boom"},
        )
        monkeypatch.setattr(admin, "_bridge_reachability", _ok_check)
        monkeypatch.setattr(
            admin, "_bridge_cache_health",
            lambda recent_full_res_failures=0: _ok_check(),
        )
        out = admin.run_checks()
        assert out["passed"] == out["total"] - 1

    def test_bridge_cache_health_flags_stale(self, app_settings, monkeypatch):
        class _FakeBC:
            def cache_status(self):
                return {"ok": True, "files": [{"name": "cache-x.sqlite", "size": 1, "mtime": 0}]}

        import bridge_client

        monkeypatch.setattr(bridge_client, "BridgeClient", lambda *a, **k: _FakeBC())
        stale = admin._bridge_cache_health(recent_full_res_failures=3)
        assert stale["ok"] is False
        assert stale["status"] == "stale"
        fresh = admin._bridge_cache_health(recent_full_res_failures=0)
        assert fresh["ok"] is True

    def test_bridge_reachability_down(self, app_settings, monkeypatch):
        class _FakeBC:
            def health(self):
                return {"ok": False, "loggedIn": False}

        import bridge_client

        monkeypatch.setattr(bridge_client, "BridgeClient", lambda *a, **k: _FakeBC())
        out = admin._bridge_reachability()
        assert out["ok"] is False
        assert out["status"] == "down"

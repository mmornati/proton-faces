import os
import time

import config
import full_cache
from conftest import APP_SRC  # noqa: F401  (ensures sys.path wiring)


def _put(name: str, size: int, age: float):
    p = config.settings.full_cache_dir / name
    p.write_bytes(b"x" * size)
    t = time.time() - age
    os.utime(p, (t, t))
    return p


class TestLookup:
    def test_miss_and_hit_refresh_lru(self, app_settings):
        assert full_cache.lookup("p1") is None
        p = _put("p1.png", 10, age=1000)
        path, ct = full_cache.lookup("p1")
        assert path == p and ct == "image/png"
        assert time.time() - p.stat().st_mtime < 5

    def test_disabled(self, app_settings, monkeypatch):
        _put("p1.jpg", 10, age=0)
        monkeypatch.setattr(config.settings, "full_cache_max_mb", 0)
        assert full_cache.lookup("p1") is None
        assert full_cache.writer("p1", "image/jpeg") is None


class TestWriter:
    def test_commit_publishes_atomically(self, app_settings):
        w = full_cache.writer("p1", "image/webp")
        w.write(b"abc")
        assert not (config.settings.full_cache_dir / "p1.webp").exists()
        assert w.commit(3)
        assert (config.settings.full_cache_dir / "p1.webp").read_bytes() == b"abc"
        assert [p.name for p in config.settings.full_cache_dir.iterdir()] == ["p1.webp"]

    def test_non_image_type_not_cached(self, app_settings):
        assert full_cache.writer("p1", "application/octet-stream") is None
        assert full_cache.writer("p1", "video/mp4") is None

    def test_oversized_entry_dropped(self, app_settings, monkeypatch):
        monkeypatch.setattr(full_cache, "MAX_ENTRY_BYTES", 4)
        w = full_cache.writer("p1", "image/jpeg")
        w.write(b"12345")
        assert not w.commit(None)
        assert list(config.settings.full_cache_dir.iterdir()) == []

    def test_abort_removes_temp(self, app_settings):
        w = full_cache.writer("p1", "image/jpeg")
        w.write(b"12")
        w.abort()
        assert list(config.settings.full_cache_dir.iterdir()) == []


class TestEviction:
    def test_evicts_oldest_until_under_target(self, app_settings, monkeypatch):
        monkeypatch.setattr(config.settings, "full_cache_max_mb", 1)
        mb = 1024 * 1024
        old = _put("old.jpg", mb // 2, age=300)
        mid = _put("mid.jpg", mb // 2, age=200)
        new = _put("new.jpg", mb // 4, age=100)
        assert full_cache.evict_if_needed(force=True) == 1
        assert not old.exists() and mid.exists() and new.exists()

    def test_under_cap_keeps_everything(self, app_settings):
        p = _put("a.jpg", 100, age=10)
        assert full_cache.evict_if_needed(force=True) == 0
        assert p.exists()

    def test_throttled(self, app_settings, monkeypatch):
        monkeypatch.setattr(config.settings, "full_cache_max_mb", 1)
        full_cache.evict_if_needed(force=True)
        _put("big.jpg", 2 * 1024 * 1024, age=10)
        assert full_cache.evict_if_needed() == 0

    def test_stale_part_files_swept(self, app_settings):
        stale = _put(".p1.abc.part", 10, age=7200)
        live = _put(".p2.def.part", 10, age=5)
        full_cache.evict_if_needed(force=True)
        assert not stale.exists() and live.exists()


def test_remove(app_settings):
    a = _put("p1.jpg", 1, age=0)
    b = _put("p2.heic", 1, age=0)
    full_cache.remove(["p1", "p2", "p3"])
    assert not a.exists() and not b.exists()

import pytest

import demo
import store


@pytest.fixture
def demo_bridge(app_settings):
    return demo.DemoBridge()


class TestDemoBridge:
    def test_health(self, demo_bridge):
        h = demo_bridge.health()
        assert h == {"ok": True, "loggedIn": True, "demo": True}

    def test_timeline_sorted_desc(self, demo_bridge):
        items = demo_bridge.timeline()
        times = [i["captureTime"] for i in items]
        assert times == sorted(times, reverse=True)

    def test_timeline_ids_shape(self, demo_bridge):
        items = demo_bridge.timeline_ids()
        assert all({"uid", "captureTime"} <= set(i) for i in items)

    def test_nodes_returns_fixture(self, demo_bridge):
        item = demo_bridge.timeline()[0]
        nodes = demo_bridge.nodes([item["uid"]])
        assert nodes and nodes[0]["uid"] == item["uid"]

    def test_albums_shape(self, demo_bridge):
        out = demo_bridge.albums()
        assert "albums" in out

    def test_thumbnails_writes_work_file(self, demo_bridge, app_settings):
        item = demo_bridge.timeline()[0]
        out = demo_bridge.thumbnails([item["uid"]])
        assert out["results"][0]["uid"] == item["uid"]
        assert out["results"][0]["ok"] is True
        assert (app_settings.work_dir / f"{item['uid']}.webp").exists()

    def test_full_photo_returns_bytes(self, demo_bridge):
        item = demo_bridge.timeline()[0]
        resp = demo_bridge.full_photo(item["uid"])
        assert resp.status_code == 200
        assert len(b"".join(resp.iter_bytes())) > 0

    def test_full_photo_unknown_uid(self, demo_bridge):
        resp = demo_bridge.full_photo("missing-uid")
        assert resp.status_code == 200
        assert b"".join(resp.iter_bytes()) == b""

    def test_cache_status_and_clear(self, demo_bridge):
        assert demo_bridge.cache_status()["demo"] is True
        assert demo_bridge.clear_cache()["demo"] is True

    def test_close_is_noop(self, demo_bridge):
        demo_bridge.close()  # must not raise


class TestEnsureDefaultAdmin:
    def test_creates_demo_admin(self, tmp_db, monkeypatch):
        monkeypatch.delenv("DEMO_ADMIN_PASSWORD", raising=False)
        demo.ensure_default_admin()
        user = store.get_user_by_username("demo")
        assert user is not None
        assert user["role"] == "admin"

    def test_idempotent(self, tmp_db, monkeypatch):
        monkeypatch.delenv("DEMO_ADMIN_PASSWORD", raising=False)
        demo.ensure_default_admin()
        demo.ensure_default_admin()
        assert len(store.list_users()) == 1

    def test_uses_env_password(self, tmp_db, monkeypatch):
        monkeypatch.setenv("DEMO_ADMIN_PASSWORD", "custom-secret")
        demo.ensure_default_admin()
        import auth

        user = store.get_user_by_username("demo")
        assert auth.verify_password("custom-secret", user["password_hash"])


class TestApplyDemoGps:
    def test_applies_gps_once(self, tmp_db):
        fixture = demo._load_fixture()
        gps_nodes = [n for n in fixture["timeline"] if n.get("gps")]
        assert gps_nodes, "fixture should contain GPS-tagged photos"
        for n in gps_nodes:
            store.upsert_photos(
                [{"uid": n["uid"], "name": n["name"], "media_type": n["mediaType"],
                  "capture_time": 1}]
            )
        assert demo.apply_demo_gps() == len(gps_nodes)
        # Second run is a no-op (already set)
        assert demo.apply_demo_gps() == 0
        row = store.get_photo(gps_nodes[0]["uid"])
        assert row["gps_lat"] is not None and row["gps_lng"] is not None

from fastapi.testclient import TestClient

import indexer
import indexer_status
import store


class TestHealthz:
    def test_ok(self):
        client = TestClient(indexer_status.app)
        assert client.get("/healthz").json() == {"ok": True}


class TestStatus:
    def test_status_shape(self, tmp_db, monkeypatch):
        monkeypatch.setattr(
            indexer, "get_indexer_state", lambda: {"remote": True, "running": False}
        )
        client = TestClient(indexer_status.app)
        out = client.get("/status").json()
        assert out["remote"] is True
        assert "pending_db" in out
        assert out["pending_db"] == 0

    def test_pending_db_counts(self, tmp_db, monkeypatch):
        monkeypatch.setattr(
            indexer, "get_indexer_state", lambda: {"remote": True, "running": False}
        )
        store.upsert_photos([{"uid": "a1", "name": "a1", "media_type": "image/jpeg", "capture_time": 1}])
        store.upsert_photos([{"uid": "a2", "name": "a2", "media_type": "image/jpeg", "capture_time": 1}])
        client = TestClient(indexer_status.app)
        assert client.get("/status").json()["pending_db"] == 2


class TestTriggerSync:
    def test_triggers_full_sync(self, tmp_db, monkeypatch):
        calls = []
        monkeypatch.setattr(indexer, "request_full_sync", lambda: calls.append(1))
        client = TestClient(indexer_status.app)
        assert client.post("/trigger-sync").status_code == 200
        assert len(calls) == 1


class TestSyncConfig:
    def test_get_config(self, tmp_db):
        client = TestClient(indexer_status.app)
        out = client.get("/sync-config").json()
        assert out["enabled"] is True
        assert out["tip_size"] == 10

    def test_update_config(self, tmp_db):
        client = TestClient(indexer_status.app)
        resp = client.put("/sync-config", json={"tip_size": 25})
        assert resp.status_code == 200
        assert resp.json()["tip_size"] == 25

    def test_invalid_config_returns_400(self, tmp_db):
        client = TestClient(indexer_status.app)
        resp = client.put("/sync-config", json={"tip_size": 0})
        assert resp.status_code == 400

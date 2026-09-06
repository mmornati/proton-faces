
import reclaim_deleted
import store


class TestRowFromNode:
    def test_basic_mapping(self):
        node = {
            "uid": "abc",
            "name": "IMG_1.jpg",
            "mediaType": "image/jpeg",
            "captureTime": "2024-01-15T10:30:00Z",
            "sha1": "deadbeef",
            "size": 12345,
            "albums": ["album1"],
        }
        row = reclaim_deleted._row_from_node(node)
        assert row["uid"] == "abc"
        assert row["media_type"] == "image/jpeg"
        assert row["capture_time"] == 1705314600
        assert row["sha1"] == "deadbeef"
        assert row["size"] == 12345
        assert row["albums"] == ["album1"]

    def test_missing_optional_fields(self):
        node = {"uid": "x", "name": "x", "mediaType": "image/png"}
        row = reclaim_deleted._row_from_node(node)
        assert row["capture_time"] == 0
        assert row["albums"] == []
        assert row["size"] is None

    def test_invalid_capture_time(self):
        node = {"uid": "x", "name": "x", "mediaType": "image/png", "captureTime": "not-a-date"}
        assert reclaim_deleted._row_from_node(node)["capture_time"] == 0


class TestMain:
    def test_reclaims_deleted_photos(self, tmp_db, monkeypatch):
        store.upsert_photos(
            [{"uid": "d1", "name": "d1", "media_type": "image/jpeg", "capture_time": 1}]
        )
        store.mark_deleted(["d1"])
        assert store.get_photo("d1")["status"] == "deleted"

        def fake_nodes(uids):
            return [
                {
                    "uid": "d1",
                    "name": "d1.jpg",
                    "mediaType": "image/jpeg",
                    "captureTime": "2024-01-15T10:30:00Z",
                    "sha1": "abc",
                }
            ]

        monkeypatch.setattr(reclaim_deleted, "_nodes", fake_nodes)
        assert reclaim_deleted.main() == 0
        assert store.get_photo("d1")["status"] == "new"
        assert store.get_photo("d1")["name"] == "d1.jpg"

    def test_no_deleted_photos(self, tmp_db):
        assert reclaim_deleted.main() == 0

    def test_empty_batch_skips_nodes(self, tmp_db, monkeypatch):
        store.upsert_photos(
            [{"uid": "d1", "name": "d1", "media_type": "image/jpeg", "capture_time": 1}]
        )
        store.mark_deleted(["d1"])
        calls = []

        def fake_nodes(uids):
            calls.append(uids)
            return []

        monkeypatch.setattr(reclaim_deleted, "_nodes", fake_nodes)
        reclaim_deleted.main()
        assert len(calls) == 1
        assert calls[0] == ["d1"]

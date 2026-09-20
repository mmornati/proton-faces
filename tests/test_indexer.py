import json
import queue
import time

import numpy as np
import pytest
from PIL import Image

import indexer
import store


class TestEpoch:
    def test_none(self):
        assert indexer._epoch(None) is None

    def test_int_float(self):
        assert indexer._epoch(123) == 123
        assert indexer._epoch(123.9) == 123

    def test_iso_z(self):
        assert indexer._epoch("2024-01-15T10:30:00Z") == 1705314600

    def test_iso_offset(self):
        assert indexer._epoch("2024-01-15T11:30:00+01:00") == 1705314600

    def test_bad_string(self):
        assert indexer._epoch("garbage") is None


class TestRowsFromItems:
    def test_basic(self):
        items = [
            {"uid": "a", "name": "a.jpg", "mediaType": "image/jpeg",
             "captureTime": "2024-01-15T10:30:00Z", "sha1": "s", "albums": ["x"]},
            {"uid": "b", "name": "b.jpg", "mediaType": "image/png",
             "captureTime": 5},
        ]
        rows = indexer._rows_from_items(items)
        assert len(rows) == 2
        assert rows[0]["uid"] == "a"
        assert rows[0]["capture_time"] == 1705314600
        assert rows[1]["capture_time"] == 5

    def test_skips_missing(self):
        rows = indexer._rows_from_items(
            [{"uid": "a", "mediaType": "image/jpeg", "missing": True}]
        )
        assert rows == []

    def test_defaults(self):
        rows = indexer._rows_from_items([{"uid": "a", "mediaType": "image/jpeg"}])
        assert rows[0]["name"] is None
        assert rows[0]["capture_time"] is None
        assert rows[0]["albums"] == []


class TestNormBbox:
    def test_normalizes(self):
        assert indexer._norm_bbox([0, 0, 100, 100], 200, 100) == [0.0, 0.0, 0.5, 1.0]


class TestIsImageVideo:
    def test_image(self, tmp_db):
        store.upsert_photos([{"uid": "p1", "name": "p1", "media_type": "image/jpeg", "capture_time": 1}])
        assert indexer._is_image("p1") is True
        assert indexer._is_video("p1") is False

    def test_video(self, tmp_db):
        store.upsert_photos([{"uid": "v1", "name": "v1", "media_type": "video/mp4", "capture_time": 1}])
        assert indexer._is_video("v1") is True
        assert indexer._is_image("v1") is False

    def test_unknown(self, tmp_db):
        assert indexer._is_image("nope") is False
        assert indexer._is_video("nope") is False


class TestResizeToThumb:
    def test_writes_webp(self, tmp_path):
        src = tmp_path / "in.jpg"
        Image.fromarray(np.full((100, 80, 3), 128, dtype=np.uint8)).save(src, "JPEG")
        dest = tmp_path / "out.webp"
        indexer._resize_to_thumb(src, dest)
        assert dest.exists()
        with Image.open(dest) as img:
            assert img.format == "WEBP"
            assert img.size[0] <= 512 and img.size[1] <= 512

    def test_downscales_large(self, tmp_path):
        src = tmp_path / "in.jpg"
        Image.fromarray(np.full((2000, 1000, 3), 128, dtype=np.uint8)).save(src, "JPEG")
        dest = tmp_path / "out.webp"
        indexer._resize_to_thumb(src, dest)
        with Image.open(dest) as img:
            assert img.size[0] <= 512 and img.size[1] <= 512

    def test_uses_configured_webp_method(self, tmp_path, monkeypatch):
        src = tmp_path / "in.jpg"
        Image.fromarray(np.full((100, 80, 3), 128, dtype=np.uint8)).save(src, "JPEG")
        dest = tmp_path / "out.webp"
        monkeypatch.setattr(indexer.settings, "webp_method", 2)
        seen = {}
        original_save = Image.Image.save

        def fake_save(self, *args, **kwargs):
            seen.update(kwargs)
            return original_save(self, *args, **kwargs)

        monkeypatch.setattr(Image.Image, "save", fake_save)
        indexer._resize_to_thumb(src, dest)
        assert seen.get("method") == 2


def _jpg_with_gps(path, lat=39.564, lng=2.619):
    """Write a JPEG carrying EXIF GPS coordinates (Mallorca-style)."""
    from fractions import Fraction

    from PIL import ExifTags

    def _deg(coord):
        d = int(coord)
        m = int((coord - d) * 60)
        s = (coord - d - m / 60) * 3600
        return (Fraction(d, 1), Fraction(m, 1), Fraction(round(s * 100), 100))

    img = Image.fromarray(np.full((100, 80, 3), 128, dtype=np.uint8))
    exif = Image.Exif()
    gps = {
        1: "N" if lat >= 0 else "S",
        2: _deg(abs(lat)),
        3: "E" if lng >= 0 else "W",
        4: _deg(abs(lng)),
    }
    exif[ExifTags.IFD.GPSInfo] = gps
    img.save(path, "JPEG", exif=exif)


class TestExtractExifGps:
    def test_jpg_with_gps(self, tmp_path):
        src = tmp_path / "gps.jpg"
        _jpg_with_gps(src)
        lat, lng = indexer._extract_exif_gps(src)
        assert lat == pytest.approx(39.564, abs=1e-3)
        assert lng == pytest.approx(2.619, abs=1e-3)

    def test_jpg_without_gps(self, tmp_path):
        src = tmp_path / "plain.jpg"
        Image.fromarray(np.full((100, 80, 3), 128, dtype=np.uint8)).save(src, "JPEG")
        assert indexer._extract_exif_gps(src) is None

    def test_southern_hemisphere_negative(self, tmp_path):
        src = tmp_path / "south.jpg"
        _jpg_with_gps(src, lat=-33.8688, lng=151.2093)
        lat, lng = indexer._extract_exif_gps(src)
        assert lat == pytest.approx(-33.8688, abs=1e-3)
        assert lng == pytest.approx(151.2093, abs=1e-3)

    def test_missing_file(self, tmp_path):
        assert indexer._extract_exif_gps(tmp_path / "nope.jpg") is None


class TestSyncConfig:
    def test_defaults(self, app_settings):
        cfg = indexer.get_sync_config()
        assert cfg["enabled"] is True
        assert cfg["tip_size"] == 10
        assert cfg["last_full_scan"] is None

    def test_set_and_persist(self, app_settings):
        cfg = indexer.set_sync_config({"tip_size": 25, "enabled": False})
        assert cfg["tip_size"] == 25
        assert cfg["enabled"] is False
        saved = json.loads((app_settings.data_dir / "sync_config.json").read_text())
        assert saved["tip_size"] == 25

    def test_unknown_key_raises(self, app_settings):
        with pytest.raises(ValueError):
            indexer.set_sync_config({"nope": 1})

    def test_invalid_values_raise(self, app_settings):
        with pytest.raises(ValueError):
            indexer.set_sync_config({"tip_size": 0})
        with pytest.raises(ValueError):
            indexer.set_sync_config({"full_scan_interval": -1})
        with pytest.raises(ValueError):
            indexer.set_sync_config({"deletion_threshold": 0})
        with pytest.raises(ValueError):
            indexer.set_sync_config({"deletion_threshold": 1.5})

    def test_last_full_scan_accepts_float_or_none(self, app_settings):
        cfg = indexer.set_sync_config({"last_full_scan": 123.0})
        assert cfg["last_full_scan"] == 123.0
        cfg = indexer.set_sync_config({"last_full_scan": None})
        assert cfg["last_full_scan"] is None


class TestRequestFullSync:
    def test_sets_event(self):
        indexer._sync_wakeup.clear()
        indexer.request_full_sync()
        assert indexer._sync_wakeup.is_set()


class TestGetIndexerState:
    def test_remote_stub_when_not_started(self, monkeypatch):
        monkeypatch.setitem(indexer._runtime, "threads", {})
        state = indexer.get_indexer_state()
        assert state["remote"] is True


class TestProcessOne:
    def _seed_work_photo(self, uid="w1"):
        store.upsert_photos(
            [{"uid": uid, "name": uid, "media_type": "image/jpeg", "capture_time": 1}]
        )
        assert store.claim_photo_for_download(uid) is True
        work = indexer._work_path(uid)
        work.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((50, 40, 3), 128, dtype=np.uint8)).save(work, "WEBP")
        return uid

    def test_process_one_happy_path(self, tmp_db, app_settings, monkeypatch):
        uid = self._seed_work_photo()
        monkeypatch.setattr(indexer, "detect_faces", lambda bgr: [])
        monkeypatch.setattr(indexer, "embed_pil", lambda img: np.ones(512, dtype=np.float32))
        indexer._process_one(uid)
        row = store.get_photo(uid)
        assert row["status"] == "done"
        assert row["thumb_path"] == f"{uid}.webp"
        assert store.clip_exists(uid) is True
        assert not indexer._work_path(uid).exists()

    def test_process_one_inserts_faces(self, tmp_db, app_settings, monkeypatch):
        uid = self._seed_work_photo()
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)

        def fake_detect(bgr):
            return [{"bbox": [0, 0, 20, 20], "confidence": 0.95, "embedding": emb}]

        monkeypatch.setattr(indexer, "detect_faces", fake_detect)
        monkeypatch.setattr(indexer, "embed_pil", lambda img: np.ones(512, dtype=np.float32))
        indexer._process_one(uid)
        assert store.count_faces_for_photo(uid) == 1
        row = store.get_photo(uid)
        assert row["status"] == "done"

    def test_process_one_preserves_existing_faces(self, tmp_db, app_settings, monkeypatch):
        uid = self._seed_work_photo()
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        store.insert_face(uid, None, 0.9, "[0,0,10,10]", emb.tobytes())

        def boom(*a, **k):
            raise AssertionError("detect must be skipped when faces exist")

        monkeypatch.setattr(indexer, "detect_faces", boom)
        monkeypatch.setattr(indexer, "embed_pil", lambda img: np.ones(512, dtype=np.float32))
        indexer._process_one(uid)
        assert store.count_faces_for_photo(uid) == 1
        assert store.get_photo(uid)["status"] == "done"

    def test_process_one_missing_work_file(self, tmp_db, app_settings, monkeypatch):
        store.upsert_photos(
            [{"uid": "nofile", "name": "nofile", "media_type": "image/jpeg", "capture_time": 1}]
        )
        assert store.claim_photo_for_download("nofile") is True
        indexer._process_one("nofile")
        row = store.get_photo("nofile")
        assert row["status"] == "error"
        assert row["error"] == "work file missing"

    def test_process_one_single_commit(self, tmp_db, app_settings, monkeypatch):
        # Regression test for #94: the whole per-photo sequence (claim, read,
        # face inserts, clip insert, done) must be exactly ONE commit.
        class CountingConn:
            def __init__(self, real):
                self._real = real
                self._n_commits = 0

            def __getattr__(self, name):
                return getattr(self._real, name)

            def commit(self, *a, **k):
                result = self._real.commit(*a, **k)
                self._n_commits += 1
                return result

        uid = self._seed_work_photo()
        monkeypatch.setattr(indexer, "detect_faces", lambda bgr: [])
        monkeypatch.setattr(indexer, "embed_pil", lambda img: np.ones(512, dtype=np.float32))
        orig_get_conn = store._get_persistent_conn
        holder: dict = {}

        def counting_conn(db_path, timeout=30):
            if not holder:
                holder["conn"] = CountingConn(orig_get_conn(db_path, timeout))
            return holder["conn"]

        # The person-means warm-up runs before the transaction (its own
        # connection use); prime it so only the per-photo sequence is counted.
        import cluster
        cluster._person_means_cached()
        monkeypatch.setattr(store, "_get_persistent_conn", counting_conn)
        indexer._process_one(uid)
        assert holder["conn"]._n_commits == 1
        assert store.get_photo(uid)["status"] == "done"


class _FakePending:
    """Minimal stand-in for indexer._pending supporting the batching reads."""

    def __init__(self, items=None, raise_get=False):
        self._items = list(items or [])
        self.raise_get = raise_get

    def qsize(self):
        return len(self._items)

    def get_nowait(self):
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)

    def get(self, timeout=None):
        if self.raise_get or not self._items:
            raise queue.Empty
        return self._items.pop(0)


class TestNextWorkBatch:
    def test_deep_queue_pulls_up_to_batch_size(self, app_settings, monkeypatch):
        monkeypatch.setattr(indexer.settings, "clip_batch_size", 4)
        monkeypatch.setattr(indexer.settings, "clip_batch_queue_depth", 8)
        monkeypatch.setattr(indexer, "_pending", _FakePending(items=list("abcdefghij")))
        assert indexer._next_work_batch() == ["a", "b", "c", "d"]

    def test_deep_queue_returns_what_is_available(self, app_settings, monkeypatch):
        monkeypatch.setattr(indexer.settings, "clip_batch_size", 8)
        monkeypatch.setattr(indexer.settings, "clip_batch_queue_depth", 1)
        monkeypatch.setattr(indexer, "_pending", _FakePending(items=["a", "b"]))
        assert indexer._next_work_batch() == ["a", "b"]

    def test_shallow_queue_stays_single(self, app_settings, monkeypatch):
        monkeypatch.setattr(indexer.settings, "clip_batch_size", 4)
        monkeypatch.setattr(indexer.settings, "clip_batch_queue_depth", 8)
        monkeypatch.setattr(indexer, "_pending", _FakePending(items=["a"]))
        assert indexer._next_work_batch() == ["a"]

    def test_batch_disabled_always_single(self, app_settings, monkeypatch):
        monkeypatch.setattr(indexer.settings, "clip_batch_size", 1)
        monkeypatch.setattr(indexer.settings, "clip_batch_queue_depth", 1)
        monkeypatch.setattr(indexer, "_pending", _FakePending(items=["a", "b", "c"]))
        assert indexer._next_work_batch() == ["a"]

    def test_empty_queue_falls_back_to_downloading(self, app_settings, monkeypatch):
        monkeypatch.setattr(indexer.settings, "clip_batch_size", 1)
        monkeypatch.setattr(indexer.settings, "clip_batch_queue_depth", 1)
        monkeypatch.setattr(indexer, "_pending", _FakePending(raise_get=True))
        monkeypatch.setattr(
            indexer, "get_photos",
            lambda status, limit=500, offset=0: [{"uid": "leftover"}],
        )
        assert indexer._next_work_batch() == ["leftover"]

    def test_empty_everywhere_returns_empty(self, app_settings, monkeypatch):
        monkeypatch.setattr(indexer.settings, "clip_batch_size", 1)
        monkeypatch.setattr(indexer.settings, "clip_batch_queue_depth", 1)
        monkeypatch.setattr(indexer, "_pending", _FakePending(raise_get=True))
        monkeypatch.setattr(indexer, "get_photos", lambda status, limit=500, offset=0: [])
        assert indexer._next_work_batch() == []


class TestProcessBatch:
    def _seed_work_photo(self, uid="w1"):
        store.upsert_photos(
            [{"uid": uid, "name": uid, "media_type": "image/jpeg", "capture_time": 1}]
        )
        assert store.claim_photo_for_download(uid) is True
        work = indexer._work_path(uid)
        work.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((50, 40, 3), 128, dtype=np.uint8)).save(work, "WEBP")
        return uid

    def test_batch_embeds_once_and_marks_done(self, tmp_db, app_settings, monkeypatch):
        a = self._seed_work_photo("b1")
        b = self._seed_work_photo("b2")

        calls = []

        def fake_embed_batch(images):
            calls.append(len(images))
            return np.stack(
                [np.ones(512, dtype=np.float32) * (i + 1) for i in range(len(images))]
            )

        monkeypatch.setattr(indexer, "detect_faces", lambda bgr: [])
        monkeypatch.setattr(indexer, "embed_batch", fake_embed_batch)
        indexer._process_batch([a, b])

        assert calls == [2]
        assert store.get_photo(a)["status"] == "done"
        assert store.get_photo(b)["status"] == "done"
        assert store.clip_exists(a) is True
        assert store.clip_exists(b) is True

    def test_existing_clip_skips_batch_embedding(self, tmp_db, app_settings, monkeypatch):
        a = self._seed_work_photo("r1")
        b = self._seed_work_photo("r2")
        keep = np.full(512, 0.25, dtype=np.float32)
        store.insert_clip(a, keep.tobytes())

        calls = []

        def fake_embed_batch(images):
            calls.append(len(images))
            return np.ones((len(images), 512), dtype=np.float32)

        monkeypatch.setattr(indexer, "detect_faces", lambda bgr: [])
        monkeypatch.setattr(indexer, "embed_batch", fake_embed_batch)
        indexer._process_batch([a, b])

        # 'a' already had a clip, so only 'b' reaches the batched embedding
        # and a's original vector is left untouched (reclaim fast-path).
        assert calls == [1]
        clips = {r["photo_uid"]: r["embedding"] for r in store.all_clips()}
        assert bytes(clips[a]) == keep.tobytes()
        assert store.get_photo(b)["status"] == "done"

    def test_missing_work_file_errors_alongside_batch(self, tmp_db, app_settings, monkeypatch):
        good = self._seed_work_photo("g1")
        store.upsert_photos(
            [{"uid": "gone", "name": "gone", "media_type": "image/jpeg", "capture_time": 1}]
        )
        assert store.claim_photo_for_download("gone") is True

        calls = []

        def fake_embed_batch(images):
            calls.append(len(images))
            return np.ones((len(images), 512), dtype=np.float32)

        monkeypatch.setattr(indexer, "detect_faces", lambda bgr: [])
        monkeypatch.setattr(indexer, "embed_batch", fake_embed_batch)
        indexer._process_batch([good, "gone"])

        assert calls == [1]
        assert store.get_photo(good)["status"] == "done"
        assert store.get_photo("gone")["status"] == "error"
        assert store.get_photo("gone")["error"] == "work file missing"

    def test_process_one_uses_passed_clip_vec(self, tmp_db, app_settings, monkeypatch):
        # Regression for the single-photo path with a precomputed vector:
        # embed_pil must NOT be re-invoked when _process_batch provided one.
        uid = self._seed_work_photo("p1")
        monkeypatch.setattr(indexer, "detect_faces", lambda bgr: [])

        def boom(*a, **k):
            raise AssertionError("embed_pil must not run when a clip_vec is supplied")

        monkeypatch.setattr(indexer, "embed_pil", boom)
        vec = np.ones(512, dtype=np.float32)
        indexer._process_one(uid, clip_vec=vec)
        assert store.clip_exists(uid) is True
        assert store.get_photo(uid)["status"] == "done"


class TestSyncOnceDeletionSweep:
    """Regression: a full scan with zero new `gone` must still confirm
    pending_removal rows whose grace window has elapsed.

    Previously confirm_deletions() was gated on `if gone:`, so a scan that
    found no new deletions never promoted previously-staged rows — they stayed
    stuck in pending_removal until a scan with new deletions happened to run.
    """

    class _FakeBridge:
        def __init__(self, uids):
            self._uids = uids

        def timeline_ids(self, limit=0):
            return [{"uid": u, "captureTime": 1} for u in self._uids]

        def nodes(self, uids):
            return [
                {"uid": u, "name": f"{u}.jpg", "mediaType": "image/jpeg",
                 "captureTime": 1, "sha1": f"sha-{u}", "albums": []}
                for u in uids
            ]

    def _seed_photo(self, uid):
        store.upsert_photos(
            [{"uid": uid, "name": f"{uid}.jpg", "media_type": "image/jpeg",
              "capture_time": 1, "sha1": f"sha-{uid}", "albums": [], "size": 1}]
        )
        with store.get_conn() as conn:
            conn.execute("UPDATE photos SET status='done' WHERE uid=?", (uid,))

    def _set_pending_removal(self, uid, processed_at):
        with store.get_conn() as conn:
            conn.execute(
                "UPDATE photos SET status='pending_removal', processed_at=? WHERE uid=?",
                (processed_at, uid),
            )

    def test_confirms_past_grace_without_new_gone(self, tmp_db, app_settings, monkeypatch):
        # p1/p2 are still on the remote timeline (no new deletions this scan).
        # p3 was staged long ago (past grace); p4 was staged just now. The
        # scan must confirm p3 even though `gone` is empty.
        self._seed_photo("p1")
        self._seed_photo("p2")
        self._seed_photo("p3")
        self._seed_photo("p4")
        grace = indexer._deletion_grace_seconds()
        self._set_pending_removal("p3", int(time.time()) - grace - 60)
        self._set_pending_removal("p4", int(time.time()))

        monkeypatch.setattr(indexer, "get_bridge", lambda: self._FakeBridge(["p1", "p2"]))
        indexer._sync_once()

        assert store.get_photo("p1")["status"] == "done"
        assert store.get_photo("p2")["status"] == "done"
        assert store.get_photo("p3")["status"] == "deleted"
        assert store.get_photo("p3")["was_deleted_at"] is not None
        # Still within grace: must remain pending_removal.
        assert store.get_photo("p4")["status"] == "pending_removal"

    def test_confirms_past_grace_with_new_gone(self, tmp_db, app_settings, monkeypatch):
        # p1 is gone from the remote timeline (new deletion, staged this scan)
        # and p2 was staged earlier (past grace). p1 must be staged but stay
        # pending_removal; p2 must be confirmed deleted in the same scan.
        # p3 stays on the remote timeline so the listing isn't empty.
        self._seed_photo("p1")
        self._seed_photo("p2")
        self._seed_photo("p3")
        grace = indexer._deletion_grace_seconds()
        self._set_pending_removal("p2", int(time.time()) - grace - 60)

        monkeypatch.setattr(indexer.settings, "sync_deletion_threshold", 0.9)
        monkeypatch.setattr(indexer, "get_bridge", lambda: self._FakeBridge(["p3"]))
        indexer._sync_once()

        assert store.get_photo("p1")["status"] == "pending_removal"
        assert store.get_photo("p2")["status"] == "deleted"
        assert store.get_photo("p3")["status"] == "done"


class TestRebuildPendingKeyset:
    """_rebuild_pending must re-queue every 'downloading' photo whose work
    file exists, exactly once, using keyset pagination (no OFFSET)."""

    def _seed_downloading(self, uid):
        store.upsert_photos(
            [{"uid": uid, "name": uid, "media_type": "image/jpeg", "capture_time": 1}]
        )
        assert store.claim_photo_for_download(uid) is True
        return uid

    def _write_work(self, uid):
        work = indexer._work_path(uid)
        work.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((50, 40, 3), 128, dtype=np.uint8)).save(work, "WEBP")

    def test_requeues_all_with_work_file(self, tmp_db, app_settings, monkeypatch):
        # More rows than one batch would need 2001 rows; use a small batch to
        # force multiple pages cheaply.
        monkeypatch.setattr(indexer, "_pending", queue.Queue())
        uids = [self._seed_downloading(f"p{i}") for i in range(5)]
        for uid in uids:
            self._write_work(uid)
        # One photo without a work file must be skipped.
        self._seed_downloading("missing")

        indexer._rebuild_pending(batch=2)

        requeued = []
        while not indexer._pending.empty():
            requeued.append(indexer._pending.get())
        assert sorted(requeued) == sorted(uids)

    def test_no_work_file_not_requeued(self, tmp_db, app_settings, monkeypatch):
        monkeypatch.setattr(indexer, "_pending", queue.Queue())
        self._seed_downloading("p1")
        indexer._rebuild_pending()
        assert indexer._pending.empty()


class TestTipCheck:
    """_tip_check must only consult the tip uids, never the full local set."""

    class _FakeBridge:
        def __init__(self, uids):
            self._uids = uids

        def timeline_ids(self, limit=0):
            return [{"uid": u, "captureTime": 1} for u in self._uids]

    def _seed(self, uid, status="done"):
        store.upsert_photos(
            [{"uid": uid, "name": uid, "media_type": "image/jpeg",
              "capture_time": 1, "sha1": f"sha-{uid}", "albums": [], "size": 1}]
        )
        with store.get_conn() as conn:
            conn.execute("UPDATE photos SET status=? WHERE uid=?", (status, uid))

    def test_all_known_returns_false(self, tmp_db, app_settings, monkeypatch):
        self._seed("p1")
        self._seed("p2")
        monkeypatch.setattr(indexer, "get_bridge", lambda: self._FakeBridge(["p1", "p2"]))
        assert indexer._tip_check({"tip_size": 10}) is False

    def test_new_uid_returns_true(self, tmp_db, app_settings, monkeypatch):
        self._seed("p1")
        monkeypatch.setattr(indexer, "get_bridge", lambda: self._FakeBridge(["p1", "p2"]))
        assert indexer._tip_check({"tip_size": 10}) is True

    def test_deleted_uid_counts_as_unknown(self, tmp_db, app_settings, monkeypatch):
        self._seed("p1", status="deleted")
        monkeypatch.setattr(indexer, "get_bridge", lambda: self._FakeBridge(["p1"]))
        assert indexer._tip_check({"tip_size": 10}) is True

    def test_empty_listing_returns_false(self, tmp_db, app_settings, monkeypatch):
        monkeypatch.setattr(indexer, "get_bridge", lambda: self._FakeBridge([]))
        assert indexer._tip_check({"tip_size": 10}) is False



class TestExifOrientation:
    def _rotated_jpeg(self, path):
        img = Image.fromarray(np.full((40, 100, 3), 128, dtype=np.uint8))  # 100 wide, 40 tall
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation: rotate 90° CW on display
        img.save(path, "JPEG", exif=exif.tobytes())

    def test_resize_applies_orientation(self, tmp_path):
        src = tmp_path / "in.jpg"
        self._rotated_jpeg(src)
        dest = tmp_path / "out.webp"
        indexer._resize_to_thumb(src, dest)
        with Image.open(dest) as img:
            assert img.size == (40, 100)  # portrait after transpose

    def test_oriented_noop_without_tag(self, tmp_path):
        img = Image.fromarray(np.full((10, 20, 3), 128, dtype=np.uint8))
        assert indexer._oriented(img).size == (20, 10)


class _FakeBridge:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    def thumbnails(self, uids):
        self.calls.append(list(uids))
        b = self.behaviour
        if isinstance(b, Exception):
            raise b
        return {"results": [{"uid": u, "ok": True} for u in uids]}


class TestDownloaderTick:
    def _seed(self, n=3):
        store.upsert_photos([
            {"uid": f"n{i}", "name": f"n{i}", "media_type": "image/jpeg", "capture_time": i} for i in range(n)
        ])

    def test_transient_releases_claims_and_pauses(self, tmp_db, app_settings, monkeypatch):
        from bridge_client import BridgeTransientError
        self._seed()
        bridge = _FakeBridge(BridgeTransientError(503, "busy", 42.0))
        pause = indexer._downloader_tick(bridge)
        assert pause == 42.0
        assert all(store.get_photo(f"n{i}")["status"] == "new" for i in range(3))

    def test_generic_failure_batches_errors_and_paces(self, tmp_db, app_settings, monkeypatch):
        self._seed()
        pause = indexer._downloader_tick(_FakeBridge(RuntimeError("boom")))
        assert pause == indexer._DOWNLOAD_ERROR_PAUSE_SEC
        assert all(store.get_photo(f"n{i}")["status"] == "error" for i in range(3))

    def test_success_queues_work(self, tmp_db, app_settings, monkeypatch):
        self._seed()
        q = queue.Queue()
        monkeypatch.setattr(indexer, "_pending", q)
        assert indexer._downloader_tick(_FakeBridge(None)) == 0.0
        assert q.qsize() == 3

    def test_idle(self, tmp_db, app_settings):
        assert indexer._downloader_tick(_FakeBridge(None)) == 5.0


class TestFullresPickNext:
    def test_skips_backed_off_head(self, tmp_db, app_settings, monkeypatch):
        n = indexer._FULLRES_PICK_PAGE + 5
        with store.get_conn() as conn:
            conn.executemany(
                "INSERT INTO photos (uid, name, media_type, capture_time, status) "
                "VALUES (?, ?, 'video/mp4', ?, 'full')",
                [(f"f{i:03d}", f"f{i:03d}", i) for i in range(n)],
            )
        now = time.time()
        monkeypatch.setattr(indexer, "_fullres_backoff",
                            {f"f{i:03d}": now + 900 for i in range(indexer._FULLRES_PICK_PAGE)})
        assert indexer._fullres_pick_next(now) == f"f{indexer._FULLRES_PICK_PAGE:03d}"
        monkeypatch.setattr(indexer, "_fullres_backoff", {f"f{i:03d}": now + 900 for i in range(n)})
        assert indexer._fullres_pick_next(now) is None


class TestCleanupDeletedPurge:
    def test_purges_thumb_faces_clips_once(self, tmp_db, app_settings, monkeypatch):
        thumb = app_settings.thumb_dir / "del.webp"
        thumb.write_bytes(b"x")
        with store.get_conn() as conn:
            conn.execute(
                "INSERT INTO photos (uid, name, media_type, capture_time, status, thumb_path, was_deleted_at) "
                "VALUES ('del', 'del', 'image/jpeg', 1, 'deleted', 'del.webp', 1)",
            )
            conn.execute("INSERT INTO clips (photo_uid, embedding) VALUES ('del', ?)",
                         (np.ones(512, dtype=np.float32).tobytes(),))
        calls = []
        monkeypatch.setattr(indexer, "purge_deleted_photo_media",
                            lambda uids: calls.append(uids) or store.purge_deleted_photo_media(uids))
        indexer.cleanup_deleted()
        assert not thumb.exists()
        assert calls == [["del"]]
        assert store.clip_exists("del") is False
        indexer.cleanup_deleted()
        assert calls == [["del"]]  # already purged rows are not touched again


class TestSidecarFlush:
    def test_flush_retries_after_failure(self, monkeypatch):
        monkeypatch.setattr(indexer, "_sidcar_dirty", True)
        monkeypatch.setattr(indexer, "_sidcar_last_write", 0.0)
        monkeypatch.setattr(indexer, "_sidcar_write_face", lambda: (_ for _ in ()).throw(OSError("disk")))
        assert indexer._sidcar_flush() is False
        assert indexer._sidcar_dirty is True
        monkeypatch.setattr(indexer, "_sidcar_write_face", lambda: None)
        monkeypatch.setattr(indexer, "_sidcar_write_clip", lambda: None)
        assert indexer._sidcar_flush() is True
        assert indexer._sidcar_dirty is False

    def test_sidecar_thread_is_started(self, monkeypatch):
        import threading
        started = []
        monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
        monkeypatch.setattr(indexer, "init_db", lambda: None)
        monkeypatch.setattr(indexer, "_db_conn", lambda: __import__("contextlib").nullcontext(
            type("C", (), {"execute": lambda *a, **k: None})()))
        monkeypatch.setattr(indexer, "backfill_fullres_images", lambda: 0)
        monkeypatch.setattr(indexer, "reset_stuck_fullres", lambda retry_after_sec: (0, 0))
        monkeypatch.setattr(indexer, "_rebuild_pending", lambda: None)
        monkeypatch.setattr(indexer, "_sync_albums_once", lambda: None)
        monkeypatch.setattr(indexer, "_record_threads", lambda threads: None)
        indexer.start()
        assert "sidecar" in started


class TestAtomicConfigWrite:
    def test_no_tmp_left_and_content(self, tmp_path):
        target = tmp_path / "cfg.json"
        indexer._write_json_atomic(target, {"a": 1})
        assert json.loads(target.read_text()) == {"a": 1}
        assert list(tmp_path.iterdir()) == [target]


class TestDeletionGrace:
    def test_grace_follows_full_scan_cadence(self, tmp_db, app_settings, monkeypatch):
        monkeypatch.setattr(indexer.settings, "sync_interval", 300)
        monkeypatch.setattr(indexer.settings, "grace_cycles", 2)
        indexer.set_sync_config({"full_scan_interval": 21600})
        assert indexer._deletion_grace_seconds() == 2 * 21600
        indexer.set_sync_config({"full_scan_interval": 0})
        assert indexer._deletion_grace_seconds() == 2 * 300



class TestSyncBatchesNodes:
    class _Bridge:
        def __init__(self):
            self.batches = []

        def timeline_ids(self, limit=0):
            return [{"uid": f"n{i:03d}", "captureTime": 1} for i in range(12)]

        def nodes(self, uids, **kw):
            self.batches.append(len(uids))
            return [{"uid": u, "name": f"{u}.jpg", "mediaType": "image/jpeg", "captureTime": 1,
                     "sha1": f"s-{u}", "albums": []} for u in uids]

        def albums(self):
            return {"albums": []}

    def test_nodes_fetched_in_bounded_batches(self, tmp_db, app_settings, monkeypatch):
        monkeypatch.setattr(indexer, "NODES_BATCH", 5)
        b = self._Bridge()
        monkeypatch.setattr(indexer, "get_bridge", lambda: b)
        indexer._sync_once()
        assert b.batches == [5, 5, 2]
        assert store.count_photos_by_status("new") == 12 if hasattr(store, "count_photos_by_status") else True
        with store.get_conn() as conn:
            assert conn.execute("SELECT COUNT(*) FROM photos WHERE status='new'").fetchone()[0] == 12


class TestEnrichPlacesChunked:
    def test_chunks_updates(self, tmp_db, app_settings, monkeypatch):
        monkeypatch.setattr(indexer, "UPDATE_CHUNK", 2)
        store.upsert_photos([{"uid": f"g{i}", "name": f"g{i}", "media_type": "image/jpeg", "capture_time": i}
                             for i in range(5)])
        for i in range(5):
            store.set_photo_gps(f"g{i}", 10.0 + i, 20.0)
        monkeypatch.setattr(indexer, "reverse_geocode_many",
                            lambda pts: {p: (None if p[0] == 12.0 else f"Place{p[0]}") for p in pts})
        assert indexer.enrich_places() == 4
        assert store.get_photo("g2")["place"] is None
        assert store.get_photo("g4")["place"] == "Place14.0"

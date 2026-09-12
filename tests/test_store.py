import contextlib
import json
import threading
import time

import numpy as np
import pytest

import config
import store
from conftest import APP_SRC  # noqa: F401  (ensures sys.path wiring)

EMB = np.zeros(512, dtype=np.float32)


def _embedding(seed=0.0):
    v = np.full(512, seed, dtype=np.float32)
    v[0] = 1.0
    return v / np.linalg.norm(v)


def _photo(uid="p1", media_type="image/jpeg", capture_time=1000, **overrides):
    row = {
        "uid": uid,
        "name": f"{uid}.jpg",
        "media_type": media_type,
        "capture_time": capture_time,
        "sha1": f"sha-{uid}",
        "albums": [],
        "size": 1234,
    }
    row.update(overrides)
    return row


def _set_processed_at(uid, ts):
    with store.get_conn() as conn:
        conn.execute("UPDATE photos SET processed_at=? WHERE uid=?", (ts, uid))


def _set_status(uid, status, retry_count=0):
    with store.get_conn() as conn:
        conn.execute(
            "UPDATE photos SET status=?, retry_count=? WHERE uid=?", (status, retry_count, uid)
        )


class TestInitAndUpsert:
    def test_init_db_creates_tables(self, tmp_db):
        with store.get_conn() as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"photos", "people", "faces", "clips", "albums", "users", "auth_tokens"} <= tables

    def test_init_db_creates_poll_composite_index(self, tmp_db):
        # Composite (status, capture_time) serves the indexer poll query
        # WHERE status=? ORDER BY capture_time.
        with store.get_conn() as conn:
            idx = {r[1] for r in conn.execute("PRAGMA index_list(photos)")}
        assert "idx_photos_status_time" in idx

    def test_migrate_creates_poll_composite_index(self, tmp_db):
        # Simulate a DB that predates idx_photos_status_time: drop the one from
        # _SCHEMA and confirm migrate() recreates it for existing installs.
        with store.get_conn() as conn:
            conn.execute("DROP INDEX idx_photos_status_time")
        with store.get_conn() as conn:
            store.migrate(conn)
            idx = {r[1] for r in conn.execute("PRAGMA index_list(photos)")}
        assert "idx_photos_status_time" in idx

    def test_migrate_backfills_issue85_join_tables(self, tmp_db):
        # Simulate a pre-#85 database: photos still carry JSON tags/albums, but
        # the normalized join tables are empty (or didn't exist). init_db()
        # must backfill them so tagged/album lookups stay sargable.
        store.upsert_photos([_photo("p1"), _photo("p2", capture_time=2000)])
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        with store.get_conn() as conn:
            conn.execute("UPDATE photos SET tags=? WHERE uid='p1'", (json.dumps(["sunset"]),))
            conn.execute("UPDATE photos SET tags=? WHERE uid='p2'", (json.dumps(["sunset", "sea"]),))
            conn.execute("UPDATE photos SET albums=? WHERE uid='p1'", (json.dumps(["al1"]),))
            conn.execute("UPDATE photos SET albums=? WHERE uid='p2'", (json.dumps(["al1", "al2"]),))
            conn.execute("DELETE FROM photo_tags")
            conn.execute("DELETE FROM photo_albums")
            conn.execute("PRAGMA user_version = 0")
        store.init_db()  # idempotent migrate + _SCHEMA + one-time backfill
        assert [r["uid"] for r in store.photos_by_tag("sunset")] == ["p2", "p1"]
        assert store.all_tags() == [("sunset", 2), ("sea", 1)]
        assert [r["uid"] for r in store.album_photos("al1")] == ["p2", "p1"]
        assert [r["uid"] for r in store.album_photos("al2")] == ["p2"]
        with store.get_conn() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == store._SCHEMA_VERSION

    def test_init_db_is_idempotent(self, tmp_db):
        store.upsert_photos([_photo("p1")])
        store.init_db()
        store.init_db()  # second run must not clobber anything
        assert store.get_photo("p1")["status"] == "new"

    def test_migrate_backfills_denormalized_counts(self, tmp_db):
        # Simulate a pre-#81 row whose counts were never populated (or were
        # zeroed). migrate() must recount it and create the sort index.
        store.upsert_photos([_photo("p1"), _photo("p2")])
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        pid = store.create_person("Bob", "p1", None)
        store.insert_face("p1", pid, 0.9, "[]", EMB.tobytes())
        store.insert_face("p2", pid, 0.9, "[]", EMB.tobytes())
        with store.get_conn() as conn:
            conn.execute("UPDATE people SET face_count=0, photo_count=0 WHERE id=?", (pid,))
            store.migrate(conn)
            row = conn.execute(
                "SELECT face_count, photo_count FROM people WHERE id=?", (pid,)
            ).fetchone()
            assert (row["face_count"], row["photo_count"]) == (2, 2)
            idx = {r[1] for r in conn.execute("PRAGMA index_list(people)")}
        assert "idx_people_photo_count" in idx

    def test_upsert_new_photo(self, tmp_db):
        assert store.upsert_photos([_photo()]) == 1
        row = store.get_photo("p1")
        assert row["status"] == "new"
        assert row["media_type"] == "image/jpeg"
        assert row["name"] == "p1.jpg"

    def test_upsert_existing_counts_zero(self, tmp_db):
        store.upsert_photos([_photo()])
        assert store.upsert_photos([_photo()]) == 0

    def test_upsert_reclaims_deleted_photo(self, tmp_db):
        store.upsert_photos([_photo()])
        _set_status("p1", "deleted")
        assert store.upsert_photos([_photo()]) == 0
        row = store.get_photo("p1")
        assert row["status"] == "new"
        assert row["was_deleted_at"] is None

    def test_upsert_reclaims_pending_removal(self, tmp_db):
        store.upsert_photos([_photo()])
        _set_status("p1", "pending_removal")
        store.upsert_photos([_photo()])
        assert store.get_photo("p1")["status"] == "new"

    def test_upsert_empty(self, tmp_db):
        assert store.upsert_photos([]) == 0


class TestDeletionLifecycle:
    def test_mark_pending_removal_sets_processed_at(self, tmp_db):
        store.upsert_photos([_photo()])
        assert store.mark_pending_removal(["p1"]) == 1
        assert store.get_photo("p1")["status"] == "pending_removal"
        # Idempotent on second call
        assert store.mark_pending_removal(["p1"]) == 0

    def test_mark_pending_removal_preserves_original_processed_at(self, tmp_db):
        store.upsert_photos([_photo()])
        _set_processed_at("p1", 111)
        store.mark_pending_removal(["p1"])
        _set_processed_at("p1", 222)
        store.mark_pending_removal(["p1"])  # no-op, keeps 222
        assert store.get_photo("p1")["processed_at"] == 222

    def test_confirm_deletions_within_grace_keeps_pending(self, tmp_db):
        store.upsert_photos([_photo()])
        store.mark_pending_removal(["p1"])
        _set_processed_at("p1", 1000)
        assert store.confirm_deletions(grace_seconds=60, now=1040) == 0
        assert store.get_photo("p1")["status"] == "pending_removal"

    def test_confirm_deletions_after_grace(self, tmp_db):
        store.upsert_photos([_photo()])
        store.mark_pending_removal(["p1"])
        _set_processed_at("p1", 1000)
        assert store.confirm_deletions(grace_seconds=60, now=1061) == 1
        row = store.get_photo("p1")
        assert row["status"] == "deleted"
        assert row["was_deleted_at"] == 1061

    def test_mark_deleted(self, tmp_db):
        store.upsert_photos([_photo()])
        store.mark_deleted(["p1"])
        assert store.get_photo("p1")["status"] == "deleted"


class TestFullresRetry:
    def test_requeues_stuck_fullres(self, tmp_db):
        store.upsert_photos([_photo()])
        _set_status("p1", "full", retry_count=0)
        _set_processed_at("p1", 1000)
        requeued, parked = store.reset_stuck_fullres(retry_after_sec=60, now=2000)
        assert (requeued, parked) == (1, 0)
        row = store.get_photo("p1")
        assert row["status"] == "new"
        assert row["retry_count"] == 1

    def test_parks_after_max_retries(self, tmp_db):
        store.upsert_photos([_photo()])
        _set_status("p1", "full", retry_count=5)
        _set_processed_at("p1", 1000)
        requeued, parked = store.reset_stuck_fullres(retry_after_sec=60, max_retry_count=5, now=2000)
        assert (requeued, parked) == (0, 1)
        row = store.get_photo("p1")
        assert row["status"] == "error"
        assert "stuck in full" in row["error"]

    def test_fresh_fullres_not_touched(self, tmp_db):
        store.upsert_photos([_photo()])
        _set_status("p1", "full")
        _set_processed_at("p1", 2000)
        assert store.reset_stuck_fullres(retry_after_sec=60, now=2000) == (0, 0)


class TestPhotoClaims:
    def test_download_processing_flow(self, tmp_db):
        store.upsert_photos([_photo()])
        assert store.claim_photo_for_download("p1") is True
        assert store.claim_photo_for_download("p1") is False
        assert store.claim_photo_for_processing("p1") is True
        assert store.claim_photo_for_processing("p1") is False

    def test_claim_requires_new_status(self, tmp_db):
        store.upsert_photos([_photo()])
        _set_status("p1", "done")
        assert store.claim_photo_for_download("p1") is False

    def test_full_claim_flow(self, tmp_db):
        store.upsert_photos([_photo()])
        _set_status("p1", "full")
        assert store.claim_photo_for_full("p1") is True
        assert store.get_photo("p1")["status"] == "fullres"
        assert store.claim_photo_for_full("p1") is False

    def test_set_photo_done(self, tmp_db):
        store.upsert_photos([_photo()])
        store.set_photo_done("p1", "thumbs/p1.webp", (45.0, 9.0), "Milano")
        row = store.get_photo("p1")
        assert row["status"] == "done"
        assert row["thumb_path"] == "thumbs/p1.webp"
        assert row["gps_lat"] == pytest.approx(45.0)
        assert row["place"] == "Milano"
        assert row["processed_at"] is not None

    def test_set_photo_full_and_error(self, tmp_db):
        store.upsert_photos([_photo()])
        store.set_photo_full("p1")
        assert store.get_photo("p1")["status"] == "full"
        store.set_photo_error("p1", "boom")
        assert store.get_photo("p1")["error"] == "boom"

    def test_set_photo_gps(self, tmp_db):
        store.upsert_photos([_photo()])
        store.set_photo_gps("p1", 39.564, 2.619)
        row = store.get_photo("p1")
        assert row["gps_lat"] == pytest.approx(39.564)
        assert row["gps_lng"] == pytest.approx(2.619)

    def test_get_photos_without_gps(self, tmp_db):
        store.upsert_photos([_photo("p1", media_type="image/heic")])
        store.upsert_photos([_photo("p2", media_type="image/heic")])
        store.upsert_photos([_photo("p3", media_type="image/jpeg")])
        store.set_photo_gps("p2", 1.0, 2.0)
        rows = store.get_photos_without_gps()
        assert {r["uid"] for r in rows} == {"p1", "p3"}
        rows = store.get_photos_without_gps(media_type="image/heic")
        assert [r["uid"] for r in rows] == ["p1"]

    def test_backfill_fullres_images(self, tmp_db):
        store.upsert_photos([_photo("p1", media_type="image/heic")])
        store.upsert_photos([_photo("p2", media_type="video/mp4")])
        store.upsert_photos([_photo("p3", media_type="image/jpeg")])
        for uid in ("p1", "p2", "p3"):
            store.set_photo_done(uid, "", None, None)
        assert store.backfill_fullres_images() == 3
        for uid in ("p1", "p2", "p3"):
            assert store.get_photo(uid)["status"] == "full"

    def test_get_photos_filters_by_status(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2", capture_time=2000)])
        store.set_photo_done("p1", "t.webp", None, None)
        rows = store.get_photos("done")
        assert [r["uid"] for r in rows] == ["p1"]
        rows = store.get_photos("new", limit=1)
        assert [r["uid"] for r in rows] == ["p2"]

    def test_claim_photos_for_download_batch(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2"), _photo("p3")])
        _set_status("p2", "done")
        claimed = store.claim_photos_for_download(["p1", "p2", "p3"])
        assert sorted(claimed) == ["p1", "p3"]
        assert store.get_photo("p1")["status"] == "downloading"
        assert store.get_photo("p2")["status"] == "done"
        assert store.get_photo("p3")["status"] == "downloading"
        # Already-claimed uids are left alone on a second pass.
        assert store.claim_photos_for_download(["p1", "p3"]) == []

    def test_claim_photos_for_download_empty(self, tmp_db):
        assert store.claim_photos_for_download([]) == []

    def test_claim_photos_for_download_unknown_uids(self, tmp_db):
        store.upsert_photos([_photo("p1")])
        claimed = store.claim_photos_for_download(["p1", "nope"])
        assert claimed == ["p1"]


class TestTransaction:
    def test_commit(self, tmp_db):
        store.upsert_photos([_photo()])
        with store.transaction() as conn:
            conn.execute("UPDATE photos SET status='processing' WHERE uid='p1'")
        assert store.get_photo("p1")["status"] == "processing"

    def test_rollback_on_exception(self, tmp_db):
        store.upsert_photos([_photo()])
        with pytest.raises(RuntimeError):
            with store.transaction() as conn:
                conn.execute("UPDATE photos SET status='processing' WHERE uid='p1'")
                raise RuntimeError("boom")
        assert store.get_photo("p1")["status"] == "new"

    def test_transaction_rollback_is_swallowed(self, tmp_db):
        store.upsert_photos([_photo()])
        with store.transaction() as conn:
            conn.execute("UPDATE photos SET status='processing' WHERE uid='p1'")
            raise store.TransactionRollback
        assert store.get_photo("p1")["status"] == "new"

    def test_helpers_share_transaction_conn(self, tmp_db):
        # Everything written on the shared conn is one commit: the claim, the
        # face insert and the error mark appear together — or not at all.
        store.upsert_photos([_photo()])
        assert store.claim_photo_for_download("p1") is True
        with pytest.raises(RuntimeError):
            with store.transaction() as conn:
                assert store.claim_photo_for_processing("p1", conn) is True
                store.insert_face("p1", None, 0.9, "[0,0,10,10]", EMB.tobytes(), conn=conn)
                store.set_photo_error("p1", "boom", conn=conn)
                raise RuntimeError("boom")
        # The whole batch was rolled back: p1 is back at its pre-transaction
        # status and the face insert never landed.
        assert store.get_photo("p1")["status"] == "downloading"
        assert store.count_faces_for_photo("p1") == 0


class TestStats:
    def test_stats_counts(self, tmp_db):
        store.upsert_photos([_photo("p1", media_type="image/jpeg"), _photo("p2", media_type="video/mp4")])
        store.set_photo_done("p1", "t.webp", None, None)
        store.set_photo_done("p2", "", None, None)
        face_id = store.insert_face("p1", None, 0.99, json.dumps([0, 0, 10, 10]), EMB.tobytes())
        store.insert_clip("p1", EMB.tobytes())
        store.create_person("Alice", "p1", face_id)
        s = store.stats()
        assert s["photos"]["total"] == 2
        assert s["photos"]["done"] == 2
        assert s["photos"]["done_with_thumb"] == 1
        assert s["photos"]["done_without_thumb_videos"] == 1
        assert s["faces"] == 1
        assert s["clips"] == 1
        assert s["people"] == 1


class TestFaces:
    def _seed_photo_done(self, uid="p1"):
        store.upsert_photos([_photo(uid)])
        store.set_photo_done(uid, f"thumbs/{uid}.webp", None, None)

    def test_insert_face_and_count(self, tmp_db):
        self._seed_photo_done()
        fid = store.insert_face("p1", None, 0.9, json.dumps([1, 2, 3, 4]), EMB.tobytes())
        assert store.count_faces_for_photo("p1") == 1
        assert store.get_photo("p1")["uid"] == "p1"
        assert isinstance(fid, int)

    def test_face_counts_for_photos_batch(self, tmp_db):
        self._seed_photo_done("p1")
        self._seed_photo_done("p2")
        self._seed_photo_done("p3")
        store.insert_face("p1", None, 0.9, json.dumps([1, 2, 3, 4]), EMB.tobytes())
        store.insert_face("p1", None, 0.9, json.dumps([5, 6, 7, 8]), EMB.tobytes())
        store.insert_face("p3", None, 0.9, json.dumps([1, 2, 3, 4]), EMB.tobytes())
        assert store.face_counts_for_photos(["p1", "p2", "p3"]) == {"p1": 2, "p3": 1}
        # empty input is a no-op
        assert store.face_counts_for_photos([]) == {}
        # uids with no faces stay absent from the result
        assert "p2" not in store.face_counts_for_photos(["p2", "nope"])

    def test_faces_without_person_and_assign(self, tmp_db):
        self._seed_photo_done()
        fid = store.insert_face("p1", None, 0.9, json.dumps([1, 2, 3, 4]), EMB.tobytes())
        pid = store.create_person("Bob", "p1", fid)
        store.assign_face_person(fid, pid)
        assert store.faces_without_person() == []
        store.unassign_face(fid)
        assert len(store.faces_without_person()) == 1

    def test_faces_without_person_limit(self, tmp_db):
        self._seed_photo_done()
        fids = [
            store.insert_face("p1", None, 0.9, json.dumps([0, i, 10, i + 10]), EMB.tobytes())
            for i in range(5)
        ]
        assert len(store.faces_without_person(limit=2)) == 2
        # ordered by id, so the first two rows are the earliest inserts
        got = store.faces_without_person(limit=2)
        assert [r["id"] for r in got] == fids[:2]
        # limit=None returns every unassigned face (no cap)
        assert len(store.faces_without_person(limit=None)) == 5

    def test_assign_faces_person_bulk(self, tmp_db):
        self._seed_photo_done()
        fids = [
            store.insert_face("p1", None, 0.9, json.dumps([0, i, 10, i + 10]), EMB.tobytes())
            for i in range(3)
        ]
        pid = store.create_person("Bulk", "p1", fids[0])
        store.assign_faces_person_bulk(fids, pid)
        assert store.faces_without_person() == []
        from store import faces_for_photo

        assert {f["person_id"] for f in faces_for_photo("p1")} == {pid}

    def test_assign_faces_person_bulk_empty(self, tmp_db):
        self._seed_photo_done()
        pid = store.create_person("Bulk", "p1", None)
        store.assign_faces_person_bulk([], pid)
        assert store.get_person(pid) is not None

    def test_create_and_get_person(self, tmp_db):
        self._seed_photo_done()
        fid = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        pid = store.create_person("Carol", "p1", fid)
        store.assign_face_person(fid, pid)
        person = store.get_person(pid)
        assert person["name"] == "Carol"
        assert person["cover_uid"] == "p1"
        assert person["face_count"] == 1
        assert person["photo_count"] == 1

    def test_update_person_cover_only_when_unset(self, tmp_db):
        self._seed_photo_done()
        fid = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        pid = store.create_person("Dave", "p1", fid)
        store.update_person_cover(pid, "other")
        assert store.get_person(pid)["cover_uid"] == "p1"

    def test_rename_and_find_person(self, tmp_db):
        self._seed_photo_done()
        fid = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        pid = store.create_person("Erin", "p1", fid)
        store.rename_person(pid, "Erin B")
        assert store.get_person(pid)["name"] == "Erin B"
        assert store.find_person_by_name("erin b")["id"] == pid
        assert store.find_person_by_name("nope") is None

    def test_merge_person(self, tmp_db):
        self._seed_photo_done("p1")
        self._seed_photo_done("p2")
        f1 = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        f2 = store.insert_face("p2", None, 0.8, "[]", EMB.tobytes())
        a = store.create_person("A", "p1", f1)
        b = store.create_person("B", "p2", f2)
        store.assign_face_person(f2, b)
        store.merge_person(b, a)
        assert store.get_person(a) is not None
        assert store.get_person(b) is None
        assert store.get_photo("p2") is not None
        # faces moved to target
        from store import faces_for_photo

        assert {f["person_id"] for f in faces_for_photo("p2")} == {a}

    def test_live_person_ids_filters_deleted(self, tmp_db):
        self._seed_photo_done("p1")
        self._seed_photo_done("p2")
        f1 = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        f2 = store.insert_face("p2", None, 0.8, "[]", EMB.tobytes())
        a = store.create_person("A", "p1", f1)
        b = store.create_person("B", "p2", f2)
        store.assign_face_person(f2, b)
        store.merge_person(b, a)
        # b is gone from the DB; a survives; a never-existing id is dropped.
        assert store.live_person_ids([a, b, 999999]) == [a]
        assert store.live_person_ids([]) == []
        # A large list is chunked safely (no 999-placeholder overflow).
        many = list(range(1, 1200))
        assert store.live_person_ids(many) == [a]

    def test_faces_for_photo_joins_name(self, tmp_db):
        self._seed_photo_done()
        fid = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        pid = store.create_person("Grace", "p1", fid)
        store.assign_face_person(fid, pid)
        faces = store.faces_for_photo("p1")
        assert faces[0]["person_name"] == "Grace"

    def test_face_embedding(self, tmp_db):
        self._seed_photo_done()
        fid = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        assert store.face_embedding(fid) == EMB.tobytes()

    def test_unassigned_faces(self, tmp_db):
        self._seed_photo_done()
        store.insert_face("p1", None, 0.9, json.dumps([0, 0, 10, 10]), EMB.tobytes())
        rows = store.unassigned_faces()
        assert len(rows) == 1
        assert rows[0]["thumb_path"] == "thumbs/p1.webp"


class TestDenormalizedCounts:
    """Denormalized people.face_count / photo_count (issue #81) stay correct
    across every face<->person write site."""

    def _seed_photo_done(self, uid):
        store.upsert_photos([_photo(uid)])
        store.set_photo_done(uid, f"thumbs/{uid}.webp", None, None)

    def test_counts_follow_insert_assign_unassign(self, tmp_db):
        self._seed_photo_done("p1")
        self._seed_photo_done("p2")
        pid = store.create_person("Carol", "p1", None)
        assert (store.get_person(pid)["face_count"], store.get_person(pid)["photo_count"]) == (0, 0)
        f1 = store.insert_face("p1", pid, 0.9, "[]", EMB.tobytes())
        store.insert_face("p2", pid, 0.9, "[]", EMB.tobytes())
        # same person, same photo -> 2 faces, 1 distinct photo
        store.insert_face("p1", pid, 0.9, "[]", EMB.tobytes())
        assert (store.get_person(pid)["face_count"], store.get_person(pid)["photo_count"]) == (3, 2)
        store.unassign_face(f1)
        assert (store.get_person(pid)["face_count"], store.get_person(pid)["photo_count"]) == (2, 2)
        store.assign_face_person(f1, pid)
        assert (store.get_person(pid)["face_count"], store.get_person(pid)["photo_count"]) == (3, 2)

    def test_counts_follow_assign_bulk(self, tmp_db):
        self._seed_photo_done("p1")
        self._seed_photo_done("p2")
        pid = store.create_person("Bulk", "p1", None)
        fids = []
        for uid in ["p1", "p1", "p2"]:
            fids.append(store.insert_face(uid, None, 0.9, "[]", EMB.tobytes()))
        store.assign_faces_person_bulk(fids, pid)
        assert (store.get_person(pid)["face_count"], store.get_person(pid)["photo_count"]) == (3, 2)

    def test_counts_follow_assign_move_between_people(self, tmp_db):
        self._seed_photo_done("p1")
        a = store.create_person("A", "p1", None)
        b = store.create_person("B", "p1", None)
        f1 = store.insert_face("p1", a, 0.9, "[]", EMB.tobytes())
        store.insert_face("p1", b, 0.9, "[]", EMB.tobytes())
        assert (store.get_person(a)["face_count"], store.get_person(a)["photo_count"]) == (1, 1)
        assert (store.get_person(b)["face_count"], store.get_person(b)["photo_count"]) == (1, 1)
        # move f1 from a -> b: a drops to 0, b rises to 2
        store.assign_face_person(f1, b)
        assert (store.get_person(a)["face_count"], store.get_person(a)["photo_count"]) == (0, 0)
        assert (store.get_person(b)["face_count"], store.get_person(b)["photo_count"]) == (2, 1)

    def test_counts_follow_merge_person(self, tmp_db):
        self._seed_photo_done("p1")
        self._seed_photo_done("p2")
        f1 = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        f2 = store.insert_face("p2", None, 0.9, "[]", EMB.tobytes())
        a = store.create_person("A", "p1", f1)
        b = store.create_person("B", "p2", f2)
        store.assign_face_person(f1, a)
        store.assign_face_person(f2, b)
        store.merge_person(b, a)
        assert store.get_person(b) is None
        assert (store.get_person(a)["face_count"], store.get_person(a)["photo_count"]) == (2, 2)

    def test_counts_follow_merge_people_bulk(self, tmp_db):
        self._seed_photo_done("p1")
        self._seed_photo_done("p2")
        self._seed_photo_done("p3")
        target = store.create_person("T", "p1", None)
        sources = []
        for uid in ["p1", "p2", "p3"]:
            pid = store.create_person(uid, uid, None)
            sources.append(pid)
            store.insert_face(uid, pid, 0.9, "[]", EMB.tobytes())
        merged = store.merge_people_bulk(sources, target)
        assert merged == 3
        assert (store.get_person(target)["face_count"], store.get_person(target)["photo_count"]) == (3, 3)

    def test_all_people_uses_denormalized_counts(self, tmp_db):
        self._seed_photo_done("p1")
        self._seed_photo_done("p2")
        pid = store.create_person("Zed", "p1", None)
        store.insert_face("p1", pid, 0.9, "[]", EMB.tobytes())
        store.insert_face("p1", pid, 0.9, "[]", EMB.tobytes())
        store.insert_face("p2", pid, 0.9, "[]", EMB.tobytes())
        rows = store.all_people()
        row = next(r for r in rows if r["id"] == pid)
        assert (row["face_count"], row["photo_count"]) == (3, 2)

    def test_all_people_q_substring_collate_nocase(self, tmp_db):
        self._seed_photo_done("p1")
        for name in ["Alice", "alice", "Bob", "Alicia", "xalicia"]:
            store.create_person(name, "p1", None)
        hits = store.all_people(q="al")
        names = sorted(r["name"] for r in hits)
        # case-insensitive substring: everything containing "al", including
        # the non-prefix "xalicia".
        assert names == ["Alice", "Alicia", "alice", "xalicia"]
        # substring in the middle of the name matches too.
        assert sorted(r["name"] for r in store.all_people(q="lici")) == ["Alicia", "xalicia"]
        assert store.all_people(q="zzz") == []

    def test_all_people_q_escapes_wildcards(self, tmp_db):
        self._seed_photo_done("p1")
        for name in ["100% Real", "100X Real", "under_score", "underscore"]:
            store.create_person(name, "p1", None)
        # `%` and `_` in the query are matched literally, not as wildcards.
        assert sorted(r["name"] for r in store.all_people(q="100%")) == ["100% Real"]
        assert sorted(r["name"] for r in store.all_people(q="under_")) == ["under_score"]
        assert store.count_people(q="100%") == 1
        assert store.count_people(q="under_") == 1

    def test_all_people_q_prefix_uses_index(self, tmp_db):
        self._seed_photo_done("p1")
        store.create_person("Alice", "p1", None)
        store.create_person("Bob", "p1", None)
        with store.get_conn() as conn:
            plan = conn.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT id FROM people p "
                "WHERE p.name COLLATE NOCASE LIKE 'al%'"
            ).fetchall()
        assert any("idx_people_name" in str(row[3]) for row in plan)


class TestSimilarFaces:
    def test_similar_faces_matches_near_embedding(self, tmp_db):
        self_ = _embedding(0.5)
        other = _embedding(0.5)
        other[10] = 0.0
        other /= np.linalg.norm(other)
        store.upsert_photos([_photo("p1"), _photo("p2")])
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        store.insert_face("p1", None, 0.9, "[]", self_.tobytes())
        store.insert_face("p2", None, 0.9, "[]", other.tobytes())
        hits = store.similar_faces(self_.tobytes(), threshold=0.5)
        assert len(hits) == 2
        assert hits[0][0] != hits[1][0]

    def test_similar_faces_below_threshold(self, tmp_db):
        v = _embedding(0.5)
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        store.insert_face("p1", None, 0.9, "[]", v.tobytes())
        assert store.similar_faces(v.tobytes(), threshold=1.5) == []


class TestEmbeddingCacheSWR:
    def _seed_face(self, uid, seed):
        store.upsert_photos([_photo(uid)])
        store.set_photo_done(uid, f"thumbs/{uid}.webp", None, None)
        return store.insert_face(uid, None, 0.9, "[]", _embedding(seed).tobytes())

    def _force_expiry(self):
        store._embedding_cache_ts -= store._EMBEDDING_CACHE_TTL + 1

    def _wait_refresh(self, timeout=5.0):
        deadline = time.time() + timeout
        while store._embedding_cache_refreshing and time.time() < deadline:
            time.sleep(0.01)

    def test_first_load_is_synchronous(self, tmp_db):
        self._seed_face("p1", 0.1)
        assert not store._embedding_cache_refreshing
        data = store._embedding_cache_data()
        assert data["mat"].shape[0] == 1
        assert not store._embedding_cache_refreshing

    def test_expired_serves_stale_then_background_refresh_swaps(self, tmp_db):
        self._seed_face("p1", 0.1)
        first = store._embedding_cache_data()
        assert first["mat"].shape[0] == 1
        self._force_expiry()
        stale = store._embedding_cache_data()
        # The expired call must return the existing object immediately (no
        # blocking rebuild), and kick off a background refresh.
        assert stale is first
        assert store._embedding_cache_refreshing
        self._wait_refresh()
        # A new face added after expiry is picked up by the swap.
        self._seed_face("p2", 0.2)
        self._force_expiry()
        store._embedding_cache_data()
        self._wait_refresh()
        assert store._embedding_cache["mat"].shape[0] == 2

    def test_concurrent_expired_callers_share_one_refresh(self, tmp_db, monkeypatch):
        self._seed_face("p1", 0.1)
        store._embedding_cache_data()
        self._force_expiry()
        real_build = store._build_embedding_cache
        started = threading.Event()
        release = threading.Event()
        builds = {"n": 0}

        def slow_build():
            builds["n"] += 1
            started.set()
            release.wait(5)
            return real_build()

        monkeypatch.setattr(store, "_build_embedding_cache", slow_build)
        results = []
        errors = []

        def call():
            try:
                results.append(store._embedding_cache_data())
            except Exception as exc:  # pragma: no cover - failure surface
                errors.append(exc)

        threads = [threading.Thread(target=call) for _ in range(5)]
        for t in threads:
            t.start()
        # Wait until the single background refresh is in-flight and blocked.
        assert started.wait(5)
        for t in threads:
            t.join()
        assert not errors
        # Every caller got the stale cache and only one refresh was launched.
        assert all(r is store._embedding_cache for r in results)
        assert builds["n"] == 1
        # Let the refresh finish and swap in the rebuilt cache.
        release.set()
        self._wait_refresh()
        assert store._embedding_cache["mat"].shape[0] == 1


class TestPersonMeans:
    def test_person_mean_embeddings(self, tmp_db):
        v = _embedding(0.5)
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        fid = store.insert_face("p1", None, 0.9, "[]", v.tobytes())
        pid = store.create_person("Hank", "p1", fid)
        store.assign_face_person(fid, pid)
        means = store.person_mean_embeddings()
        assert pid in means
        assert np.linalg.norm(means[pid]) == pytest.approx(1.0)

    def test_person_mean_embeddings_from_cache(self, tmp_db):
        v = _embedding(0.5)
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        fid = store.insert_face("p1", None, 0.9, "[]", v.tobytes())
        pid = store.create_person("Iris", "p1", fid)
        store.assign_face_person(fid, pid)
        means = store.person_mean_embeddings_from_cache()
        assert pid in means

    def test_person_mean_matrix_from_cache(self, tmp_db):
        v = _embedding(0.5)
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        fid = store.insert_face("p1", None, 0.9, "[]", v.tobytes())
        pid = store.create_person("Iris", "p1", fid)
        store.assign_face_person(fid, pid)
        store.person_mean_embeddings_from_cache()
        pids, M = store.person_mean_matrix_from_cache()
        assert M.shape == (1, 512)
        assert M.dtype == np.float32
        assert pids.dtype == np.int64
        # Rows align 1:1 with the dict keys, in sorted order.
        means = store.person_mean_embeddings_from_cache()
        assert list(pids) == sorted(means.keys())
        assert np.allclose(M[0], means[pid])
        # The cached generation is not re-stacked per request: a second call
        # returns the exact same (pids, M) tuple.
        pids2, M2 = store.person_mean_matrix_from_cache()
        assert pids2 is pids
        assert M2 is M

    def test_person_mean_matrix_from_cache_empty(self, tmp_db):
        pids, M = store.person_mean_matrix_from_cache()
        assert pids.shape == (0,)
        assert pids is store._EMPTY_PIDS
        assert M.shape == (0, 512)
        assert M.dtype == np.float32
        assert M is store._EMPTY_MAT

    def test_person_mean_matrix_invalidates_with_generation(self, tmp_db):
        v = _embedding(0.5)
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        fid = store.insert_face("p1", None, 0.9, "[]", v.tobytes())
        pid = store.create_person("Iris", "p1", fid)
        store.assign_face_person(fid, pid)
        store.person_mean_embeddings_from_cache()
        _, M1 = store.person_mean_matrix_from_cache()
        # Simulate the embedding matrix reloading (different generation): the
        # cached means/matrix are dropped and rekeyed to the new ts.
        store._person_means_cache = None
        store._person_means_cache_ts = 0.0
        store._person_means_matrix = None
        store._embedding_cache_ts += 1000
        pids2, M2 = store.person_mean_matrix_from_cache()
        assert M2 is not M1  # rebuilt for the new generation
        assert store._person_means_cache_ts == store._embedding_cache_ts
        assert M2.shape == (1, 512)
        assert np.allclose(M2[0], store._person_means_cache[pid])

    def test_invalidate_embedding_cache_drops_all_layers(self, tmp_db):
        v = _embedding(0.5)
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        fid = store.insert_face("p1", None, 0.9, "[]", v.tobytes())
        pid = store.create_person("Iris", "p1", fid)
        store.assign_face_person(fid, pid)
        store.person_mean_embeddings_from_cache()
        store.person_mean_matrix_from_cache()
        assert store._embedding_cache is not None
        assert store._person_means_cache is not None
        assert store._person_means_matrix is not None
        store.invalidate_embedding_cache()
        assert store._embedding_cache is None
        assert store._embedding_cache_ts == 0.0
        assert store._person_means_cache is None
        assert store._person_means_cache_ts == 0.0
        assert store._person_means_matrix is None
        # Rebuild works and reflects the current DB state.
        pids, M = store.person_mean_matrix_from_cache()
        assert list(pids) == [pid]
        assert M.shape == (1, 512)


class TestMergePeopleBulk:
    def _seed_person(self, uid, name, emb_seed=0.0):
        store.upsert_photos([_photo(uid)])
        store.set_photo_done(uid, f"thumbs/{uid}.webp", None, None)
        fid = store.insert_face(uid, None, 0.9, "[]", _embedding(emb_seed).tobytes())
        pid = store.create_person(name, uid, fid)
        store.assign_face_person(fid, pid)
        return pid

    def test_merge_person_invalidates_embedding_cache(self, tmp_db):
        a = self._seed_person("p1", "A")
        b = self._seed_person("p2", "B")
        store.person_mean_matrix_from_cache()
        assert store._embedding_cache is not None
        store.merge_person(b, a)
        assert store._embedding_cache is None
        assert store._person_means_matrix is None
        # The rebuilt matrix no longer contains the merged-away person.
        pids, _ = store.person_mean_matrix_from_cache()
        assert b not in pids
        assert a in pids

    def test_merge_people_bulk_invalidates_embedding_cache(self, tmp_db):
        a = self._seed_person("p1", "A")
        b = self._seed_person("p2", "B")
        c = self._seed_person("p3", "C")
        store.person_mean_matrix_from_cache()
        assert store._embedding_cache is not None
        store.merge_people_bulk([b, c], a)
        assert store._embedding_cache is None
        pids, _ = store.person_mean_matrix_from_cache()
        assert b not in pids
        assert c not in pids
        assert a in pids

    def test_reparents_and_deletes(self, tmp_db):
        a = self._seed_person("p1", "A")
        b = self._seed_person("p2", "B")
        c = self._seed_person("p3", "C")
        assert store.merge_people_bulk([b, c], a) == 2
        assert store.get_person(b) is None
        assert store.get_person(c) is None
        # faces moved to target
        from store import faces_for_photo

        assert {f["person_id"] for f in faces_for_photo("p2")} == {a}
        assert {f["person_id"] for f in faces_for_photo("p3")} == {a}
        assert store.get_person(a)["face_count"] == 3

    def test_inherits_name_and_cover_when_target_unset(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2")])
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        fa = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        fb = store.insert_face("p2", None, 0.9, "[]", EMB.tobytes())
        target = store.create_person(None, None, None)      # unnamed target
        src = store.create_person("Source", "p2", fb)        # named source w/ cover
        store.assign_face_person(fa, target)
        store.assign_face_person(fb, src)
        store.merge_people_bulk([src], target)
        merged = store.get_person(target)
        assert merged["name"] == "Source"
        assert merged["cover_uid"] == "p2"
        assert merged["cover_face_id"] == fb

    def test_keeps_target_name(self, tmp_db):
        a = self._seed_person("p1", "Target")
        b = self._seed_person("p2", "Source")
        store.merge_people_bulk([b], a)
        assert store.get_person(a)["name"] == "Target"

    def test_skips_self_missing_and_duplicates(self, tmp_db):
        a = self._seed_person("p1", "A")
        b = self._seed_person("p2", "B")
        # target itself, a duplicate, and a non-existent id are all skipped
        assert store.merge_people_bulk([a, b, b, 99999], a) == 1
        assert store.get_person(b) is None
        assert store.get_person(a)["face_count"] == 2

    def test_many_sources_chunked(self, tmp_db):
        # More sources than the SQL chunk size (500) must not trip SQLite's
        # 999-variable placeholder limit.
        target = self._seed_person("p0", "Target")
        sources = [self._seed_person(f"p{i}", f"S{i}") for i in range(1, 620)]
        assert store.merge_people_bulk(sources, target) == 619
        assert store.get_person(target)["face_count"] == 620

    def test_commits_per_chunk(self, tmp_db, monkeypatch):
        # Issue #87: a bulk merge must commit after each ~500-id chunk, not
        # hold one giant transaction for the whole campaign (the WAL writer
        # lock then never blocks indexer claims / API writes for seconds).
        # 601 sources span two chunks: 2 chunk transactions + the closing
        # recount = 3 commit points. A single-transaction merge would be 2.
        target = self._seed_person("p0", "Target")
        sources = [self._seed_person(f"p{i}", f"S{i}") for i in range(1, 602)]
        real_get_conn = store.get_conn
        commits = 0

        @contextlib.contextmanager
        def counting_get_conn():
            nonlocal commits
            with real_get_conn() as conn:
                commits += 1
                yield conn

        monkeypatch.setattr(store, "get_conn", counting_get_conn)
        assert store.merge_people_bulk(sources, target) == 601
        assert commits >= 3
        assert store.get_person(target)["face_count"] == 602

    def test_face_ids_for_people(self, tmp_db):
        a = self._seed_person("p1", "A")
        b = self._seed_person("p2", "B")
        ids = store.face_ids_for_people([a, b])
        assert len(ids) == 2
        assert store.face_ids_for_people([]) == []
        assert store.face_ids_for_people([99999]) == []


class TestGcEmptyPeople:
    """Anonymous placeholder people rows are swept once they hold no faces."""

    def _backdate(self, pid, ts=1000):
        with store.get_conn() as conn:
            conn.execute("UPDATE people SET created=? WHERE id=?", (ts, pid))

    def _seed_photo_done(self, uid):
        store.upsert_photos([_photo(uid)])
        store.set_photo_done(uid, f"thumbs/{uid}.webp", None, None)

    def test_sweep_removes_old_unnamed_empty(self, tmp_db):
        pid = store.create_person(None, None, None)
        self._backdate(pid)
        assert store.delete_empty_people() == 1
        assert store.get_person(pid) is None

    def test_sweep_removes_empty_string_name(self, tmp_db):
        pid = store.create_person("", None, None)
        self._backdate(pid)
        assert store.delete_empty_people() == 1
        assert store.get_person(pid) is None

    def test_sweep_is_idempotent(self, tmp_db):
        pid = store.create_person(None, None, None)
        self._backdate(pid)
        assert store.delete_empty_people() == 1
        assert store.delete_empty_people() == 0
        assert store.get_person(pid) is None

    def test_sweep_keeps_recent_or_named_or_with_faces(self, tmp_db):
        self._seed_photo_done("p1")
        recent = store.create_person(None, None, None)          # too young for the age guard
        named = store.create_person("Named", None, None)        # name is set
        fid = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        has_face = store.create_person(None, "p1", fid)
        store.assign_face_person(fid, has_face)
        assert store.delete_empty_people() == 0
        for pid in (recent, named, has_face):
            assert store.get_person(pid) is not None

    def test_unassign_last_face_deletes_unnamed_person(self, tmp_db):
        self._seed_photo_done("p1")
        fid = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        pid = store.create_person(None, "p1", fid)
        store.assign_face_person(fid, pid)
        assert store.get_person(pid)["face_count"] == 1
        store.unassign_face(fid)
        assert store.get_person(pid) is None

    def test_moving_last_face_away_deletes_unnamed_source(self, tmp_db):
        self._seed_photo_done("p1")
        self._seed_photo_done("p2")
        f1 = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        dst = store.create_person("Dst", "p2", None)
        src = store.create_person(None, "p1", f1)
        store.assign_face_person(f1, src)
        store.assign_face_person(f1, dst)  # moves the last face away
        assert store.get_person(src) is None
        assert store.get_person(dst)["face_count"] == 1

    def test_named_person_survives_last_face_move(self, tmp_db):
        self._seed_photo_done("p1")
        f1 = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        pid = store.create_person("Carol", "p1", f1)
        store.assign_face_person(f1, pid)
        store.unassign_face(f1)
        assert store.get_person(pid) is not None  # named people are never auto-deleted


class TestPhotosQuery:
    def test_done_photos_and_filters(self, tmp_db):
        store.upsert_photos(
            [_photo("p1", capture_time=1000), _photo("p2", capture_time=2000), _photo("p3", capture_time=3000)]
        )
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        store.set_photo_done("p3", "t3.webp", None, None)
        uid = store.create_user("alice", "hash", role="read", display_name="Alice")
        store.favorite_photo(uid, "p2")
        rows = store.done_photos(limit=10, user_id=uid)
        assert [r["uid"] for r in rows] == ["p3", "p2", "p1"]
        favs = store.done_photos(limit=10, only_favorites=True, user_id=uid)
        assert [r["uid"] for r in favs] == ["p2"]
        before = store.done_photos(limit=10, before=2500)
        assert [r["uid"] for r in before] == ["p2", "p1"]
        excluded = store.done_photos(limit=10, include_archived=False)
        assert [r["uid"] for r in excluded] == ["p3", "p2", "p1"]

    def test_archived_photos(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2")])
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        store.set_archived("p1", True)
        assert [r["uid"] for r in store.archived_photos()] == ["p1"]

    def test_favorited_archived_hidden_flags(self, tmp_db):
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        uid = store.create_user("bob", "hash")
        assert store.is_favorite(uid, "p1") is False
        assert store.favorite_photo(uid, "p1") is False  # newly added
        assert store.is_favorite(uid, "p1") is True
        assert store.favorite_photo(uid, "p1") is True  # already favorited
        assert store.favorite_uids(uid, ["p1"]) == {"p1"}
        assert store.unfavorite_photo(uid, "p1") is True
        assert store.is_favorite(uid, "p1") is False
        store.set_hidden("p1", True)
        assert store.get_photo("p1")["hidden"] == 1


class TestTags:
    def test_set_and_get_tags(self, tmp_db):
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        assert store.set_tags("p1", ["Dog", "dog", " Beach "]) == ["dog", "beach"]
        assert store.get_tags("p1") == ["dog", "beach"]

    def test_all_tags_and_photos_by_tag(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2", capture_time=2000)])
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        store.set_tags("p1", ["sunset"])
        store.set_tags("p2", ["sunset", "sea"])
        tags = store.all_tags()
        assert {t: n for t, n in tags} == {"sunset": 2, "sea": 1}
        assert [r["uid"] for r in store.photos_by_tag("sunset")] == ["p2", "p1"]

    def test_photos_by_tag_no_false_substring_match(self, tmp_db):
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        store.set_tags("p1", ["cat"])
        assert store.photos_by_tag("car") == []


class TestSargableQueries:
    """Issue #85: filters on tags/albums/memories/anchors must not scan the
    whole photo set. These assert the query planner chooses an index."""

    def _plans(self, sql: str, params: tuple = ()) -> list[str]:
        with store.get_conn() as conn:
            return [r["detail"] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params)]

    def test_tags_join_uses_index(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2")])
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        store.set_tags("p1", ["sunset"])
        store.set_tags("p2", ["sunset"])
        plans = self._plans(
            "SELECT p.* FROM photo_tags pt "
            "CROSS JOIN photos p ON p.uid = pt.photo_uid "
            "WHERE p.status='done' AND p.thumb_path IS NOT NULL AND p.thumb_path != '' "
            "AND p.hidden = 0 AND pt.tag = ? "
            "ORDER BY p.capture_time DESC LIMIT ? OFFSET ?",
            ("sunset", 200, 0),
        )
        assert any("SEARCH pt USING COVERING INDEX idx_photo_tags_tag" in p for p in plans)
        # No full scan of the photos table for the tag match.
        assert not any("SCAN photos" in p or "SCAN photo_tags" in p for p in plans)

    def test_album_join_uses_index(self, tmp_db):
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t1.webp", None, None)
        with store.get_conn() as conn:
            conn.execute("UPDATE photos SET albums=? WHERE uid='p1'", (json.dumps(["al1"]),))
        store.sync_albums([{"uid": "al1", "name": "Trip"}])
        plans = self._plans(
            "SELECT p.* FROM photo_albums pa "
            "CROSS JOIN photos p ON p.uid = pa.photo_uid "
            "WHERE p.status='done' AND p.thumb_path IS NOT NULL AND p.thumb_path != '' "
            "AND pa.album_uid = ? "
            "ORDER BY p.capture_time DESC LIMIT ? OFFSET ?",
            ("al1", 200, 0),
        )
        assert any("SEARCH pa USING COVERING INDEX idx_photo_albums_album" in p for p in plans)
        assert not any("SCAN photos" in p or "SCAN photo_albums" in p for p in plans)

    def test_memories_uses_index(self, tmp_db):
        store.upsert_photos([_photo("p1", capture_time=1609459200)])
        store.set_photo_done("p1", "t.webp", None, None)
        plans = self._plans(
            "SELECT * FROM photos INDEXED BY idx_photos_month_day "
            "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
            "AND hidden = 0 AND strftime('%m-%d', capture_time, 'unixepoch') = ? "
            "AND capture_time IS NOT NULL "
            "ORDER BY capture_time DESC LIMIT ?",
            ("01-01", 200),
        )
        assert any("SEARCH photos USING INDEX idx_photos_month_day" in p for p in plans)

    def test_anchors_uses_index(self, tmp_db):
        store.upsert_photos([_photo("p1", capture_time=1609459200)])
        store.set_photo_done("p1", "t.webp", None, None)
        plans = self._plans(
            "SELECT substr(date(capture_time, 'unixepoch'), 1, 7) AS ym, "
            "MAX(capture_time) AS first_ts FROM photos INDEXED BY idx_photos_ym "
            "WHERE status='done' AND thumb_path IS NOT NULL AND thumb_path != '' "
            "AND capture_time IS NOT NULL "
            "GROUP BY ym ORDER BY ym DESC LIMIT ?",
            (500,),
        )
        assert any("USING INDEX idx_photos_ym" in p for p in plans)


class TestPlacesAndMap:
    def test_search_photos_by_place(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2")])
        store.set_photo_done("p1", "t1.webp", (45.4, 9.2), "Milano")
        store.set_photo_done("p2", "t2.webp", None, None)
        rows = store.search_photos_by_place("mila")
        assert [r["uid"] for r in rows] == ["p1"]

    def test_place_stats_and_map_markers(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2")])
        store.set_photo_done("p1", "t1.webp", (45.4, 9.2), "Milano")
        store.set_photo_done("p2", "t2.webp", (41.9, 12.5), "Roma")
        stats_ = store.place_stats()
        assert {r["place"]: r["photo_count"] for r in stats_} == {"Milano": 1, "Roma": 1}
        markers = store.map_markers()
        assert len(markers) == 2

    def test_place_stats_and_map_markers_q_filter(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2"), _photo("p3")])
        store.set_photo_done("p1", "t1.webp", (45.4, 9.2), "Milano")
        store.set_photo_done("p2", "t2.webp", (41.9, 12.5), "Roma")
        store.set_photo_done("p3", "t3.webp", (48.8, 2.3), "Paris, Île-de-France, France")
        assert [r["place"] for r in store.place_stats(q="mila")] == ["Milano"]
        assert [r["place"] for r in store.place_stats(q="MIL")] == ["Milano"]
        assert [r["place"] for r in store.place_stats(q="paris")] == ["Paris, Île-de-France, France"]
        # substring in the middle of the place name matches too.
        assert [r["place"] for r in store.place_stats(q="de-france")] == ["Paris, Île-de-France, France"]
        assert store.place_stats(q="nope") == []
        assert [r["place"] for r in store.map_markers(q="mila")] == ["Milano"]
        assert [r["place"] for r in store.map_markers(q="de-france")] == ["Paris, Île-de-France, France"]
        assert store.map_markers(q="nope") == []
        assert store.count_places() == 3

    def test_person_map_markers(self, tmp_db):
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t1.webp", (45.4, 9.2), "Milano")
        fid = store.insert_face("p1", None, 0.9, "[]", EMB.tobytes())
        pid = store.create_person("Joe", "p1", fid)
        store.assign_face_person(fid, pid)
        markers = store.person_map_markers(pid)
        assert len(markers) == 1


class TestAlbums:
    def test_sync_albums(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2", capture_time=2000)])
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        with store.get_conn() as conn:
            conn.execute("UPDATE photos SET albums=? WHERE uid='p1'", (json.dumps(["al1"]),))
            conn.execute("UPDATE photos SET albums=? WHERE uid='p2'", (json.dumps(["al1", "al2"]),))
        store.sync_albums(
            [
                {"uid": "al1", "name": "Trip", "cover_uid": "p1"},
                {"uid": "al2", "name": "Empty", "cover_uid": None},
            ]
        )
        albums = store.all_albums()
        by_uid = {a["uid"]: a for a in albums}
        assert by_uid["al1"]["photo_count"] == 2
        assert by_uid["al2"]["photo_count"] == 1
        assert [r["uid"] for r in store.album_photos("al1")] == ["p2", "p1"]

    def test_album_names_resolves_locally(self, tmp_db):
        store.sync_albums([{"uid": "al1", "name": "Trip"}, {"uid": "al2", "name": None}])
        assert store.album_names(["al1", "al2", "missing"]) == {
            "al1": "Trip",
            "al2": "al2",
        }
        assert store.album_names([]) == {}

    def test_all_albums_q_filter(self, tmp_db):
        store.upsert_photos([_photo("p1"), _photo("p2")])
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        with store.get_conn() as conn:
            conn.execute("UPDATE photos SET albums=? WHERE uid='p1'", (json.dumps(["al1"]),))
            conn.execute("UPDATE photos SET albums=? WHERE uid='p2'", (json.dumps(["al2"]),))
        store.sync_albums(
            [
                {"uid": "al1", "name": "Holiday Trip"},
                {"uid": "al2", "name": "Work Notes"},
            ]
        )
        assert [a["uid"] for a in store.all_albums(q="hol")] == ["al1"]
        assert [a["uid"] for a in store.all_albums(q="HOLIDAY")] == ["al1"]
        # substring in the middle of the album name matches too.
        assert [a["uid"] for a in store.all_albums(q="Trip")] == ["al1"]
        assert store.all_albums(q="nope") == []
        assert len(store.all_albums()) == 2

    def _seed_albums(self):
        """Seeds one album each with (p1) and (p1, p2) and a lone photo in al3,
        so al3 is a *counted* album that must never be recomputed by an
        incremental pass touching only al1/al2."""
        store.upsert_photos(
            [
                _photo("p1", capture_time=1000, albums=["al1"]),
                _photo("p2", capture_time=2000, albums=["al1", "al2"]),
                _photo("p3", capture_time=3000, albums=["al3"]),
            ]
        )
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        store.set_photo_done("p3", "t3.webp", None, None)
        store.sync_albums(
            [
                {"uid": "al1", "name": "Trip", "cover_uid": "p1"},
                {"uid": "al2", "name": "Empty", "cover_uid": None},
                {"uid": "al3", "name": "Untouched", "cover_uid": None},
            ]
        )

    @staticmethod
    def _assert_dirty(expect: set[str]) -> None:
        with store.get_conn() as conn:
            dirty = {r[0] for r in conn.execute("SELECT uid FROM albums WHERE dirty=1")}
        assert dirty == expect

    def test_albums_incremental_nochange_does_not_rescan(self, tmp_db):
        """A second consecutive sync with no changes recomputes ~nothing:
        the dirty set is empty, so no album recount runs."""
        self._seed_albums()
        self._assert_dirty(set())
        # No dirty flags, and the periodic full rescan is far away — this pass
        # must be a no-op apart from the name upsert.
        store.sync_albums([{"uid": "al1", "name": "Trip"}])
        self._assert_dirty(set())
        by_uid = {a["uid"]: a for a in store.all_albums()}
        assert by_uid["al1"]["photo_count"] == 2
        assert by_uid["al2"]["photo_count"] == 1
        assert by_uid["al3"]["photo_count"] == 1

    def test_albums_incremental_recounts_only_dirty(self, tmp_db):
        """Moving a photo between albums flags both albums; the next sync
        recounts them and leaves untouched albums alone."""
        self._seed_albums()
        store.upsert_photos([_photo("p1", capture_time=1000, albums=["al2"])])
        self._assert_dirty({"al1", "al2"})
        store.sync_albums([])
        by_uid = {a["uid"]: a for a in store.all_albums()}
        assert by_uid["al1"]["photo_count"] == 1  # p2 only
        assert by_uid["al2"]["photo_count"] == 2  # p1 + p2
        # al3 was never flagged and must not have been recalculated.
        self._assert_dirty(set())
        assert by_uid["al3"]["photo_count"] == 1
        assert [r["uid"] for r in store.album_photos("al1")] == ["p2"]

    def test_albums_incremental_counts_newly_done_photo(self, tmp_db):
        """A photo that becomes done-with-thumb after the last sync is picked
        up without a full rescan (set_photo_done flags its albums)."""
        self._seed_albums()
        store.upsert_photos([_photo("p4", capture_time=4000, albums=["al1"])])
        store.set_photo_done("p4", "t4.webp", None, None)
        self._assert_dirty({"al1"})
        store.sync_albums([])
        by_uid = {a["uid"]: a for a in store.all_albums()}
        assert by_uid["al1"]["photo_count"] == 3
        assert by_uid["al1"]["cover_uid"] == "p4"
        assert [r["uid"] for r in store.album_photos("al1")] == ["p4", "p2", "p1"]

    def test_albums_incremental_deletion_drops_count(self, tmp_db):
        """Photos that leave the counted set (deleted) flag their albums; the
        next sync drops the count without a full rescan."""
        self._seed_albums()
        store.mark_deleted(["p1"])
        self._assert_dirty({"al1"})
        store.sync_albums([])
        by_uid = {a["uid"]: a for a in store.all_albums()}
        assert by_uid["al1"]["photo_count"] == 1  # p2 only
        assert by_uid["al1"]["cover_uid"] == "p2"
        assert by_uid["al2"]["photo_count"] == 1  # untouched
        assert [r["uid"] for r in store.album_photos("al1")] == ["p2"]

    def test_albums_full_rescan_repairs_raw_sql_edits(self, tmp_db, monkeypatch):
        """Edits that bypass the dirty-flag hooks (raw SQL) are reconciled the
        next time the periodic full rescan is due."""
        self._seed_albums()
        with store.get_conn() as conn:
            conn.execute(
                "UPDATE photos SET albums=? WHERE uid='p2'", (json.dumps(["al3"]),)
            )
        # Incremental pass sees no dirty flags and must not recompute p2's old
        # album — counts stay stale, as designed.
        store.sync_albums([])
        by_uid = {a["uid"]: a for a in store.all_albums()}
        assert by_uid["al1"]["photo_count"] == 2
        assert by_uid["al3"]["photo_count"] == 1  # p3 only, still missing p2
        # Force the full rescan to become due; the repair pass reconciles.
        monkeypatch.setattr(config.settings, "albums_full_rescan_sec", 0)
        store.sync_albums([])
        by_uid = {a["uid"]: a for a in store.all_albums()}
        assert by_uid["al1"]["photo_count"] == 1
        assert by_uid["al3"]["photo_count"] == 2  # p2 + p3
        assert [r["uid"] for r in store.album_photos("al3")] == ["p3", "p2"]


class TestDuplicatesAndMemories:
    def test_duplicate_groups(self, tmp_db):
        store.upsert_photos(
            [_photo("p1", sha1="same"), _photo("p2", sha1="same"), _photo("p3", sha1="other")]
        )
        store.set_photo_done("p1", "t1.webp", None, None)
        store.set_photo_done("p2", "t2.webp", None, None)
        store.set_photo_done("p3", "t3.webp", None, None)
        groups = store.duplicate_groups()
        assert len(groups) == 1
        assert [r["uid"] for r in groups[0]] == ["p1", "p2"]

    def test_duplicate_groups_ordered_count_desc_hidden_last(self, tmp_db):
        store.upsert_photos(
            [
                _photo("a1", sha1="two", capture_time=3000),
                _photo("a2", sha1="two", capture_time=1000),
                _photo("b1", sha1="three", capture_time=10),
                _photo("b2", sha1="three", capture_time=2000),
                _photo("b3", sha1="three", capture_time=100),
            ]
        )
        for uid in ("a1", "a2", "b1", "b2", "b3"):
            store.set_photo_done(uid, f"{uid}.webp", None, None)
        store.set_hidden("a2", True)
        groups = store.duplicate_groups()
        # worst offenders first; members hidden-ASC then capture_time-DESC
        assert [[r["uid"] for r in g] for g in groups] == [
            ["b2", "b3", "b1"],
            ["a1", "a2"],
        ]

    def test_duplicate_groups_single_round_trip(self, tmp_db):
        # issue #90: one self-join must fetch every group + member, not one
        # query per group (a 500-group library used to cost ~1000 queries).
        store.upsert_photos(
            [_photo("p1", sha1="same"), _photo("p2", sha1="same"), _photo("p3", sha1="other")]
        )
        for uid in ("p1", "p2", "p3"):
            store.set_photo_done(uid, f"{uid}.webp", None, None)
        statements: list[str] = []
        with store.get_conn() as conn:
            conn.set_trace_callback(statements.append)
            try:
                store.duplicate_groups()
            finally:
                conn.set_trace_callback(None)
        selects = [s for s in statements if s.strip().upper().startswith("SELECT")]
        assert len(selects) == 1

    def test_memories_for_today(self, tmp_db):
        store.upsert_photos([_photo("p1", capture_time=1609459200)])  # 2021-01-01
        store.set_photo_done("p1", "t.webp", None, None)
        rows = store.memories_for_today(1, 1)
        assert [r["uid"] for r in rows] == ["p1"]
        assert store.memories_for_today(6, 15) == []

    def test_photo_anchors(self, tmp_db):
        store.upsert_photos([_photo("p1", capture_time=1609459200)])  # 2021-01-01
        store.set_photo_done("p1", "t.webp", None, None)
        anchors = store.photo_anchors()
        assert anchors[0]["ym"] == "2021-01"
        assert anchors[0]["first_ts"] == 1609459200


class TestUsersAndTokens:
    def test_create_and_get_user(self, tmp_db):
        uid = store.create_user("carol", "hash", role="admin", display_name="Carol")
        user = store.get_user_by_username("CAROL")  # case-insensitive
        assert user["id"] == uid
        assert user["role"] == "admin"
        assert user["password_hash"] == "hash"
        assert store.get_user_by_id(uid)["display_name"] == "Carol"

    def test_list_and_update_user(self, tmp_db):
        uid = store.create_user("dave", "hash", role="read")
        assert [u["username"] for u in store.list_users()] == ["dave"]
        assert store.update_user(uid, display_name="Dave", role="write", disabled=False) is True
        assert store.get_user_by_id(uid)["role"] == "write"
        with pytest.raises(ValueError):
            store.update_user(uid, role="superuser")

    def test_delete_user(self, tmp_db):
        uid = store.create_user("erin", "hash")
        store.delete_user(uid)
        assert store.get_user_by_id(uid) is None

    def test_token_issue_lookup_revoke(self, tmp_db):
        uid = store.create_user("frank", "hash")
        token = store.issue_token(uid, "access", 3600, user_agent="t", ip="1.2.3.4")
        row = store.lookup_token(token)
        assert row["username"] == "frank"
        assert row["kind"] == "access"
        assert store.revoke_token(token) is True
        assert store.lookup_token(token) is None

    def test_revoke_all_tokens(self, tmp_db):
        uid = store.create_user("grace", "hash")
        t1 = store.issue_token(uid, "access", 3600)
        t2 = store.issue_token(uid, "refresh", 86400)
        store.revoke_all_tokens(uid)
        assert store.lookup_token(t1) is None
        assert store.lookup_token(t2) is None

    def test_revoke_all_tokens_except(self, tmp_db):
        uid = store.create_user("grace", "hash")
        keep = store.issue_token(uid, "access", 3600)
        t2 = store.issue_token(uid, "access", 3600)
        t3 = store.issue_token(uid, "refresh", 86400)
        assert store.revoke_all_tokens_except(uid, keep) == 2
        assert store.lookup_token(keep) is not None
        assert store.lookup_token(t2) is None
        assert store.lookup_token(t3) is None

    def test_purge_expired_tokens(self, tmp_db):
        uid = store.create_user("henry", "hash")
        store.issue_token(uid, "access", -1)  # already expired
        store.issue_token(uid, "access", 3600)
        assert store.purge_expired_tokens() == 1

    def test_backfill_legacy_favorites(self, tmp_db):
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        with store.get_conn() as conn:
            conn.execute("UPDATE photos SET favorited=1 WHERE uid='p1'")
        uid = store.create_user("iris", "hash")
        store.backfill_legacy_favorites(uid)
        assert store.is_favorite(uid, "p1") is True
        assert store.get_photo("p1")["favorited"] == 0


class TestTotpStore:
    def test_secret_roundtrip(self, tmp_db):
        uid = store.create_user("jane", "hash")
        assert store.get_totp_secret(uid) is None
        store.set_totp_secret(uid, "ciphertext")
        assert store.get_totp_secret(uid) == "ciphertext"
        store.set_totp_secret(uid, None)
        assert store.get_totp_secret(uid) is None

    def test_enabled_flag(self, tmp_db):
        uid = store.create_user("jane", "hash")
        assert store.totp_enabled(uid) is False
        store.set_totp_enabled(uid, True)
        assert store.totp_enabled(uid) is True
        store.set_totp_enabled(uid, False)
        assert store.totp_enabled(uid) is False

    def test_new_users_default_to_no_2fa(self, tmp_db):
        uid = store.create_user("jane", "hash")
        row = store.get_user_by_id(uid)
        assert row["totp_enabled"] == 0
        assert row["totp_secret_enc"] is None

    def test_pending_2fa_create_consume(self, tmp_db):
        uid = store.create_user("jane", "hash")
        token = store.create_pending_2fa(uid, 300)
        row = store.consume_pending_2fa(token)
        assert row is not None
        assert row["user_id"] == uid
        # Single-use: a second consume returns nothing.
        assert store.consume_pending_2fa(token) is None

    def test_pending_2fa_new_login_invalidates_old(self, tmp_db):
        uid = store.create_user("jane", "hash")
        old = store.create_pending_2fa(uid, 300)
        new = store.create_pending_2fa(uid, 300)
        assert store.consume_pending_2fa(old) is None
        assert store.consume_pending_2fa(new) is not None

    def test_pending_2fa_revoke_for_user(self, tmp_db):
        uid = store.create_user("jane", "hash")
        token = store.create_pending_2fa(uid, 300)
        assert store.revoke_pending_2fa_for_user(uid) == 1
        assert store.consume_pending_2fa(token) is None

    def test_pending_2fa_cascade_on_user_delete(self, tmp_db):
        uid = store.create_user("jane", "hash")
        token = store.create_pending_2fa(uid, 300)
        store.delete_user(uid)
        assert store.consume_pending_2fa(token) is None


class TestClips:
    def test_insert_clip_upsert(self, tmp_db):
        store.upsert_photos([_photo("p1")])
        store.set_photo_done("p1", "t.webp", None, None)
        store.insert_clip("p1", EMB.tobytes())
        assert store.clip_count() == 1
        assert store.clip_exists("p1") is True
        assert len(store.all_clips()) == 1
        store.insert_clip("p1", np.zeros(512, dtype=np.float32).tobytes())  # upsert, no dup
        assert store.clip_count() == 1

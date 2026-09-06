import numpy as np

import cluster
import store


def _emb(index, value=1.0):
    v = np.zeros(512, dtype=np.float32)
    v[index] = value
    return (v / np.linalg.norm(v)).tobytes()


def _seed_person(name, embs):
    uid = f"p-{name}"
    store.upsert_photos(
        [{"uid": uid, "name": uid, "media_type": "image/jpeg", "capture_time": 1}]
    )
    person_id = store.create_person(name, uid)
    for e in embs:
        face_id = store.insert_face(uid, None, 0.9, "[0,0,10,10]", e)
        store.assign_face_person(face_id, person_id)
    return person_id


class TestMatchPerson:
    def test_matches_same_cluster(self, tmp_db):
        _seed_person("alice", [_emb(0)])
        pid = cluster.match_person(_emb(0), threshold=0.45)
        assert pid is not None

    def test_no_match_below_threshold(self, tmp_db):
        _seed_person("alice", [_emb(0)])
        pid = cluster.match_person(_emb(1), threshold=0.99)
        assert pid is None

    def test_no_people_returns_none(self, tmp_db):
        assert cluster.match_person(_emb(0), threshold=0.1) is None

    def test_uses_mean_of_multiple_faces(self, tmp_db):
        _seed_person("bob", [_emb(0), _emb(0, 1.1), _emb(0, 0.9)])
        pid = cluster.match_person(_emb(0), threshold=0.45)
        assert pid is not None

    def test_cache_ignored_in_match(self, tmp_db):
        # match_person must not require a warm cache
        cluster._person_means = None
        _seed_person("carol", [_emb(0)])
        assert cluster.match_person(_emb(0), threshold=0.45) is not None


class TestClusterOnce:
    def test_creates_person_from_labels(self, tmp_db, monkeypatch):
        monkeypatch.setattr(cluster.settings, "min_cluster_size", 3)
        monkeypatch.setattr(cluster.settings, "min_samples", 2)
        uid = "cluster-photo"
        store.upsert_photos(
            [{"uid": uid, "name": uid, "media_type": "image/jpeg", "capture_time": 1}]
        )
        for i in range(4):
            store.insert_face(uid, None, 0.9, f"[0,{i},10,{i+10}]", _emb(i))

        class _FakeHDBSCAN:
            def __init__(self, *a, **k):
                pass

            def fit_predict(self, X):
                return np.array([0, 0, 0, -1])

        monkeypatch.setattr(cluster, "HDBSCAN", _FakeHDBSCAN)
        n = cluster.cluster_once()
        assert n == 1
        people = store.all_people()
        assert len(people) == 1
        pid = people[0]["id"]
        assert store.count_faces_for_person(pid) == 3
        # the outlier remains unassigned
        assert len(store.faces_without_person()) == 1

    def test_skips_small_clusters(self, tmp_db, monkeypatch):
        monkeypatch.setattr(cluster.settings, "min_cluster_size", 3)
        monkeypatch.setattr(cluster.settings, "min_samples", 2)
        uid = "cluster-photo"
        store.upsert_photos(
            [{"uid": uid, "name": uid, "media_type": "image/jpeg", "capture_time": 1}]
        )
        for i in range(4):
            store.insert_face(uid, None, 0.9, f"[0,{i},10,{i+10}]", _emb(i))

        class _FakeHDBSCAN:
            def __init__(self, *a, **k):
                pass

            def fit_predict(self, X):
                # one cluster of 2 faces, below min_cluster_size
                return np.array([0, 0, -1, -1])

        monkeypatch.setattr(cluster, "HDBSCAN", _FakeHDBSCAN)
        assert cluster.cluster_once() == 0
        assert store.all_people() == []

    def test_not_enough_faces(self, tmp_db, monkeypatch):
        monkeypatch.setattr(cluster.settings, "min_cluster_size", 5)
        uid = "lonely"
        store.upsert_photos(
            [{"uid": uid, "name": uid, "media_type": "image/jpeg", "capture_time": 1}]
        )
        store.insert_face(uid, None, 0.9, "[0,0,10,10]", _emb(1))

        def boom(*a, **k):
            raise AssertionError("HDBSCAN must not be called")

        monkeypatch.setattr(cluster, "HDBSCAN", boom)
        assert cluster.cluster_once() == 0

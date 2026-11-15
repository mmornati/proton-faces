import threading
import time

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

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


class TestPersonMeansCacheSWR:
    def _force_expiry(self):
        cluster._person_means_ts -= cluster._PERSON_MEANS_TTL + 1

    def _wait_refresh(self, timeout=5.0):
        deadline = time.time() + timeout
        while cluster._person_means_refreshing:
            if time.time() > deadline:
                raise AssertionError("person-means refresh did not complete")
            time.sleep(0.01)

    def test_first_load_is_synchronous(self, tmp_db):
        _seed_person("sync", [_emb(0)])
        means = cluster._person_means_cached()
        assert cluster._person_means is means
        assert cluster._person_means_refreshing is False

    def test_expired_serves_stale_then_background_refresh(self, tmp_db):
        _seed_person("alice", [_emb(0)])
        stale = cluster._person_means_cached()
        self._force_expiry()
        got = cluster._person_means_cached()
        assert got is stale
        assert cluster._person_means_refreshing is True
        self._wait_refresh()
        # a new person becomes visible after the background refresh
        _seed_person("bob", [_emb(1)])
        self._force_expiry()
        got = cluster._person_means_cached()
        self._wait_refresh()
        pid = cluster.match_person(_emb(1), threshold=0.45)
        assert pid is not None
        assert cluster.match_person(_emb(0), threshold=0.45) != pid

    def test_concurrent_expired_callers_share_one_refresh(self, tmp_db, monkeypatch):
        _seed_person("carol", [_emb(0)])
        cluster._person_means_cached()
        self._force_expiry()

        real_build = cluster._build_person_means
        counter = {"n": 0}
        started = threading.Event()
        release = threading.Event()

        def slow_build():
            counter["n"] += 1
            started.set()
            assert release.wait(5.0)
            return real_build()

        monkeypatch.setattr(cluster, "_build_person_means", slow_build)
        results = []
        threads = []
        for _ in range(5):
            t = threading.Thread(target=lambda: results.append(cluster._person_means_cached()))
            t.start()
            threads.append(t)
        assert started.wait(5.0)
        for t in threads:
            t.join(5.0)
        assert counter["n"] == 1
        release.set()
        self._wait_refresh()
        assert cluster._person_means_refreshing is False


class TestCosineEuclideanEquivalence:
    """HDBSCAN metric='euclidean' on L2-normalized vectors must reproduce the
    old metric='cosine' clustering exactly (d² = 2 − 2·cos for unit norms).

    Uses realistic dense 512-d embeddings (random unit centroids + Gaussian
    noise), not degenerate low-rank layouts where metrics disagree.
    """

    @staticmethod
    def _faces(per=40, sigma=0.1, npeople=4, seed=0):
        rng = np.random.default_rng(seed)
        X = []
        for _ in range(npeople):
            center = rng.normal(size=512)
            center /= np.linalg.norm(center)
            pts = center + rng.normal(0.0, sigma, (per, 512))
            X.append(pts)
        X = np.vstack(X).astype(np.float32)
        # unit-norm embeddings, as ArcFace emits
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def test_identical_labels_between_metrics(self):
        X = self._faces()
        kw = dict(min_cluster_size=3, min_samples=2)
        cosine = HDBSCAN(**kw, metric="cosine").fit_predict(X)
        euclidean = HDBSCAN(**kw, metric="euclidean").fit_predict(X)
        # identical partitions (label-permutation invariant)
        assert adjusted_rand_score(cosine, euclidean) == 1.0
        # same noise assignments on both paths
        assert int((cosine == -1).sum()) == int((euclidean == -1).sum())

    def test_identical_labels_over_sigma_range(self):
        for sigma in (0.05, 0.1, 0.2):
            X = self._faces(sigma=sigma)
            kw = dict(min_cluster_size=3, min_samples=2)
            cosine = HDBSCAN(**kw, metric="cosine").fit_predict(X)
            euclidean = HDBSCAN(**kw, metric="euclidean").fit_predict(X)
            assert adjusted_rand_score(cosine, euclidean) == 1.0
            assert int((cosine == -1).sum()) == int((euclidean == -1).sum())


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



class TestPersonMeansPublish:
    def test_stack_and_dict_come_from_one_generation(self, tmp_db):
        store.upsert_photos([{"uid": "p", "name": "p", "media_type": "image/jpeg", "capture_time": 1}])
        for i in range(3):
            emb = np.zeros(512, dtype=np.float32)
            emb[i] = 1.0
            fid = store.insert_face(photo_uid="p", person_id=None, confidence=0.9, bbox=[0, 0, 1, 1],
                                    embedding=emb.tobytes())
            pid = store.create_person(name=None, cover_uid="p", cover_face_id=fid)
            store.assign_face_person(fid, pid)
        means = cluster._person_means_cached()
        pids, mat = cluster._person_means_stack
        assert len(means) == 3 and len(pids) == mat.shape[0] == 3
        assert set(pids.tolist()) == set(means.keys())
        assert not hasattr(cluster, "_person_means_mat")


class TestLazySklearn:
    def test_importing_cluster_does_not_import_sklearn(self):
        import subprocess
        import sys
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "app" / "src"
        code = "import sys; sys.path.insert(0, %r); import cluster; print('sklearn' in sys.modules)" % str(src)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False"


class TestClusterWindowRotates:
    """Noise faces stay unassigned; the capped window must move past them."""

    def test_cursor_advances_and_wraps(self, tmp_db, monkeypatch):
        import cluster

        seen = []

        def fake(limit=None, after_id=0):
            seen.append(after_id)
            ids = [i for i in range(1, 11) if i > after_id][: limit or 99]
            return [{"id": i, "embedding": np.ones(512, dtype=np.float32).tobytes()} for i in ids]

        class NoiseOnly:
            def __init__(self, **kw):
                pass

            def fit_predict(self, X):
                return np.full(len(X), -1)

        monkeypatch.setattr(cluster, "faces_without_person", fake)
        monkeypatch.setattr(cluster, "HDBSCAN", NoiseOnly)
        monkeypatch.setattr(cluster.settings, "min_cluster_size", 2)
        for _ in range(4):
            cluster.cluster_once(max_faces=4)
        # 1-4 → cursor 3; 4-7 → cursor 6; 7-10 → cursor 9; only face 10 left → wrap.
        assert seen == [0, 3, 6, 9]
        assert cluster._cluster_cursor == 0

    def test_small_backlog_never_moves_the_cursor(self, tmp_db, monkeypatch):
        import cluster

        monkeypatch.setattr(cluster, "faces_without_person", lambda limit=None, after_id=0: [])
        cluster.cluster_once(max_faces=5000)
        assert cluster._cluster_cursor == 0

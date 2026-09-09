import numpy as np
import pytest

import faces


class _FakeFace:
    def __init__(self, bbox, score=0.9, emb=None):
        if emb is None:
            emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        self.bbox = np.array(bbox, dtype=np.float64)
        self.det_score = score
        self.normed_embedding = emb


class _FakeApp:
    def __init__(self, results):
        self._results = list(results)
        self.get_calls = 0

    def get(self, image):
        self.get_calls += 1
        return self._results


@pytest.fixture
def fake_app(monkeypatch):
    def install(results):
        app = _FakeApp(results)
        monkeypatch.setattr(faces, "_load", lambda: app)
        return app

    return install


class TestWarmUp:
    def test_runs_dummy_detection_pass(self, fake_app):
        app = fake_app([])
        faces.warm_up()
        assert app.get_calls == 1

    def test_load_failure_propagates(self, monkeypatch):
        def _boom():
            raise RuntimeError("no model")

        monkeypatch.setattr(faces, "_load", _boom)
        with pytest.raises(RuntimeError):
            faces.warm_up()


class TestDetectFaces:
    def test_empty_image(self, fake_app):
        fake_app([])
        assert faces.detect_faces(np.zeros((10, 10, 3), dtype=np.uint8)) == []

    def test_builds_face_dicts(self, fake_app):
        fake_app([_FakeFace([0, 0, 100, 100])])
        out = faces.detect_faces(np.zeros((10, 10, 3), dtype=np.uint8))
        assert len(out) == 1
        assert out[0]["bbox"] == [0.0, 0.0, 100.0, 100.0]
        assert out[0]["confidence"] == 0.9
        assert isinstance(out[0]["embedding"], np.ndarray)
        assert out[0]["embedding"].dtype == np.float32

    def test_load_failure_propagates(self, fake_app, monkeypatch):
        fake_app([])

        def _boom():
            raise RuntimeError("no model")

        monkeypatch.setattr(faces, "_load", _boom)
        with pytest.raises(RuntimeError):
            faces.detect_faces(np.zeros((10, 10, 3), dtype=np.uint8))


class TestEmbedQueryFace:
    def test_returns_largest_face(self, fake_app):
        small = _FakeFace([0, 0, 40, 40], emb=np.zeros(512, dtype=np.float32))
        large = _FakeFace([0, 0, 200, 200], emb=np.ones(512, dtype=np.float32))
        fake_app([small, large])
        emb = faces.embed_query_face(np.zeros((10, 10, 3), dtype=np.uint8))
        assert emb is not None
        assert (emb == large.normed_embedding).all()

    def test_no_faces_returns_none(self, fake_app):
        fake_app([])
        assert faces.embed_query_face(np.zeros((10, 10, 3), dtype=np.uint8)) is None

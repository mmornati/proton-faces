
import numpy as np
import pytest
from PIL import Image

import clip


def _rgb_image(size=(64, 64)):
    return Image.fromarray(
        np.full((size[1], size[0], 3), 128, dtype=np.uint8), mode="RGB"
    )


class TestPreprocess:
    def test_shape_and_dtype(self):
        out = clip._preprocess(_rgb_image())
        assert out.shape == (1, 3, 224, 224)
        assert out.dtype == np.float32

    def test_normalization(self):
        out = clip._preprocess(_rgb_image())
        # (128/255 - mean)/std applied per-channel
        expected = (128 / 255.0 - clip._MEAN) / clip._STD
        assert np.allclose(out[0, 0, 0, 0], expected[0], atol=1e-3)
        assert np.allclose(out[0, 1, 0, 0], expected[1], atol=1e-3)
        assert np.allclose(out[0, 2, 0, 0], expected[2], atol=1e-3)

    def test_resizes_larger_images(self):
        out = clip._preprocess(_rgb_image((400, 300)))
        assert out.shape == (1, 3, 224, 224)

    def test_handles_rgba_and_palette(self):
        rgba = Image.fromarray(np.full((10, 10, 4), 128, dtype=np.uint8), mode="RGBA")
        assert clip._preprocess(rgba).shape == (1, 3, 224, 224)
        pal = Image.new("P", (10, 10))
        assert clip._preprocess(pal).shape == (1, 3, 224, 224)


class TestL2Norm:
    def test_normalizes(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        out = clip._l2norm(v)
        assert np.isclose(np.linalg.norm(out), 1.0)
        assert np.allclose(out, [0.6, 0.8])

    def test_zero_vector(self):
        out = clip._l2norm(np.zeros(4, dtype=np.float32))
        assert not np.any(np.isnan(out))
        assert np.linalg.norm(out) == 0.0


class TestSessionOptions:
    def test_defaults_to_one_thread(self, monkeypatch):
        monkeypatch.delenv("ORT_INTRA_OP_THREADS", raising=False)
        so = clip._session_options(_FakeOrt())
        assert so.intra_op_num_threads == 1

    def test_reads_env_override(self, monkeypatch):
        monkeypatch.setenv("ORT_INTRA_OP_THREADS", "4")
        so = clip._session_options(_FakeOrt())
        assert so.intra_op_num_threads == 4

    def test_explicit_threads_wins(self, monkeypatch):
        monkeypatch.setenv("ORT_INTRA_OP_THREADS", "4")
        so = clip._session_options(_FakeOrt(), threads=2)
        assert so.intra_op_num_threads == 2

    def test_invalid_env_falls_back_to_one(self, monkeypatch):
        monkeypatch.setenv("ORT_INTRA_OP_THREADS", "lots")
        so = clip._session_options(_FakeOrt())
        assert so.intra_op_num_threads == 1

    def test_zero_or_negative_clamps_to_one(self, monkeypatch):
        monkeypatch.setenv("ORT_INTRA_OP_THREADS", "0")
        so = clip._session_options(_FakeOrt())
        assert so.intra_op_num_threads == 1


class _FakeOrt:
    class SessionOptions:
        def __init__(self):
            self.intra_op_num_threads = 0


class _FakeSession:
    def __init__(self, output):
        self._output = output
        self.run_calls = []

    def run(self, feeds, inputs):
        self.run_calls.append(inputs)
        return [self._output]


class _FakeTokenizer:
    class _Encoded:
        def __init__(self):
            self.ids = [49406, 1, 2, 3, 49407]

    def encode(self, text):
        return self._Encoded()


class TestWarmUp:
    def test_runs_dummy_vision_and_text(self, monkeypatch):
        vec = np.ones((1, 512), dtype=np.float32)
        vis = _FakeSession(vec)
        txt = _FakeSession(vec)
        monkeypatch.setattr(clip, "_load", lambda: (vis, txt, _FakeTokenizer()))
        clip.warm_up()
        assert "pixel_values" in vis.run_calls[0]
        assert vis.run_calls[0]["pixel_values"].shape == (1, 3, 224, 224)
        assert "input_ids" in txt.run_calls[0]
        assert txt.run_calls[0]["input_ids"].shape == (1, len(_FakeTokenizer()._Encoded().ids))

    def test_load_failure_propagates(self, monkeypatch):
        def _boom():
            raise RuntimeError("no model")

        monkeypatch.setattr(clip, "_load", _boom)
        with pytest.raises(RuntimeError):
            clip.warm_up()


class TestEmbed:
    @pytest.fixture
    def fake_sessions(self, monkeypatch):
        vec = np.ones((1, 512), dtype=np.float32) * 3.0
        vis = _FakeSession(vec)
        txt = _FakeSession(vec)
        monkeypatch.setattr(clip, "_load", lambda: (vis, txt, _FakeTokenizer()))
        return vis, txt

    def test_embed_image_returns_normalized(self, fake_sessions, tmp_path):
        p = tmp_path / "a.jpg"
        _rgb_image().save(p, format="JPEG")
        out = clip.embed_image(str(p))
        assert out is not None
        assert out.shape == (512,)
        assert out.dtype == np.float32
        assert np.isclose(np.linalg.norm(out), 1.0)

    def test_embed_pil_feeds_pixel_values(self, fake_sessions):
        vis, _ = fake_sessions
        out = clip.embed_pil(_rgb_image())
        assert out is not None
        assert np.isclose(np.linalg.norm(out), 1.0)
        assert "pixel_values" in vis.run_calls[0]
        assert vis.run_calls[0]["pixel_values"].shape == (1, 3, 224, 224)

    def test_embed_text_feeds_input_ids(self, fake_sessions):
        _, txt = fake_sessions
        out = clip.embed_text("a beach")
        assert out is not None
        assert np.isclose(np.linalg.norm(out), 1.0)
        assert "input_ids" in txt.run_calls[0]

    def test_missing_file_returns_none(self, fake_sessions, tmp_path):
        assert clip.embed_image(str(tmp_path / "nope.jpg")) is None


class _MeanSession:
    """Fake vision session: output depends on pixel contents per row.

    Each row's embedding is the flattened-mean of its normalized pixels
    tiled to 512-d, so two different images produce distinguishable vectors
    and batch output == stacked per-image output.
    """

    def __init__(self):
        self.run_calls = []

    def run(self, feeds, inputs):
        self.run_calls.append(inputs)
        pv = inputs["pixel_values"]
        means = pv.mean(axis=(1, 2, 3))  # (N,)
        return [np.tile(means[:, None], (1, 512))]


class TestEmbedBatch:
    @pytest.fixture
    def fake_sessions(self, monkeypatch):
        vis = _MeanSession()
        txt = _FakeSession(np.ones((1, 512), dtype=np.float32) * 3.0)
        monkeypatch.setattr(clip, "_load", lambda: (vis, txt, _FakeTokenizer()))
        return vis, txt

    def test_empty_returns_none(self, fake_sessions):
        assert clip.embed_batch([]) is None

    def test_stacked_tensor_single_session_run(self, fake_sessions):
        vis, _ = fake_sessions
        out = clip.embed_batch([_rgb_image(), _rgb_image(), _rgb_image()])
        assert out is not None
        assert out.shape == (3, 512)
        assert out.dtype == np.float32
        assert len(vis.run_calls) == 1
        feats = vis.run_calls[0]["pixel_values"]
        assert feats.shape == (3, 3, 224, 224)

    def test_rows_normalized(self, fake_sessions):
        out = clip.embed_batch([_rgb_image(), _rgb_image(size=(100, 80))])
        assert np.allclose(np.linalg.norm(out, axis=-1), 1.0)

    def test_matches_embed_pil_per_image(self, fake_sessions):
        a = _rgb_image((96, 64))
        b = _rgb_image((48, 128))
        batch = clip.embed_batch([a, b])
        assert batch is not None
        assert np.allclose(batch[0], clip.embed_pil(a), atol=1e-5)
        assert np.allclose(batch[1], clip.embed_pil(b), atol=1e-5)

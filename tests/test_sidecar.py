"""Sidecar atomicity + length guards (audit 2026-09, B-3)."""
import json

import numpy as np

import sidecar


def _face_inputs(n):
    ids = list(range(1, n + 1))
    uids = [f"u{i}" for i in ids]
    pids = [None if i % 2 else i for i in ids]
    mat = np.random.default_rng(0).random((n, 512), dtype=np.float32)
    return ids, uids, pids, mat


class TestFaceSidecar:
    def test_roundtrip(self, tmp_path):
        sidecar._set_sidecar_dir(tmp_path)
        ids, uids, pids, mat = _face_inputs(5)
        sidecar.write_face_sidecar(ids, uids, pids, mat)
        data = sidecar.read_face_sidecar()
        assert data is not None
        assert data["ids"].tolist() == ids
        assert data["photo_uids"] == uids
        assert data["person_ids"] == pids
        assert np.allclose(np.asarray(data["mat"]), mat)
        # No staging temps left behind.
        assert not [p for p in tmp_path.iterdir() if ".tmp" in p.name]

    def test_write_rejects_mismatched_inputs(self, tmp_path):
        sidecar._set_sidecar_dir(tmp_path)
        ids, uids, pids, mat = _face_inputs(4)
        import pytest
        with pytest.raises(ValueError):
            sidecar.write_face_sidecar(ids[:3], uids, pids, mat)

    def test_read_refuses_meta_matrix_mismatch(self, tmp_path):
        sidecar._set_sidecar_dir(tmp_path)
        ids, uids, pids, mat = _face_inputs(4)
        sidecar.write_face_sidecar(ids, uids, pids, mat)
        # Simulate a torn generation: meta advertises 5 rows, matrix has 4.
        meta_path = tmp_path / sidecar._FACE_META_PATH
        meta = json.loads(meta_path.read_text())
        meta["ids"].append(99)
        meta["person_ids"].append(None)
        meta_path.write_text(json.dumps(meta))
        sidecar.invalidate_face_cache()
        assert sidecar.read_face_sidecar() is None

    def test_meta_is_renamed_last(self, tmp_path, monkeypatch):
        sidecar._set_sidecar_dir(tmp_path)
        ids, uids, pids, mat = _face_inputs(3)
        order = []
        real_replace = sidecar.os.replace

        def spy(src, dst):
            order.append(str(dst).rsplit("/", 1)[-1])
            return real_replace(src, dst)

        monkeypatch.setattr(sidecar.os, "replace", spy)
        sidecar.write_face_sidecar(ids, uids, pids, mat)
        assert order[-1] == sidecar._FACE_META_PATH
        assert set(order[:-1]) == {sidecar._FACE_MAT_PATH, sidecar._FACE_UIDS_PATH}


class TestClipSidecar:
    def test_roundtrip_and_mismatch(self, tmp_path):
        sidecar._set_sidecar_dir(tmp_path)
        X = np.ones((3, 512), dtype=np.float32)
        sidecar.write_clip_sidecar(["a", "b", "c"], X)
        uids, got = sidecar.read_clip_sidecar()
        assert uids == ["a", "b", "c"] and got.shape == (3, 512)
        meta_path = tmp_path / sidecar._CLIP_META_PATH
        meta = json.loads(meta_path.read_text())
        meta["uids"].append("d")
        meta_path.write_text(json.dumps(meta))
        sidecar._clip_mmap = None
        assert sidecar.read_clip_sidecar() is None
        import pytest
        with pytest.raises(ValueError):
            sidecar.write_clip_sidecar(["a"], X)

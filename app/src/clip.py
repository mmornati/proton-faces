"""CLIP image + text embeddings for object/scene semantic search.

Zero-shot text queries ("dog", "car", "Lille cathedral") are matched against
per-photo CLIP embeddings stored in the clips table.

Runs OpenAI CLIP ViT-B-32 through ONNX Runtime (no PyTorch dependency).
Models: Xenova/clip-vit-base-patch32 export (vision_model.onnx +
text_model.onnx + tokenizer.json), baked into the image at /models/clip.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np
from PIL import Image, ImageOps

from config import settings

log = logging.getLogger("clip")

# CLIP preprocessing constants (transformers CLIPFeatureExtractor defaults).
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_SIZE = 224
_SEQ = 77  # CLIP context length
_PAD_ID = 49407  # <|endoftext|>
_BOS_ID = 49406  # <|startoftext|>

_lock = threading.Lock()
_sess_vision = None
_sess_text = None
_tokenizer = None


def _model_dir():
    return settings.models_dir / "clip"


def _session_options(ort, threads: int | None = None):
    """Build ORT SessionOptions with a bounded intra-op thread pool.

    Each InferenceSession otherwise spawns one thread per physical core,
    so N sessions on a multi-core box oversubscribe the CPU. Default 1
    thread keeps the indexer's WORKERS + CLIP sessions from fighting for
    cores; override with ORT_INTRA_OP_THREADS.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads if threads is not None else _intra_op_threads()
    return so


def _intra_op_threads() -> int:
    raw = os.environ.get("ORT_INTRA_OP_THREADS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("ORT_INTRA_OP_THREADS=%r is not an int; using 1", raw)
        return 1


def _load():
    global _sess_vision, _sess_text, _tokenizer
    if _sess_vision is not None:
        return _sess_vision, _sess_text, _tokenizer
    with _lock:
        if _sess_vision is not None:
            return _sess_vision, _sess_text, _tokenizer
        import onnxruntime as ort

        clip_dir = _model_dir()
        vision_path = clip_dir / "vision_model.onnx"
        text_path = clip_dir / "text_model.onnx"
        tokenizer_path = clip_dir / "tokenizer.json"

        _sess_vision = ort.InferenceSession(
            str(vision_path),
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        _sess_text = ort.InferenceSession(
            str(text_path),
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )

        from tokenizers import Tokenizer

        _tokenizer = Tokenizer.from_file(str(tokenizer_path))
        _tokenizer.enable_truncation(max_length=_SEQ)
        _tokenizer.enable_padding(
            length=_SEQ, pad_id=_PAD_ID, pad_token="<|endoftext|>"
        )
        log.info("CLIP ONNX loaded (%s, %s)", vision_path.name, text_path.name)
        return _sess_vision, _sess_text, _tokenizer


def _preprocess(img: Image.Image) -> np.ndarray:
    """RGB image -> normalized [1, 3, 224, 224] float32 tensor."""
    img = img.convert("RGB")
    img = ImageOps.fit(
        img, (_SIZE, _SIZE), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5)
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, ...]


def _l2norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


def embed_image(path: str) -> np.ndarray | None:
    """Embed a local image file into a normalized 512-d CLIP vector."""
    sess, _, _ = _load()
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:  # pragma: no cover
        log.warning("could not open %s: %s", path, exc)
        return None
    return embed_pil(img)


def embed_batch(images: list[Image.Image]) -> np.ndarray | None:
    """Normalized (N, 512) CLIP vectors for N images in a single session.run.

    ONNX CPU inference batches cheaply: with intra_op=1 the matmul reuses
    per-core work across the batch, so the indexer embeds several photos per
    session call instead of paying the per-call overhead N times (issue #98).
    """
    if not images:
        return None
    sess, _, _ = _load()
    pixel_values = np.concatenate([_preprocess(img) for img in images], axis=0)  # (N,3,224,224)
    out = sess.run(None, {"pixel_values": pixel_values})[0]  # image_embeds (N,512)
    return _l2norm(out).astype(np.float32)


def embed_pil(img: Image.Image) -> np.ndarray | None:
    out = embed_batch([img])
    return None if out is None else out[0]


def embed_text(text: str) -> np.ndarray:
    """Embed a free-text query into a normalized 512-d CLIP vector."""
    _, sess, tokenizer = _load()
    ids = tokenizer.encode(text).ids
    input_ids = np.array([ids], dtype=np.int64)
    out = sess.run(None, {"input_ids": input_ids})[0]  # text_embeds
    return _l2norm(out).astype(np.float32)[0]

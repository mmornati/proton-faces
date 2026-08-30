"""CLIP image + text embeddings for object/scene semantic search.

Zero-shot text queries ("dog", "car", "Lille cathedral") are matched against
per-photo CLIP embeddings stored in the clips table.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

from config import settings

log = logging.getLogger("clip")

_lock = threading.Lock()
_model = None
_transform = None
_tokenizer = None


def _load():
    global _model, _transform, _tokenizer
    if _model is not None:
        return _model, _transform, _tokenizer
    with _lock:
        if _model is not None:
            return _model, _transform, _tokenizer
        import open_clip

        model_name = "ViT-B-32"
        pretrained = "openai"
        model, _, transform = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=str(settings.models_dir / "clip"),
        )
        model.eval()
        _model = model
        _transform = transform
        _tokenizer = open_clip.get_tokenizer(model_name)
        log.info("CLIP %s/%s loaded", model_name, pretrained)
        return model, transform, _tokenizer


def embed_image(path: str) -> np.ndarray | None:
    """Embed a local image file into a normalized 512-d CLIP vector."""
    model, transform, _ = _load()
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:  # pragma: no cover
        log.warning("could not open %s: %s", path, exc)
        return None
    return embed_pil(img)


def embed_pil(img: Image.Image) -> np.ndarray | None:
    model, transform, _ = _load()
    import torch

    with torch.no_grad():
        tensor = transform(img).unsqueeze(0)
        vec = model.encode_image(tensor)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.squeeze(0).numpy().astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """Embed a free-text query into a normalized 512-d CLIP vector."""
    model, _, tokenizer = _load()
    import torch

    with torch.no_grad():
        vec = model.encode_text(tokenizer([text]))
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.squeeze(0).numpy().astype(np.float32)
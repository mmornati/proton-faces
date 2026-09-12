# proton-faces app image.
#
# ML model weights are baked in at build time so the container has no
# runtime download step: InsightFace buffalo_l + OpenCLIP ViT-B-32.
#
# Python 3.11 + numpy<1.24: insightface 0.7.3 relies on aliases (np.bool /
# np.float) removed in numpy 1.24+ and has no cp312 wheels for older numpy.

FROM python:3.11-slim

# System deps: onnxruntime wheels need libgomp1; build-essential is required to
# compile insightface's Cython extension; curl downloads the baked CLIP models;
# unzip extracts the pinned buffalo_l release asset. ffmpeg extracts video
# poster frames + reads durations via ffprobe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 ffmpeg build-essential curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-core.txt requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- bake in ML models ---
# Models live at /models in the image (MODELS_DIR env below); the data volume
# holds only thumbnails + SQLite, not model weights.
#
# Supply-chain hardening (issue #50): every downloaded artifact is pinned to an
# immutable revision (HF commit SHA / GitHub release asset) and verified against
# the checksums in models.sha256 via `sha256sum -c` below. A checksum mismatch
# fails the build, so a tampered or swapped asset can never be baked in.
COPY models.sha256 ./

# CLIP ViT-B-32 (OpenAI weights) via ONNX — Xenova export of
# openai/clip-vit-base-patch32. Same weights as the open_clip 'openai'
# pretrained set, so existing CLIP embeddings stay in the same space.
# Pinned to commit d15189d7028b43f1d3e65039190477f6af591c2a (not mutable
# `main`); verified against models.sha256.
# Download once at build time; baked into /models/clip.
RUN mkdir -p /models/clip \
    && curl -fsSL -o /models/clip/vision_model.onnx \
        https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d7028b43f1d3e65039190477f6af591c2a/onnx/vision_model.onnx \
    && curl -fsSL -o /models/clip/text_model.onnx \
        https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d7028b43f1d3e65039190477f6af591c2a/onnx/text_model.onnx \
    && curl -fsSL -o /models/clip/tokenizer.json \
        https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d7028b43f1d3e65039190477f6af591c2a/tokenizer.json \
    && python - <<'PY'
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
# warm up both sessions with a dummy input
vis = ort.InferenceSession("/models/clip/vision_model.onnx", providers=["CPUExecutionProvider"])
txt = ort.InferenceSession("/models/clip/text_model.onnx", providers=["CPUExecutionProvider"])
tok = Tokenizer.from_file("/models/clip/tokenizer.json")
tok.enable_truncation(max_length=77)
tok.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
pix = np.zeros((1, 3, 224, 224), dtype=np.float32)
_ = vis.run(None, {"pixel_values": pix})[0]
ids = np.array([tok.encode("a beach").ids], dtype=np.int64)
_ = txt.run(None, {"input_ids": ids})[0]
print("clip onnx models baked")
PY

# InsightFace buffalo_l. The upstream `ensure_available` auto-download points at
# an http://d1gsb2o3ihr2l5.cloudfront.net URL that no longer resolves (DNS
# failure), and is not checksum-verifiable. Instead we pull the immutable
# v0.7 GitHub release asset, verify it against models.sha256, and unzip it into
# the exact layout FaceAnalysis expects: /models/insightface/models/buffalo_l/.
# Pre-placing the ONNX files there makes `ensure_available` return early, so the
# runtime never attempts its own (non-verifiable) download.
ENV INSIGHTFACE_HOME=/models/insightface
RUN mkdir -p /models/insightface/models/buffalo_l \
    && curl -fsSL -o /models/insightface/models/buffalo_l.zip \
        https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

# Verify every baked model asset against the pinned checksums. Runs after all
# downloads; a mismatch aborts the build (issue #50). The checksum file uses
# paths relative to /models, so verify from there.
RUN cd /models && sha256sum -c /app/models.sha256

# Extract buffalo_l into the layout FaceAnalysis expects, then drop the zip.
RUN unzip -o /models/insightface/models/buffalo_l.zip -d /models/insightface/models/buffalo_l/ \
    && rm /models/insightface/models/buffalo_l.zip

COPY src ./src
WORKDIR /app/src

ENV DATA_DIR=/data
ENV MODELS_DIR=/models
ENV BRIDGE_URL=http://proton-bridge:8090
ENV PORT=8080

EXPOSE 8080

USER 1000:1000

CMD ["python", "main.py"]
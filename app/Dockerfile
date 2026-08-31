# proton-faces app image.
#
# ML model weights are baked in at build time so the container has no
# runtime download step: InsightFace buffalo_l + OpenCLIP ViT-B-32.
#
# Python 3.11 + numpy<1.24: insightface 0.7.3 relies on aliases (np.bool /
# np.float) removed in numpy 1.24+ and has no cp312 wheels for older numpy.

FROM python:3.11-slim

# System deps: onnxruntime wheels need libgomp1; build-essential is required to
# compile insightface's Cython extension; curl downloads the baked CLIP models.
# ffmpeg extracts video poster frames + reads durations via ffprobe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 ffmpeg build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- bake in ML models ---
# Models live at /models in the image (MODELS_DIR env below); the data volume
# holds only thumbnails + SQLite, not model weights.
# InsightFace buffalo_l.
ENV INSIGHTFACE_HOME=/models/insightface
RUN python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["INSIGHTFACE_HOME"])
root.mkdir(parents=True, exist_ok=True)
from insightface.app import FaceAnalysis
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
    root=str(root),
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("insightface model baked at", root)
PY

# CLIP ViT-B-32 (OpenAI weights) via ONNX — Xenova export of
# openai/clip-vit-base-patch32. Same weights as the open_clip 'openai'
# pretrained set, so existing CLIP embeddings stay in the same space.
# Download once at build time; baked into /models/clip.
RUN mkdir -p /models/clip \
    && curl -fsSL -o /models/clip/vision_model.onnx \
        https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx \
    && curl -fsSL -o /models/clip/text_model.onnx \
        https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx \
    && curl -fsSL -o /models/clip/tokenizer.json \
        https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json \
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

COPY src ./src
WORKDIR /app/src

ENV DATA_DIR=/data
ENV MODELS_DIR=/models
ENV BRIDGE_URL=http://proton-bridge:8090
ENV PORT=8080

VOLUME /data
EXPOSE 8080

USER 1000:1000

CMD ["python", "main.py"]
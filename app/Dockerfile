# proton-faces app image.
#
# ML model weights are baked in at build time so the container has no
# runtime download step: InsightFace buffalo_l + OpenCLIP ViT-B-32.
#
# Python 3.11 + numpy<1.24: insightface 0.7.3 relies on aliases (np.bool /
# np.float) removed in numpy 1.24+ and has no cp312 wheels for older numpy.

FROM python:3.11-slim

# System deps: onnxruntime wheels need libgomp1; open_clip/torch bundles most.
# build-essential is required to compile insightface's Cython extension.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install PyTorch CPU wheels first (default PyPI torch drags in ~2GB of CUDA).
RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cpu
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

# OpenCLIP ViT-B-32 (openai) — force download + warm into /models/clip cache.
RUN python - <<'PY'
import open_clip, torch, PIL.Image
model, _, transform = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", cache_dir="/models/clip"
)
model.eval()
with torch.no_grad():
    _ = model.encode_image(transform(PIL.Image.new("RGB", (32, 32))).unsqueeze(0))
print("openclip model baked")
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
# Getting started

Proton Faces runs as three Docker containers: a Proton bridge that talks to your encrypted Drive, an indexer that does the recognition work, and a FastAPI app that serves the web UI. This section walks through installation, your first run, and a tour of the main features.

## Choose your path

<div class="pf-cards" markdown>

<div class="pf-card" markdown>
### 🚀 [Quickstart](quickstart.md)
Five minutes from `docker compose up` to searching for *dog*. Read this first.
</div>

<div class="pf-card" markdown>
### 🔧 [Installation](installation.md)
All the knobs: `.env`, volumes, single-process mode, local dev, GPU-free hardware requirements.
</div>

<div class="pf-card" markdown>
### 🎬 [Demo mode](demo-mode.md)
Run the full app with **zero Proton credentials**. Bundled CC0 photos, auto-created admin user, no session file needed.
</div>

<div class="pf-card" markdown>
### 🔐 [Session file](session-export.md)
How to get the Proton Drive SDK auth session and keep it safe. Required for the real install.
</div>

</div>

## What you'll need

| | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 6+ cores (face detection is CPU-bound) |
| RAM | 4 GB | 8 GB (InsightFace buffalo_l + CLIP loaded once) |
| Disk | 2 GB + `~1 KB/photo` of thumbs + a few GB of SQLite | SSD; the data volume grows with your library |
| Docker | 20.10+ | Compose v2 |
| OS | Linux, macOS, Windows (with WSL2) | Linux for production |

No GPU is required — InsightFace and CLIP both run on CPU through ONNX Runtime.

## Typical time investment

| Step | Time |
|---|---|
| Pull images and bring up containers | 30–60 s |
| Export a Proton session file | 1 min |
| First sync against a 1000-photo library | ~5 min |
| First sync against a 100 000-photo library | ~1 day (background, resumable) |
| Reading the docs (skim) | 15 min |

The first sync is fully resumable: add photos to Proton and the indexer catches up automatically.

---

**Next:** read the [Quickstart](quickstart.md) to see end-to-end usage with screenshots.

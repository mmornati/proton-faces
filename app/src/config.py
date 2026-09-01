"""Environment configuration for proton-faces."""
from __future__ import annotations

import os
from pathlib import Path


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


def is_demo_mode() -> bool:
    """True when DEMO_MODE is enabled — see demo.py."""
    return _env_bool("DEMO_MODE", False)


def demo_hardening_mode() -> bool:
    """Aggregate switch for public-demo safety.

    When DEMO_HARDENING_MODE=1 (or implied by DEMO_MODE in a public-deploy
    context), flip every "safe for public demo" flag to its safe value:

      DEMO_ALLOW_PUBLIC_THUMBS=0       (require signed URLs for /thumb /full /cover /crop)
      DEMO_DISABLE_BACKUPS=1           (404 /api/admin/backup*)
      DEMO_DISABLE_ADMIN_USER_MGMT=1   (hide /api/admin/users from /docs)
      DEMO_LOGIN_LOGS=0                (don't log demo admin password)

    Individual flags still take precedence — set them explicitly to override.
    """
    if _env_bool("DEMO_HARDENING_MODE", False):
        return True
    # In demo mode, default to hardened for any non-localhost deployment.
    # Operators that want the legacy behavior can set DEMO_HARDENING_MODE=0.
    return is_demo_mode()


class Settings:
    def __init__(self) -> None:
        self.data_dir = Path(os.environ.get("DATA_DIR", "./data")).resolve()
        self.photos_dir = os.environ.get("PHOTOS_DIR", "")  # optional local Takeout export
        self.bridge_url = os.environ.get("BRIDGE_URL", "http://proton-bridge:8090")
        self.port = int(os.environ.get("PORT", "8080"))
        self.sync_interval = int(os.environ.get("SYNC_INTERVAL", "300"))
        self.sync_limit = int(os.environ.get("SYNC_LIMIT", "0"))  # 0 = all photos
        self.workers = int(os.environ.get("WORKERS", "2"))
        self.cluster_interval = int(os.environ.get("CLUSTER_INTERVAL", "1800"))
        self.gps_interval = int(os.environ.get("GPS_INTERVAL", "21600"))  # 6h
        self.face_sim_threshold = float(os.environ.get("FACE_SIM_THRESHOLD", "0.45"))
        self.min_cluster_size = int(os.environ.get("MIN_CLUSTER_SIZE", "2"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")

        # Internal-only endpoint that the `indexer` container exposes on
        # 127.0.0.1 so the `app` container can read its live runtime state
        # (last_sync, live queue depth, thread liveness). Defaults to the
        # compose-network DNS name `indexer` on port 8091; can be overridden
        # for local single-process dev (`http://127.0.0.1:8091`).
        self.indexer_status_port = int(os.environ.get("INDEXER_STATUS_PORT", "8091"))
        self.indexer_status_url = os.environ.get(
            "INDEXER_STATUS_URL", f"http://indexer:{self.indexer_status_port}"
        )

        # Derived paths
        self.work_dir = self.data_dir / "work"
        self.thumb_dir = self.data_dir / "thumbs"
        self.crops_dir = self.data_dir / "crops"
        self.backup_dir = self.data_dir / "_backups"
        self.models_dir = Path(os.environ.get("MODELS_DIR", str(self.data_dir / "models"))).resolve()
        self.db_path = self.data_dir / "index.sqlite3"

        for d in (self.data_dir, self.work_dir, self.thumb_dir, self.crops_dir, self.backup_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)

    # Thread-safety knobs
    @property
    def thumbnails_batch(self) -> int:
        # The Proton API batches at most 30 thumbnail IDs per request.
        return 30


settings = Settings()
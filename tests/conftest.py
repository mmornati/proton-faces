import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
APP_SRC = TESTS_DIR.parent / "app" / "src"

# config.Settings() reads env at import time and computes db_path/data_dir once.
# Point them at a throwaway session dir BEFORE anything imports config.
_SESSION_DATA = Path(tempfile.mkdtemp(prefix="proton-faces-test-"))
os.environ["DATA_DIR"] = str(_SESSION_DATA)
os.environ["MODELS_DIR"] = str(_SESSION_DATA / "models")
# Non-demo runs fail closed without SIGNING_SECRET (see auth._signing_secret);
# give the suite a stable test secret so signed-URL tests exercise the
# explicit-secret path.
os.environ["SIGNING_SECRET"] = "test-signing-secret-0123456789abcdef0123456789abcdef"

if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

import auth  # noqa: E402
import bridge_client  # noqa: E402
import cluster  # noqa: E402
import config  # noqa: E402
import faces  # noqa: E402
import geocode  # noqa: E402
import store  # noqa: E402


@pytest.fixture
def app_settings(monkeypatch, tmp_path):
    """Point every settings path (incl. db_path, computed once at import) at tmp_path."""
    data_dir = tmp_path / "data"
    dirs = {
        "data_dir": data_dir,
        "work_dir": data_dir / "work",
        "thumb_dir": data_dir / "thumbs",
        "crops_dir": data_dir / "crops",
        "backup_dir": data_dir / "_backups",
        "models_dir": data_dir / "models",
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config.settings, name, path)
    db_path = data_dir / "index.sqlite3"
    monkeypatch.setattr(config.settings, "db_path", db_path)
    return config.settings


@pytest.fixture
def tmp_db(app_settings):
    """Fresh initialised database at tmp_path. Mutates config.settings.db_path."""
    store.init_db()
    return app_settings.db_path


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Clear module-level caches/singletons after every test."""
    yield
    store._embedding_cache = None
    store._embedding_cache_ts = 0.0
    store._embedding_cache_refreshing = False
    store._person_means_cache = None
    store._person_means_cache_ts = 0.0
    store._close_local_conns()
    auth._signing_secret.__dict__.pop("_ephemeral", None)
    auth._login_attempts = {}
    cluster._person_means = None
    cluster._person_means_pids = None
    cluster._person_means_mat = None
    cluster._person_means_ts = 0.0
    bridge_client._bridge = None
    bridge_client._async_client = None
    faces._app = None
    geocode._rg = None
    try:
        import clip

        clip._sess_vision = None
        clip._sess_text = None
        clip._tokenizer = None
    except ImportError:
        pass
    try:
        import api

        api._dups_cache = None
        api._anchors_cache = None
        api._people_cache = None
        api._stats_cache = None
        api._dirsize_cache = {}
        api._clip_cache = None
        api._bridge_health_cache = None
        api._indexer_proxy_cache = None
        api._full_semaphore = asyncio.Semaphore(api._FULL_SEMAPHORE_MAX)
        api._full_res_failure_ts.clear()
    except ImportError:
        pass
    try:
        import sidecar
        sidecar.reset_state()
    except ImportError:
        pass

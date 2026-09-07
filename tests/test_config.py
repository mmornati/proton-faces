

import config


def test_env_bool_true_values(monkeypatch):
    for val in ("1", "true", "yes", "on", "TRUE", "Yes"):
        monkeypatch.setenv("PF_TEST_BOOL", val)
        assert config._env_bool("PF_TEST_BOOL", False) is True


def test_env_bool_false_values(monkeypatch):
    for val in ("0", "false", "no", "off", "garbage", ""):
        monkeypatch.setenv("PF_TEST_BOOL", val)
        assert config._env_bool("PF_TEST_BOOL", True) is False


def test_env_bool_default_when_unset(monkeypatch):
    monkeypatch.delenv("PF_TEST_BOOL", raising=False)
    assert config._env_bool("PF_TEST_BOOL", True) is True
    assert config._env_bool("PF_TEST_BOOL", False) is False


def test_settings_creates_directories(monkeypatch, tmp_path):
    data_dir = tmp_path / "custom-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "custom-models"))
    s = config.Settings()
    for name in ("data_dir", "work_dir", "thumb_dir", "crops_dir", "backup_dir", "models_dir"):
        assert getattr(s, name).exists(), f"{name} not created"
    assert s.db_path == data_dir / "index.sqlite3"


def test_settings_env_values(monkeypatch):
    monkeypatch.setenv("BRIDGE_URL", "http://example.test:9999")
    monkeypatch.setenv("PORT", "8081")
    monkeypatch.setenv("SYNC_INTERVAL", "42")
    monkeypatch.setenv("FACE_SIM_THRESHOLD", "0.6")
    monkeypatch.setenv("MIN_CLUSTER_SIZE", "5")
    monkeypatch.setenv("AUTH_ACCESS_TTL", "100")
    s = config.Settings()
    assert s.bridge_url == "http://example.test:9999"
    assert s.port == 8081
    assert s.sync_interval == 42
    assert s.face_sim_threshold == 0.6
    assert s.min_cluster_size == 5


def test_settings_uvicorn_workers_default():
    s = config.Settings()
    assert s.uvicorn_workers == 2


def test_settings_uvicorn_workers_override(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "4")
    s = config.Settings()
    assert s.uvicorn_workers == 4


def test_settings_defaults():
    s = config.Settings()
    assert s.port == 8080
    assert s.sync_interval == 300
    assert s.min_cluster_size == 3
    assert s.face_sim_threshold == 0.45
    assert s.grace_cycles == 2


def test_settings_models_dir_defaults_under_data():
    s = config.Settings()
    assert s.models_dir == s.data_dir / "models"


def test_thumbnails_batch():
    s = config.Settings()
    assert s.thumbnails_batch == 30


def test_demo_and_hardening_flags(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "0")
    monkeypatch.setenv("DEMO_HARDENING_MODE", "0")
    assert config.is_demo_mode() is False
    assert config.demo_hardening_mode() is False
    monkeypatch.setenv("DEMO_MODE", "1")
    assert config.is_demo_mode() is True
    monkeypatch.setenv("DEMO_HARDENING_MODE", "on")
    assert config.demo_hardening_mode() is True

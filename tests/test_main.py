
import auth
import main
import store


class TestCreateAdmin:
    def test_creates_admin_user(self, tmp_db, monkeypatch):
        monkeypatch.setenv("ADMIN_PASSWORD", "supersecret1")
        assert main._create_admin("boss", "The Boss") == 0
        user = store.get_user_by_username("boss")
        assert user is not None
        assert user["role"] == "admin"
        assert user["display_name"] == "The Boss"

    def test_duplicate_returns_2(self, tmp_db, monkeypatch):
        monkeypatch.setenv("ADMIN_PASSWORD", "supersecret1")
        assert main._create_admin("boss", "The Boss") == 0
        assert main._create_admin("boss", "Again") == 2

    def test_short_password_returns_2(self, tmp_db, monkeypatch):
        monkeypatch.setenv("ADMIN_PASSWORD", "short")
        assert main._create_admin("boss", "The Boss") == 2
        assert store.get_user_by_username("boss") is None


class TestResetPassword:
    def test_resets_and_revokes_tokens(self, tmp_db, monkeypatch):
        monkeypatch.setenv("ADMIN_PASSWORD", "brand-new-pass")
        assert main._create_admin("boss", "The Boss") == 0
        user = store.get_user_by_username("boss")
        token = store.issue_token(user["id"], "access", 3600, user_agent="t", ip="1.2.3.4")
        assert main._reset_password("boss") == 0
        assert store.lookup_token(token) is None
        updated = store.get_user_by_username("boss")
        assert updated["password_hash"] != user["password_hash"]

    def test_missing_user_returns_2(self, tmp_db, monkeypatch):
        assert main._reset_password("ghost") == 2

    def test_short_password_returns_2(self, tmp_db, monkeypatch):
        monkeypatch.setenv("ADMIN_PASSWORD", "validpass123")
        assert main._create_admin("boss", "The Boss") == 0
        monkeypatch.setenv("ADMIN_PASSWORD", "x")
        assert main._reset_password("boss") == 2

    def test_password_verifies(self, tmp_db, monkeypatch):
        monkeypatch.setenv("ADMIN_PASSWORD", "brand-new-pass")
        assert main._create_admin("boss", "The Boss") == 0
        main._reset_password("boss")
        updated = store.get_user_by_username("boss")
        assert auth.verify_password("brand-new-pass", updated["password_hash"])


class TestDisable2FA:
    def test_disables_and_revokes_pending(self, tmp_db, monkeypatch):
        monkeypatch.setenv("ADMIN_PASSWORD", "supersecret1")
        assert main._create_admin("boss", "The Boss") == 0
        user = store.get_user_by_username("boss")
        store.set_totp_secret(user["id"], "encrypted-secret")
        store.set_totp_enabled(user["id"], True)
        store.create_pending_2fa(user["id"], 3600)
        assert main._disable_2fa("boss") == 0
        updated = store.get_user_by_username("boss")
        assert updated["totp_enabled"] == 0
        assert updated["totp_secret_enc"] is None
        # pending token was revoked
        assert store.revoke_pending_2fa_for_user(user["id"]) == 0

    def test_missing_user_returns_2(self, tmp_db):
        assert main._disable_2fa("ghost") == 2

    def test_noop_when_2fa_off(self, tmp_db, monkeypatch):
        monkeypatch.setenv("ADMIN_PASSWORD", "supersecret1")
        assert main._create_admin("boss", "The Boss") == 0
        assert main._disable_2fa("boss") == 0
        updated = store.get_user_by_username("boss")
        assert updated["totp_enabled"] == 0


class TestUvicornWorkers:
    def test_default_is_two(self, monkeypatch):
        monkeypatch.delenv("RUN_INDEXER", raising=False)
        monkeypatch.delenv("UVICORN_WORKERS", raising=False)
        assert main._uvicorn_workers() == 2

    def test_env_override(self, monkeypatch):
        monkeypatch.delenv("RUN_INDEXER", raising=False)
        monkeypatch.setenv("UVICORN_WORKERS", "4")
        assert main._uvicorn_workers() == 4

    def test_run_indexer_forces_single_process(self, monkeypatch):
        monkeypatch.setenv("RUN_INDEXER", "1")
        monkeypatch.setenv("UVICORN_WORKERS", "4")
        assert main._uvicorn_workers() == 1

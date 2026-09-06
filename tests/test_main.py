
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

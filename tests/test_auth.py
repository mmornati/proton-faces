
import logging
import time

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import auth
import store


def _request(path, token=None, query=""):
    headers = []
    if token:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "headers": headers,
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 1234),
        "root_path": "",
    }
    return Request(scope)


class TestPassword:
    def test_hash_verify_roundtrip(self):
        h = auth.hash_password("s3cret!")
        assert auth.verify_password("s3cret!", h) is True

    def test_wrong_password_fails(self):
        h = auth.hash_password("correct")
        assert auth.verify_password("wrong", h) is False

    def test_malformed_hash_fails(self):
        assert auth.verify_password("pw", "not-a-bcrypt-hash") is False


class TestSignedTokens:
    def test_valid_token(self):
        sig, exp = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=300)
        assert auth.verify_signed_token("/api/photos/abc/thumb", sig, exp) is True

    def test_exp_quantized_to_hour_bucket(self):
        # Two tokens minted within the same hour must be byte-identical so the
        # `Cache-Control: immutable` headers on the binary endpoints actually
        # get used (a fresh ?sig=&exp= per page load defeats them).
        _, exp1 = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=10)
        _, exp2 = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=10)
        now = int(time.time())
        next_hour = (now // 3600 + 1) * 3600
        assert exp1 == exp2 == next_hour

    def test_ttl_is_minimum_lifetime(self, monkeypatch):
        # Late in an hour bucket the ttl floor (now + ttl) wins over the
        # boundary, so a requested ttl is never cut short.
        next_hour = (int(time.time()) // 3600 + 1) * 3600
        near_end = next_hour - 10  # 10 s before the bucket rolls over
        monkeypatch.setattr(auth.time, "time", lambda: near_end)
        _, exp = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=300)
        assert exp == near_end + 300

    def test_expired_token(self, monkeypatch):
        # exp is quantized to a future hour boundary, so make_signed_token
        # can't mint an already-expired token; simulate time passing instead.
        sig, exp = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=300)
        assert auth.verify_signed_token("/api/photos/abc/thumb", sig, exp) is True
        monkeypatch.setattr(auth.time, "time", lambda: exp + 1)  # clock moves past exp
        assert auth.verify_signed_token("/api/photos/abc/thumb", sig, exp) is False

    def test_tampered_path(self):
        sig, exp = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=300)
        assert auth.verify_signed_token("/api/photos/other/thumb", sig, exp) is False

    def test_missing_sig_or_exp(self):
        assert auth.verify_signed_token("/x", None, 9999999999) is False
        assert auth.verify_signed_token("/x", "abc", None) is False


class TestRoles:
    def test_has_role_hierarchy(self):
        admin = auth.CurrentUser(1, "a", "A", "admin")
        write = auth.CurrentUser(2, "w", "W", "write")
        read = auth.CurrentUser(3, "r", "R", "read")
        assert admin.has_role("read") and admin.has_role("write") and admin.has_role("admin")
        assert write.has_role("write") and not write.has_role("admin")
        assert read.has_role("read") and not read.has_role("write")

    def test_require_role_unknown_role(self):
        with pytest.raises(ValueError):
            auth.require_role("superuser")


class TestRequireUser:
    def _mk_user(self, role="read", disabled=False):
        uid = store.create_user("alice", "hash", role=role, display_name="Alice")
        if disabled:
            store.update_user(uid, display_name="Alice", role=role, disabled=True)
        return uid

    def test_auth_free_path_returns_none(self, tmp_db):
        assert auth.require_user(_request("/api/health")) is None
        assert auth.require_user(_request("/api/status")) is None

    def test_missing_token_raises_401(self, tmp_db):
        with pytest.raises(HTTPException) as exc:
            auth.require_user(_request("/api/photos"))
        assert exc.value.status_code == 401

    def test_invalid_token_raises_401(self, tmp_db):
        with pytest.raises(HTTPException) as exc:
            auth.require_user(_request("/api/photos", token="bogus"))
        assert exc.value.status_code == 401

    def test_valid_token_returns_user(self, tmp_db):
        uid = self._mk_user(role="write")
        token = store.issue_token(uid, "access", 3600)
        user = auth.require_user(_request("/api/photos", token=token))
        assert user.username == "alice"
        assert user.role == "write"

    def test_expired_token_raises_401(self, tmp_db):
        uid = self._mk_user()
        token = store.issue_token(uid, "access", -1)
        with pytest.raises(HTTPException) as exc:
            auth.require_user(_request("/api/photos", token=token))
        assert exc.value.status_code == 401
        assert store.lookup_token(token) is None  # revoked on use

    def test_disabled_user_raises_403(self, tmp_db):
        uid = self._mk_user(disabled=True)
        token = store.issue_token(uid, "access", 3600)
        with pytest.raises(HTTPException) as exc:
            auth.require_user(_request("/api/photos", token=token))
        assert exc.value.status_code == 403

    def test_refresh_token_rejected_on_api(self, tmp_db):
        uid = self._mk_user()
        token = store.issue_token(uid, "refresh", 3600)
        with pytest.raises(HTTPException) as exc:
            auth.require_user(_request("/api/photos", token=token))
        assert exc.value.status_code == 401


class TestRequireRoleDep:
    def test_allows_privileged(self, tmp_db):
        dep = auth.require_role("write")
        user = auth.CurrentUser(1, "u", "U", "write")
        assert dep(user=user) == user

    def test_blocks_low_role(self, tmp_db):
        dep = auth.require_role("write")
        user = auth.CurrentUser(1, "u", "U", "read")
        with pytest.raises(HTTPException) as exc:
            dep(user=user)
        assert exc.value.status_code == 403

    def test_blocks_none_user(self, tmp_db):
        dep = auth.require_role("write")
        with pytest.raises(HTTPException) as exc:
            dep(user=None)
        assert exc.value.status_code == 401


class TestLoginRefresh:
    def test_login_success_issues_tokens(self, tmp_db):
        h = auth.hash_password("s3cret!")
        store.create_user("bob", h, role="admin", display_name="Bob")
        access, refresh, user = auth.login("bob", "s3cret!", user_agent="t", ip="1.2.3.4")
        assert user.username == "bob"
        assert store.lookup_token(access)["kind"] == "access"
        assert store.lookup_token(refresh)["kind"] == "refresh"

    def test_login_wrong_password_401(self, tmp_db):
        store.create_user("bob", auth.hash_password("s3cret!"))
        with pytest.raises(HTTPException) as exc:
            auth.login("bob", "wrong-password")
        assert exc.value.status_code == 401

    def test_login_unknown_user_401(self, tmp_db):
        with pytest.raises(HTTPException) as exc:
            auth.login("ghost", "whatever-password")
        assert exc.value.status_code == 401

    def test_login_disabled_user_401(self, tmp_db):
        h = auth.hash_password("s3cret!")
        uid = store.create_user("carol", h)
        store.update_user(uid, display_name="Carol", role="read", disabled=True)
        with pytest.raises(HTTPException) as exc:
            auth.login("carol", "s3cret!")
        assert exc.value.status_code == 401

    def test_refresh_rotates_token(self, tmp_db):
        store.create_user("dave", auth.hash_password("s3cret!"))
        access, refresh, _ = auth.login("dave", "s3cret!")
        new_access, new_refresh, user = auth.refresh(refresh)
        assert user.username == "dave"
        assert store.lookup_token(refresh) is None  # old refresh revoked
        assert store.lookup_token(new_refresh)["kind"] == "refresh"
        assert store.lookup_token(new_access)["kind"] == "access"

    def test_refresh_rejects_access_token(self, tmp_db):
        store.create_user("erin", auth.hash_password("s3cret!"))
        access, _, _ = auth.login("erin", "s3cret!")
        with pytest.raises(HTTPException) as exc:
            auth.refresh(access)
        assert exc.value.status_code == 401

    def test_refresh_rejects_expired(self, tmp_db):
        store.create_user("frank", auth.hash_password("s3cret!"))
        _, refresh, _ = auth.login("frank", "s3cret!")
        store.revoke_token(refresh)
        with pytest.raises(HTTPException) as exc:
            auth.refresh(refresh)
        assert exc.value.status_code == 401


class TestLoginRateLimit:
    def _clock(self, monkeypatch):
        state = {"now": 1_000_000.0}
        monkeypatch.setattr(auth.time, "time", lambda: state["now"])
        return state

    def test_lockout_after_max_failures(self, tmp_db, monkeypatch):
        self._clock(monkeypatch)
        store.create_user("bob", auth.hash_password("s3cret!"))
        for _ in range(auth._LOGIN_MAX_FAILURES):
            with pytest.raises(HTTPException) as exc:
                auth.login("bob", "wrong", ip="1.2.3.4")
            assert exc.value.status_code == 401
        with pytest.raises(HTTPException) as exc:
            auth.login("bob", "s3cret!", ip="1.2.3.4")
        assert exc.value.status_code == 429
        assert "Retry-After" in exc.value.headers

    def test_lockout_expires(self, tmp_db, monkeypatch):
        state = self._clock(monkeypatch)
        store.create_user("bob", auth.hash_password("s3cret!"))
        for _ in range(auth._LOGIN_MAX_FAILURES):
            with pytest.raises(HTTPException):
                auth.login("bob", "wrong", ip="1.2.3.4")
        state["now"] += auth._LOGIN_LOCKOUT_SEC + 1
        access, _, _ = auth.login("bob", "s3cret!", ip="1.2.3.4")
        assert store.lookup_token(access)["kind"] == "access"

    def test_different_username_unaffected(self, tmp_db, monkeypatch):
        self._clock(monkeypatch)
        store.create_user("bob", auth.hash_password("s3cret!"))
        store.create_user("carol", auth.hash_password("s3cret!"))
        for _ in range(auth._LOGIN_MAX_FAILURES):
            with pytest.raises(HTTPException):
                auth.login("bob", "wrong", ip="1.2.3.4")
        access, _, _ = auth.login("carol", "s3cret!", ip="1.2.3.4")
        assert store.lookup_token(access)["kind"] == "access"

    def test_success_resets_counter(self, tmp_db, monkeypatch):
        self._clock(monkeypatch)
        store.create_user("bob", auth.hash_password("s3cret!"))
        for _ in range(auth._LOGIN_MAX_FAILURES - 1):
            with pytest.raises(HTTPException):
                auth.login("bob", "wrong", ip="1.2.3.4")
        auth.login("bob", "s3cret!", ip="1.2.3.4")
        for _ in range(auth._LOGIN_MAX_FAILURES):
            with pytest.raises(HTTPException):
                auth.login("bob", "wrong", ip="1.2.3.4")
        with pytest.raises(HTTPException) as exc:
            auth.login("bob", "s3cret!", ip="1.2.3.4")
        assert exc.value.status_code == 429

    def test_neutral_body(self, tmp_db, monkeypatch):
        self._clock(monkeypatch)
        store.create_user("bob", auth.hash_password("s3cret!"))
        for _ in range(auth._LOGIN_MAX_FAILURES):
            with pytest.raises(HTTPException):
                auth.login("bob", "wrong", ip="1.2.3.4")
        with pytest.raises(HTTPException) as exc:
            auth.login("bob", "s3cret!", ip="1.2.3.4")
        assert "ip" not in exc.value.detail.lower()
        assert "username" not in exc.value.detail.lower()


class TestSignedOrToken:
    def test_valid_signed_url_passes(self, tmp_db, monkeypatch):
        monkeypatch.setenv("DEMO_ALLOW_PUBLIC_THUMBS", "0")
        sig, exp = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=300)
        req = _request("/api/photos/abc/thumb", query=f"sig={sig}&exp={exp}")
        assert auth.signed_or_token(req) is None

    def test_invalid_signed_url_falls_back_to_401(self, tmp_db, monkeypatch):
        monkeypatch.setenv("DEMO_ALLOW_PUBLIC_THUMBS", "0")
        req = _request("/api/photos/abc/thumb", query="sig=bogus&exp=9999999999")
        with pytest.raises(HTTPException) as exc:
            auth.signed_or_token(req)
        assert exc.value.status_code == 401

    def test_valid_bearer_passes(self, tmp_db, monkeypatch):
        monkeypatch.setenv("DEMO_ALLOW_PUBLIC_THUMBS", "0")
        uid = store.create_user("grace", "hash")
        token = store.issue_token(uid, "access", 3600)
        req = _request("/api/photos/abc/thumb", token=token)
        user = auth.signed_or_token(req)
        assert user is not None and user.username == "grace"


class TestSigningSecretRequired:
    def test_raises_when_unset_and_not_demo(self, monkeypatch):
        monkeypatch.delenv("SIGNING_SECRET", raising=False)
        monkeypatch.delenv("DEMO_MODE", raising=False)
        with pytest.raises(RuntimeError, match="SIGNING_SECRET"):
            auth._signing_secret()

    def test_make_signed_token_fails_closed(self, monkeypatch):
        monkeypatch.delenv("SIGNING_SECRET", raising=False)
        monkeypatch.delenv("DEMO_MODE", raising=False)
        with pytest.raises(RuntimeError, match="SIGNING_SECRET"):
            auth.make_signed_token("/api/photos/abc/thumb")

    def test_verify_signed_token_fails_closed(self, monkeypatch):
        monkeypatch.delenv("SIGNING_SECRET", raising=False)
        monkeypatch.delenv("DEMO_MODE", raising=False)
        with pytest.raises(RuntimeError, match="SIGNING_SECRET"):
            auth.verify_signed_token("/api/photos/abc/thumb", "sig", 9999999999)

    def test_explicit_secret_roundtrip(self):
        sig, exp = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=300)
        assert auth.verify_signed_token("/api/photos/abc/thumb", sig, exp) is True


class TestDemoModeEphemeralSecret:
    def test_demo_mode_falls_back_to_ephemeral(self, monkeypatch, caplog):
        monkeypatch.delenv("SIGNING_SECRET", raising=False)
        monkeypatch.setenv("DEMO_MODE", "1")
        with caplog.at_level(logging.WARNING, logger="auth"):
            sig, exp = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=300)
        assert auth.verify_signed_token("/api/photos/abc/thumb", sig, exp) is True
        assert any("ephemeral" in r.message for r in caplog.records)

    def test_ephemeral_secret_is_stable_within_process(self, monkeypatch):
        monkeypatch.delenv("SIGNING_SECRET", raising=False)
        monkeypatch.setenv("DEMO_MODE", "1")
        assert auth._signing_secret() == auth._signing_secret()

    def test_demo_mode_deleted_still_fails_closed(self, monkeypatch):
        monkeypatch.delenv("SIGNING_SECRET", raising=False)
        monkeypatch.delenv("DEMO_MODE", raising=False)
        with pytest.raises(RuntimeError, match="SIGNING_SECRET"):
            auth.require_signing_secret()


class TestAllowPublicThumbs:
    def test_default_is_false(self, monkeypatch):
        monkeypatch.delenv("DEMO_ALLOW_PUBLIC_THUMBS", raising=False)
        monkeypatch.delenv("DEMO_MODE", raising=False)
        monkeypatch.delenv("DEMO_HARDENING_MODE", raising=False)
        assert auth.allow_public_thumbs() is False

    def test_explicit_opt_in(self, monkeypatch):
        monkeypatch.setenv("DEMO_ALLOW_PUBLIC_THUMBS", "1")
        assert auth.allow_public_thumbs() is True

    def test_hardening_mode_stays_secure_by_default(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "1")
        monkeypatch.delenv("DEMO_ALLOW_PUBLIC_THUMBS", raising=False)
        assert auth.allow_public_thumbs() is False

    def test_explicit_env_wins_over_hardening_mode(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "1")
        monkeypatch.setenv("DEMO_ALLOW_PUBLIC_THUMBS", "1")
        assert auth.allow_public_thumbs() is True


class TestDemoDisableAdminUserManagement:
    """The hardening override resolves DEMO_DISABLE_ADMIN_USER_MANAGEMENT the
    same way as the other demo flags: explicit env wins, hardening mode
    flips the unset value to the safe side."""

    def test_default_is_false(self, monkeypatch):
        monkeypatch.delenv("DEMO_DISABLE_ADMIN_USER_MANAGEMENT", raising=False)
        monkeypatch.delenv("DEMO_MODE", raising=False)
        monkeypatch.delenv("DEMO_HARDENING_MODE", raising=False)
        assert auth.demo_disable_admin_user_management() is False

    def test_explicit_opt_in(self, monkeypatch):
        monkeypatch.setenv("DEMO_DISABLE_ADMIN_USER_MANAGEMENT", "1")
        assert auth.demo_disable_admin_user_management() is True

    def test_hardening_mode_flips_unset_to_safe(self, monkeypatch):
        # When DEMO_HARDENING_MODE is on and the env var is unset, the
        # safe (True) value is returned.
        monkeypatch.delenv("DEMO_DISABLE_ADMIN_USER_MANAGEMENT", raising=False)
        monkeypatch.setenv("DEMO_HARDENING_MODE", "1")
        assert auth.demo_disable_admin_user_management() is True

    def test_explicit_opt_out_wins_over_hardening_mode(self, monkeypatch):
        # Operators can disable the gate explicitly even with hardening mode on.
        monkeypatch.setenv("DEMO_HARDENING_MODE", "1")
        monkeypatch.setenv("DEMO_DISABLE_ADMIN_USER_MANAGEMENT", "0")
        assert auth.demo_disable_admin_user_management() is False


class TestDemoLoginLogs:
    """DEMO_LOGIN_LOGS defaults to OFF; hardening mode keeps it OFF;
    explicit env var wins."""

    def test_default_is_false(self, monkeypatch):
        monkeypatch.delenv("DEMO_LOGIN_LOGS", raising=False)
        monkeypatch.delenv("DEMO_MODE", raising=False)
        monkeypatch.delenv("DEMO_HARDENING_MODE", raising=False)
        assert auth.demo_login_logs() is False

    def test_explicit_opt_in(self, monkeypatch):
        monkeypatch.setenv("DEMO_LOGIN_LOGS", "1")
        assert auth.demo_login_logs() is True

    def test_hardening_mode_stays_off(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "1")
        monkeypatch.delenv("DEMO_LOGIN_LOGS", raising=False)
        assert auth.demo_login_logs() is False

    def test_explicit_env_wins_over_hardening_mode(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "1")
        monkeypatch.setenv("DEMO_LOGIN_LOGS", "1")
        assert auth.demo_login_logs() is True


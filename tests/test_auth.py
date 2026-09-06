
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

    def test_expired_token(self, monkeypatch):
        sig, exp = auth.make_signed_token("/api/photos/abc/thumb", ttl_seconds=-10)
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

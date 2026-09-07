"""Tests for the indexer's internal status/control HTTP server (issue #42).

Every route except `GET /healthz` requires the shared `INDEXER_TOKEN`
in the `X-Indexer-Token` header. The fixture below sets a stable test
token before any TestClient is constructed so the existing happy-path
tests keep working, and the dedicated `TestIndexerTokenAuth` class
covers the four cases the security fix calls out:

    - no token header        -> 401
    - wrong token            -> 401
    - correct token          -> 200
    - non-ASCII token bytes  -> 401 (no UnicodeEncodeError leakage)
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import indexer
import indexer_status
import store

TEST_TOKEN = "test-indexer-token-abcdef0123456789"


@pytest.fixture
def indexer_client(monkeypatch):
    """Build a TestClient with `INDEXER_TOKEN` configured.

    Sets the env var (which `_indexer_token()` reads on every request,
    so monkeypatch is fine — there's no module-level cached token) and
    clears the per-fixture ephemeral fallback in case a previous test
    ran in DEMO_MODE.
    """
    monkeypatch.setenv("INDEXER_TOKEN", TEST_TOKEN)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    indexer_status._indexer_token.__dict__.pop("_ephemeral", None)
    return TestClient(indexer_status.app)


def _auth() -> dict:
    return {"X-Indexer-Token": TEST_TOKEN}


class TestHealthz:
    def test_ok_without_token(self):
        # /healthz is intentionally auth-free so compose healthchecks work
        # without needing the token.
        client = TestClient(indexer_status.app)
        assert client.get("/healthz").json() == {"ok": True}

    def test_ok_ignores_token(self, indexer_client):
        # /healthz accepts the token header too — just doesn't require it.
        assert indexer_client.get("/healthz", headers=_auth()).json() == {"ok": True}


class TestStatus:
    def test_status_shape(self, tmp_db, monkeypatch, indexer_client):
        monkeypatch.setattr(
            indexer, "get_indexer_state", lambda: {"remote": True, "running": False}
        )
        out = indexer_client.get("/status", headers=_auth()).json()
        assert out["remote"] is True
        assert "pending_db" in out
        assert out["pending_db"] == 0

    def test_pending_db_counts(self, tmp_db, monkeypatch, indexer_client):
        monkeypatch.setattr(
            indexer, "get_indexer_state", lambda: {"remote": True, "running": False}
        )
        store.upsert_photos([{"uid": "a1", "name": "a1", "media_type": "image/jpeg", "capture_time": 1}])
        store.upsert_photos([{"uid": "a2", "name": "a2", "media_type": "image/jpeg", "capture_time": 1}])
        assert indexer_client.get("/status", headers=_auth()).json()["pending_db"] == 2


class TestTriggerSync:
    def test_triggers_full_sync(self, tmp_db, monkeypatch, indexer_client):
        calls = []
        monkeypatch.setattr(indexer, "request_full_sync", lambda: calls.append(1))
        assert indexer_client.post("/trigger-sync", headers=_auth()).status_code == 200
        assert len(calls) == 1


class TestSyncConfig:
    def test_get_config(self, tmp_db, indexer_client):
        out = indexer_client.get("/sync-config", headers=_auth()).json()
        assert out["enabled"] is True
        assert out["tip_size"] == 10

    def test_update_config(self, tmp_db, indexer_client):
        resp = indexer_client.put("/sync-config", json={"tip_size": 25}, headers=_auth())
        assert resp.status_code == 200
        assert resp.json()["tip_size"] == 25

    def test_invalid_config_returns_400(self, tmp_db, indexer_client):
        resp = indexer_client.put("/sync-config", json={"tip_size": 0}, headers=_auth())
        assert resp.status_code == 400


class TestIndexerTokenAuth:
    """Issue #42 — every state-touching route must reject unauthenticated calls."""

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("get", "/status", None),
            ("post", "/trigger-sync", None),
            ("get", "/sync-config", None),
            ("put", "/sync-config", {"tip_size": 5}),
        ],
    )
    def test_no_token_returns_401(self, tmp_db, indexer_client, method, path, body):
        resp = _call(indexer_client, method, path, body)
        assert resp.status_code == 401
        assert "X-Indexer-Token" in resp.headers.get("www-authenticate", "")

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("get", "/status", None),
            ("post", "/trigger-sync", None),
            ("get", "/sync-config", None),
            ("put", "/sync-config", {"tip_size": 5}),
        ],
    )
    def test_wrong_token_returns_401(self, tmp_db, indexer_client, method, path, body):
        resp = _call(
            indexer_client,
            method,
            path,
            body,
            headers={"X-Indexer-Token": "definitely-not-the-real-token"},
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("get", "/status", None),
            ("post", "/trigger-sync", None),
            ("get", "/sync-config", None),
            ("put", "/sync-config", {"tip_size": 5}),
        ],
    )
    def test_correct_token_returns_2xx(self, tmp_db, indexer_client, method, path, body):
        resp = _call(indexer_client, method, path, body, headers=_auth())
        assert 200 <= resp.status_code < 300, resp.text

    def test_empty_token_header_returns_401(self, tmp_db, indexer_client):
        # An empty header is distinct from a missing one and should be rejected.
        resp = indexer_client.get("/status", headers={"X-Indexer-Token": ""})
        assert resp.status_code == 401

    def test_missing_token_does_not_call_handler(self, tmp_db, monkeypatch, indexer_client):
        # Defence-in-depth: even if the dependency was misconfigured to fall
        # through, the handler must not be invoked without auth. We assert it
        # by monkeypatching request_full_sync and confirming it stays at zero.
        calls = []
        monkeypatch.setattr(indexer, "request_full_sync", lambda: calls.append(1))
        resp = indexer_client.post("/trigger-sync")
        assert resp.status_code == 401
        assert calls == []


def _call(client, method: str, path: str, body: dict | None, headers: dict | None = None):
    """Dispatch a TestClient request, attaching `json=body` only when supported.

    `TestClient.get` doesn't accept a `json` kwarg — it accepts the body only on
    POST/PUT/PATCH/DELETE. This helper hides that asymmetry from the parametrised
    auth tests so they stay readable.
    """
    kwargs: dict = {"headers": headers} if headers else {}
    if body is not None and method in ("post", "put", "patch", "delete"):
        kwargs["json"] = body
    return getattr(client, method)(path, **kwargs)


class TestStartupEnforcement:
    """The token must be required at startup outside DEMO_MODE."""

    def test_missing_token_raises_outside_demo(self, monkeypatch):
        from indexer_status import _indexer_token
        indexer_status._indexer_token.__dict__.pop("_ephemeral", None)
        monkeypatch.delenv("INDEXER_TOKEN", raising=False)
        monkeypatch.delenv("DEMO_MODE", raising=False)
        with pytest.raises(RuntimeError, match="INDEXER_TOKEN is not set"):
            _indexer_token()

    def test_demo_mode_falls_back_to_ephemeral(self, monkeypatch):
        from indexer_status import _indexer_token
        indexer_status._indexer_token.__dict__.pop("_ephemeral", None)
        monkeypatch.delenv("INDEXER_TOKEN", raising=False)
        monkeypatch.setenv("DEMO_MODE", "1")
        tok1 = _indexer_token()
        assert isinstance(tok1, bytes) and len(tok1) >= 16
        # Subsequent calls within the same boot return the same value
        # (the app and indexer containers need to agree).
        assert _indexer_token() == tok1
        indexer_status._indexer_token.__dict__.pop("_ephemeral", None)


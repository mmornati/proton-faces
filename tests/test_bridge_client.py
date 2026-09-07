import json

import httpx
import pytest

import bridge_client


@pytest.fixture
def client_factory(monkeypatch):
    """Return a function that builds a BridgeClient backed by a MockTransport."""
    import config  # noqa: F401

    def _build(handler):
        transport = httpx.MockTransport(handler)
        bc = bridge_client.BridgeClient(base_url="http://bridge.test")
        bc._client = httpx.Client(transport=transport, base_url="http://bridge.test", timeout=120.0)
        return bc

    return _build


class TestParseRetryAfter:
    def test_numeric(self):
        assert bridge_client._parse_retry_after("42") == 42.0

    def test_numeric_clamped(self):
        assert bridge_client._parse_retry_after("5000") == 600.0
        assert bridge_client._parse_retry_after("-5") == 0.0

    def test_http_date(self):
        assert bridge_client._parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") > 0

    def test_garbage(self):
        assert bridge_client._parse_retry_after("later") == 0.0
        assert bridge_client._parse_retry_after("") == 0.0


class TestHealth:
    def test_health_ok(self, client_factory):
        bc = client_factory(lambda req: httpx.Response(200, json={"ok": True, "loggedIn": True}))
        assert bc.health() == {"ok": True, "loggedIn": True}

    def test_health_error(self, client_factory):
        bc = client_factory(lambda req: httpx.Response(500, text="boom"))
        with pytest.raises(httpx.HTTPStatusError):
            bc.health()


class TestTimeline:
    def test_timeline_parses_ndjson_and_skips_comments(self, client_factory):
        body = '# progress 1\n{"uid": "a", "name": "A"}\n\n{"uid": "b", "name": "B"}\n# done: 2\n'

        def handler(req):
            assert req.url.path == "/timeline"
            assert "limit" not in req.url.params
            return httpx.Response(200, text=body)

        bc = client_factory(handler)
        items = bc.timeline()
        assert [i["uid"] for i in items] == ["a", "b"]

    def test_timeline_limit_param(self, client_factory):
        def handler(req):
            assert req.url.params["limit"] == "10"
            return httpx.Response(200, text='{"uid":"a"}\n')

        bc = client_factory(handler)
        assert len(bc.timeline(limit=10)) == 1


class TestTimelineIds:
    def test_parses_with_done_sentinel(self, client_factory):
        body = '{"uid": "a", "captureTime": "2024-01-01T00:00:00Z"}\n{"uid": "b"}\n# done: 2\n'

        def handler(req):
            assert req.url.path == "/timeline/ids"
            return httpx.Response(200, text=body)

        bc = client_factory(handler)
        items = bc.timeline_ids()
        assert [i["uid"] for i in items] == ["a", "b"]

    def test_truncated_raises(self, client_factory):
        body = '{"uid": "a"}\n# done: 3\n'

        def handler(req):
            return httpx.Response(200, text=body)

        bc = client_factory(handler)
        with pytest.raises(bridge_client.BridgeError):
            bc.timeline_ids()

    def test_no_sentinel_skips_verification(self, client_factory):
        def handler(req):
            return httpx.Response(200, text='{"uid": "a"}\n')

        bc = client_factory(handler)
        assert bc.timeline_ids() == [{"uid": "a"}]


class TestNodes:
    def test_nodes_empty_returns_empty(self, client_factory):
        bc = client_factory(lambda req: httpx.Response(200))
        assert bc.nodes([]) == []

    def test_nodes_posts_uids(self, client_factory):
        def handler(req):
            assert req.method == "POST"
            assert req.url.path == "/nodes"
            assert json.loads(req.content) == {"uids": ["a", "b"]}
            return httpx.Response(200, text='{"uid":"a"}\n{"uid":"b"}\n')

        bc = client_factory(handler)
        assert [n["uid"] for n in bc.nodes(["a", "b"])] == ["a", "b"]


class TestAlbumsAndThumbnails:
    def test_albums(self, client_factory):
        def handler(req):
            return httpx.Response(200, json={"albums": [{"uid": "al1", "name": "Trip"}]})

        bc = client_factory(handler)
        assert bc.albums()["albums"][0]["name"] == "Trip"

    def test_thumbnails_posts(self, client_factory):
        def handler(req):
            assert json.loads(req.content) == {"uids": ["a"]}
            return httpx.Response(200, json={"results": [{"uid": "a", "ok": True}]})

        bc = client_factory(handler)
        out = bc.thumbnails(["a"])
        assert out["results"][0]["ok"] is True


class TestIsValidUid:
    def test_accepts_plain_uids(self):
        for uid in ("abc123", "ABC_123-def", "0", "photo-uid_1", "a" * 128,
                     "PNR_abc==~def==", "uid_with=padding==", "a~b"):
            assert bridge_client._is_valid_uid(uid)

    def test_rejects_traversal_and_garbage(self):
        for uid in ("../../etc/passwd", "..", "a/b", "a\\b", "a b", "a\nb", "", "a" * 129):
            assert not bridge_client._is_valid_uid(uid)


class TestFullPhoto:
    def test_streams_with_headers(self, client_factory):
        captured = {}

        def handler(req):
            captured["range"] = req.headers.get("Range")
            captured["timeout"] = req.headers.get("X-Timeout-Ms")
            return httpx.Response(200, content=b"\xff\xd8\xff", headers={"Content-Type": "image/jpeg"})

        bc = client_factory(handler)
        resp = bc.full_photo("p1", range_header="bytes=0-100", timeout_ms=5000)
        assert resp.status_code == 200
        assert captured["range"] == "bytes=0-100"
        assert captured["timeout"] == "5000"

    def test_no_range_when_none(self, client_factory):
        def handler(req):
            assert "Range" not in req.headers
            return httpx.Response(200, content=b"x")

        bc = client_factory(handler)
        resp = bc.full_photo("p1")
        assert resp.status_code == 200

    def test_invalid_uid_raises_value_error(self, client_factory):
        bc = client_factory(lambda req: httpx.Response(200, content=b"x"))
        with pytest.raises(ValueError):
            bc.full_photo("../../etc/passwd")

    def test_429_raises_transient_with_retry_after(self, client_factory):
        def handler(req):
            return httpx.Response(429, headers={"Retry-After": "60", "X-Error-Message": "slow down"})

        bc = client_factory(handler)
        with pytest.raises(bridge_client.BridgeTransientError) as exc:
            bc.full_photo("p1")
        assert exc.value.status_code == 429
        assert exc.value.retry_after_sec == 60.0
        assert "slow down" in str(exc.value)

    def test_503_raises_transient(self, client_factory):
        def handler(req):
            return httpx.Response(503, text="unavailable")

        bc = client_factory(handler)
        with pytest.raises(bridge_client.BridgeTransientError):
            bc.full_photo("p1")

    def test_404_raises_http_status_error(self, client_factory):
        def handler(req):
            return httpx.Response(404, text="missing")

        bc = client_factory(handler)
        resp = bc.full_photo("nope")
        with pytest.raises(httpx.HTTPStatusError):
            resp.raise_for_status()


class TestCacheManagement:
    def test_cache_status(self, client_factory):
        def handler(req):
            assert req.url.path == "/cache"
            return httpx.Response(200, json={"ok": True, "files": [], "uptimeSec": 5})

        bc = client_factory(handler)
        out = bc.cache_status()
        assert out["ok"] is True and out["uptimeSec"] == 5

    def test_clear_cache(self, client_factory):
        def handler(req):
            assert req.method == "POST"
            assert req.url.path == "/cache/clear"
            return httpx.Response(200, json={"ok": True, "removed": ["cache-x.sqlite"]})

        bc = client_factory(handler)
        assert bc.clear_cache()["removed"] == ["cache-x.sqlite"]


class TestGetBridge:
    def test_bridge_client_selected_by_default(self, monkeypatch):
        monkeypatch.delenv("DEMO_MODE", raising=False)
        import bridge_client as m

        m._bridge = None
        bridge = m.get_bridge()
        assert isinstance(bridge, m.BridgeClient)

    def test_demo_bridge_in_demo_mode(self, monkeypatch):
        import bridge_client as m
        import demo

        monkeypatch.setenv("DEMO_MODE", "1")
        m._bridge = None
        bridge = m.get_bridge()
        assert isinstance(bridge, demo.DemoBridge)


class TestBridgeAuthHeaders:
    """Every bridge client request must include the Authorization header when BRIDGE_TOKEN is set."""

    def test_health_sends_auth_header(self, client_factory, monkeypatch):
        monkeypatch.setattr("bridge_client.settings.bridge_token", "test-token-123")
        captured = {}

        def handler(req):
            captured["auth"] = req.headers.get("authorization")
            return httpx.Response(200, json={"ok": True, "loggedIn": True})

        bc = client_factory(handler)
        bc._token = "test-token-123"
        bc._auth_headers = {"Authorization": "Bearer test-token-123"}
        bc.health()
        assert captured["auth"] == "Bearer test-token-123"

    def test_timeline_sends_auth_header(self, client_factory, monkeypatch):
        monkeypatch.setattr("bridge_client.settings.bridge_token", "test-token-123")
        captured = {}

        def handler(req):
            captured["auth"] = req.headers.get("authorization")
            return httpx.Response(200, text='{"uid":"a"}\n')

        bc = client_factory(handler)
        bc._token = "test-token-123"
        bc._auth_headers = {"Authorization": "Bearer test-token-123"}
        bc.timeline(limit=1)
        assert captured["auth"] == "Bearer test-token-123"

    def test_full_photo_sends_auth_header(self, client_factory, monkeypatch):
        monkeypatch.setattr("bridge_client.settings.bridge_token", "test-token-123")
        captured = {}

        def handler(req):
            captured["auth"] = req.headers.get("authorization")
            return httpx.Response(200, content=b"\xff\xd8\xff")

        bc = client_factory(handler)
        bc._token = "test-token-123"
        bc._auth_headers = {"Authorization": "Bearer test-token-123"}
        bc.full_photo("p1")
        assert captured["auth"] == "Bearer test-token-123"

    def test_no_auth_header_when_token_empty(self, client_factory, monkeypatch):
        monkeypatch.setattr("bridge_client.settings.bridge_token", "")
        captured = {}

        def handler(req):
            captured["auth"] = req.headers.get("authorization")
            return httpx.Response(200, json={"ok": True, "loggedIn": True})

        bc = client_factory(handler)
        bc._token = ""
        bc._auth_headers = {}
        bc.health()
        assert captured["auth"] is None

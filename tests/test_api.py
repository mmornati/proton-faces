"""TestClient coverage of the FastAPI app in app/src/api.py.

The app has no factory: `api.app` is built at import time with a global
`require_user` dependency. We exercise it through a real TestClient against
a tmp sqlite DB, with `bridge_client._bridge` swapped for a fake bridge and
module-level ML functions (embed_text / embed_query_face) monkeypatched.
"""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import admin
import api
import auth
import bridge_client
import config
import store


def _emb(index: int, value: float = 1.0) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[index] = value
    return v / (np.linalg.norm(v) + 1e-12)


class FakeResp:
    status_code = 200

    def __init__(self, data: bytes, content_type: str = "image/jpeg",
                 headers: dict | None = None):
        self._data = data
        self._headers = {
            "content-type": content_type,
            "content-length": str(len(data)),
        }
        if headers:
            self._headers.update(headers)

    @property
    def headers(self) -> dict:
        return self._headers

    def iter_bytes(self, chunk_size: int = 65536):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]

    async def aiter_bytes(self, chunk_size: int = 65536):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]

    async def aread(self, max_bytes: int = -1) -> bytes:
        return self._data[:max_bytes]

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


class FakeBridge:
    """Minimal in-repo fake for the Proton bridge HTTP surface."""

    def __init__(self):
        self._nodes = {}
        self._albums = []
        self._full_data = b"\xff\xd8\xfffake-jpeg"
        self._albums_calls = 0

    def add_node(self, uid: str, **kw):
        node = {"uid": uid, "name": f"{uid}.jpg", "mediaType": "image/jpeg",
                "size": 1000, "tags": [], "creationTime": "2024-01-01T00:00:00Z"}
        node.update(kw)
        self._nodes[uid] = node

    def add_album(self, uid: str, name: str):
        self._albums.append({"uid": uid, "name": name})

    def health(self):
        return {"ok": True, "loggedIn": True}

    def nodes(self, uids):
        return [self._nodes[u] for u in uids if u in self._nodes]

    def albums(self):
        self._albums_calls += 1
        return {"albums": self._albums}

    def full_photo(self, uid, range_header=None, timeout_ms=None):
        return FakeResp(self._full_data)

    async def full_photo_async(self, uid, range_header=None, timeout_ms=None):
        return FakeResp(self._full_data)

    def cache_status(self):
        return {"ok": True, "files": [], "uptimeSec": 0}

    def clear_cache(self):
        return {"ok": True, "removed": [], "uptimeSec": 0}


class FailingFullBridge(FakeBridge):
    def __init__(self, exc):
        super().__init__()
        self._exc = exc

    def full_photo(self, uid, range_header=None, timeout_ms=None):
        raise self._exc

    async def full_photo_async(self, uid, range_header=None, timeout_ms=None):
        raise self._exc


@pytest.fixture(scope="session")
def password_hash():
    return auth.hash_password("password123")


@pytest.fixture
def client(tmp_db, monkeypatch):
    monkeypatch.setattr(bridge_client, "_bridge", FakeBridge())
    with TestClient(api.app) as c:
        yield c


def test_app_startup_refuses_without_signing_secret(monkeypatch):
    monkeypatch.delenv("SIGNING_SECRET", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    with pytest.raises(RuntimeError, match="SIGNING_SECRET"):
        with TestClient(api.app):
            pass


def test_app_startup_ok_with_signing_secret(monkeypatch):
    monkeypatch.setattr(bridge_client, "_bridge", FakeBridge())
    with TestClient(api.app) as c:
        assert c.get("/api/health").status_code == 200


def _seed_user(username="admin", role="admin", password_hash=None):
    return store.create_user(
        username=username,
        password_hash=password_hash,
        role=role,
        display_name=username.title(),
    )


def _bearer(client, username="admin", password="password123"):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _seed_done_photo(uid, *, thumb=True, thumb_bytes=None,
                     capture_time=1700000000, tags=None, place=None, gps=None,
                     archived=0, hidden=0, favorited=0, media_type="image/jpeg",
                     albums=None, sha1=None, name=None):
    row = {
        "uid": uid,
        "name": name or f"{uid}.jpg",
        "media_type": media_type,
        "capture_time": capture_time,
        "sha1": sha1 or f"sha-{uid}",
        "albums": albums or [],
        "size": 1000,
        "duration_sec": None,
    }
    store.upsert_photos([row])
    store.set_photo_done(uid, f"{uid}.webp", gps, place)
    if tags is not None:
        store.set_tags(uid, tags)
    if archived:
        store.set_archived(uid, True)
    if hidden:
        store.set_hidden(uid, True)
    if thumb:
        p = config.settings.thumb_dir / f"{uid}.webp"
        if thumb_bytes is not None:
            p.write_bytes(thumb_bytes)
        else:
            from PIL import Image
            img = Image.new("RGB", (64, 64), (120, 60, 200))
            img.save(p, format="WEBP")


def _jpeg_bytes() -> bytes:
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 200, 10)).save(buf, format="JPEG")
    return buf.getvalue()


def _seed_face(photo_uid, *, person_id=None, confidence=0.99, bbox=(0.1, 0.1, 0.4, 0.4),
               emb=None):
    return store.insert_face(
        photo_uid=photo_uid,
        person_id=person_id,
        confidence=confidence,
        bbox=bbox,
        embedding=(emb if emb is not None else _emb(1)).tobytes(),
    )


# --- auth / public endpoints ---------------------------------------------


class TestAuthEndpoints:
    def test_login_success(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        r = client.post("/api/auth/login", json={"username": "admin", "password": "password123"})
        assert r.status_code == 200
        body = r.json()
        assert body["token_type"] == "Bearer"
        assert body["user"]["role"] == "admin"
        assert body["expires_in"] > 0
        assert body["access_token"]
        assert body["refresh_token"]

    def test_login_missing_fields(self, client):
        assert client.post("/api/auth/login", json={}).status_code == 400
        assert client.post("/api/auth/login", json={"username": "a"}).status_code == 400
        assert client.post("/api/auth/login", json={"password": "p"}).status_code == 400

    def test_login_bad_credentials(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        r = client.post("/api/auth/login", json={"username": "admin", "password": "wrongpass"})
        assert r.status_code == 401

    def test_login_unknown_user(self, client):
        r = client.post("/api/auth/login", json={"username": "nobody", "password": "password123"})
        assert r.status_code == 401

    def test_login_lockout_429(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        for _ in range(5):
            assert client.post(
                "/api/auth/login", json={"username": "admin", "password": "wrong"}
            ).status_code == 401
        r = client.post("/api/auth/login", json={"username": "admin", "password": "password123"})
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        assert "ip" not in r.text.lower()
        assert "username" not in r.text.lower()

    def test_limits_public(self, client):
        r = client.get("/api/auth/limits")
        assert r.status_code == 200
        assert r.json() == {"min_username": 2, "min_password": 8}

    def test_me(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        headers = _bearer(client)
        r = client.get("/api/auth/me", headers=headers)
        assert r.status_code == 200
        assert r.json()["username"] == "admin"

    def test_me_requires_auth(self, client):
        assert client.get("/api/auth/me").status_code == 401

    def test_logout_revokes_token(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        headers = _bearer(client)
        assert client.post("/api/auth/logout", headers=headers).status_code == 200
        assert client.get("/api/auth/me", headers=headers).status_code == 401

    def test_refresh_rotates(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        r = client.post("/api/auth/login", json={"username": "admin", "password": "password123"})
        old_refresh = r.json()["refresh_token"]
        r2 = client.post("/api/auth/refresh", json={"refresh_token": old_refresh})
        assert r2.status_code == 200
        new_refresh = r2.json()["refresh_token"]
        assert new_refresh != old_refresh
        # old refresh token is now revoked
        assert client.post("/api/auth/refresh", json={"refresh_token": old_refresh}).status_code == 401

    def test_refresh_requires_token(self, client):
        assert client.post("/api/auth/refresh", json={}).status_code == 400

    def test_sign_returns_verifiable_urls(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        headers = _bearer(client)
        r = client.post("/api/sign", json={"paths": ["/api/photos/abc/thumb"]}, headers=headers)
        assert r.status_code == 200
        url = r.json()["urls"][0]
        assert auth.verify_signed_token(url["path"], url["sig"], url["exp"])

    def test_sign_rejects_invalid(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        headers = _bearer(client)
        assert client.post("/api/sign", json={"paths": []}, headers=headers).status_code == 400
        assert client.post("/api/sign", json={"paths": ["/etc/passwd"]}, headers=headers).status_code == 400
        assert client.post("/api/sign", json={"paths": ["/api/photos/x/full"], "ttl": 999999},
                           headers=headers).status_code == 200  # ttl clamped


class TestPublicEndpoints:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "bridge": {"reachable": True, "loggedIn": True}}

    def test_health_bridge_down(self, client, monkeypatch):
        class Down:
            def health(self):
                raise RuntimeError("nope")
        monkeypatch.setattr(bridge_client, "_bridge", Down())
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["bridge"] == {"reachable": False, "loggedIn": False}

    def test_status_public(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert "stats" in body and "indexer" in body and "disk" in body
        assert "config" not in body

    def test_status_with_bearer_shows_config(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        headers = _bearer(client)
        r = client.get("/api/status", headers=headers)
        assert r.status_code == 200
        assert r.json()["config"]["face_sim_threshold"] == 0.45


# --- photos ---------------------------------------------------------------


class TestPhotos:
    def test_photo_list(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", capture_time=1000)
        _seed_done_photo("p2", capture_time=2000, tags=["dog"])
        headers = _bearer(client)
        r = client.get("/api/photos", headers=headers)
        assert r.status_code == 200
        photos = r.json()["photos"]
        assert [p["uid"] for p in photos] == ["p2", "p1"]  # capture_time desc

    def test_photo_list_requires_auth(self, client):
        assert client.get("/api/photos").status_code == 401

    def test_photo_list_filter_tag(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", tags=["dog"])
        _seed_done_photo("p2", tags=["cat"])
        headers = _bearer(client)
        r = client.get("/api/photos", params={"tag": "dog"}, headers=headers)
        assert [p["uid"] for p in r.json()["photos"]] == ["p1"]

    def test_photo_list_filter_place(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", place="Paris, France")
        _seed_done_photo("p2", place="Berlin, Germany")
        headers = _bearer(client)
        r = client.get("/api/photos", params={"place": "paris"}, headers=headers)
        assert [p["uid"] for p in r.json()["photos"]] == ["p1"]

    def test_photo_list_only_favorites(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        uid = _seed_user("reader", "read", password_hash)
        _seed_done_photo("p1")
        _seed_done_photo("p2")
        store.favorite_photo(uid, "p1")
        headers = _bearer(client, "reader")
        r = client.get("/api/photos", params={"only_favorites": True}, headers=headers)
        assert [p["uid"] for p in r.json()["photos"]] == ["p1"]
        assert r.json()["photos"][0]["favorited_by_me"] is True

    def test_photo_detail(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", place="Paris, France")
        headers = _bearer(client)
        r = client.get("/api/photos/p1", headers=headers)
        assert r.status_code == 200
        assert r.json()["uid"] == "p1"
        assert r.json()["kind"] == "image"
        assert r.json()["place"] == "Paris, France"

    def test_photo_detail_404(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        headers = _bearer(client)
        assert client.get("/api/photos/missing", headers=headers).status_code == 404

    def test_photo_archived(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        _seed_done_photo("p2", archived=1)
        headers = _bearer(client)
        r = client.get("/api/photos/archived", headers=headers)
        assert [p["uid"] for p in r.json()["photos"]] == ["p2"]

    def test_patch_photo_favorite_archive(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        headers = _bearer(client)
        r = client.patch("/api/photos/p1", json={"favorited": True, "archived": True}, headers=headers)
        assert r.status_code == 200
        assert client.get("/api/photos/p1", headers=headers).json()["archived"] is True
        assert store.is_favorite(store.get_user_by_username("admin")["id"], "p1")

    def test_patch_photo_read_role_forbidden(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_user("reader", "read", password_hash)
        _seed_done_photo("p1")
        headers = _bearer(client, "reader")
        assert client.patch("/api/photos/p1", json={"archived": True}, headers=headers).status_code == 403

    def test_tags_roundtrip(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", tags=["Dog"])
        headers = _bearer(client)
        r = client.put("/api/photos/p1/tags", json={"tags": ["Dog", "BEACH", "dog"]}, headers=headers)
        assert r.status_code == 200
        assert r.json()["tags"] == ["dog", "beach"]
        assert client.get("/api/photos/p1/tags", headers=headers).json()["tags"] == ["dog", "beach"]

    def test_all_tags_endpoint(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", tags=["dog"])
        _seed_done_photo("p2", tags=["dog", "cat"])
        headers = _bearer(client)
        r = client.get("/api/tags", headers=headers)
        assert r.status_code == 200
        tags = r.json()["tags"]
        assert tags[0] == {"name": "dog", "count": 2}
        assert {"name": "cat", "count": 1} in tags

    def test_memories(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        import datetime as _dt
        today = _dt.datetime.utcnow()
        last_year = today.replace(year=today.year - 1)
        _seed_done_photo("p1", capture_time=last_year.timestamp())
        headers = _bearer(client)
        r = client.get("/api/memories", headers=headers)
        assert r.status_code == 200
        assert r.json()["photos"][0]["uid"] == "p1"
        assert r.json()["photos"][0]["age_years"] >= 1

    def test_duplicates(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", sha1="same-hash")
        _seed_done_photo("p2", sha1="same-hash")
        headers = _bearer(client)
        r = client.get("/api/duplicates", headers=headers)
        group = r.json()["groups"][0]
        assert sorted(g["uid"] for g in group["photos"]) == ["p1", "p2"]

    def test_duplicates_favorited_by_me(self, client, password_hash):
        user_id = _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", sha1="same-hash")
        _seed_done_photo("p2", sha1="same-hash")
        store.favorite_photo(user_id, "p2")
        r = client.get("/api/duplicates", headers=_bearer(client))
        group = r.json()["groups"][0]
        by_uid = {p["uid"]: p["favorited_by_me"] for p in group["photos"]}
        assert by_uid == {"p1": False, "p2": True}

    def test_anchors(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", capture_time=1700000000)
        headers = _bearer(client)
        r = client.get("/api/photos/anchors", headers=headers)
        assert r.status_code == 200
        assert r.json()["anchors"][0]["ym"] == "2023-11"

    def test_thumb(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        r = client.get("/api/photos/p1/thumb", headers=_bearer(client))
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"

    def test_thumb_missing(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", thumb=False)
        assert client.get("/api/photos/p1/thumb", headers=_bearer(client)).status_code == 404

    def test_full_photo(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        r = client.get("/api/photos/p1/full", headers=_bearer(client))
        assert r.status_code == 200
        assert b"fake-jpeg" in r.content

    def test_full_photo_transient_error(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        monkeypatch.setattr(
            bridge_client, "_bridge",
            FailingFullBridge(bridge_client.BridgeTransientError(429, "rate limited", retry_after_sec=5)),
        )
        r = client.get("/api/photos/p1/full", headers=_bearer(client))
        assert r.status_code == 429
        assert r.headers.get("Retry-After") == "5"

    def test_full_photo_404(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        assert client.get("/api/photos/missing/full", headers=_bearer(client)).status_code == 404

    def test_meta(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", albums=["al1"])
        # Album names come from the local albums table; the fixture seeds the
        # store the way the 10-minute `sync_albums` loop would.
        store.sync_albums([{"uid": "al1", "name": "Holiday"}])
        fake = FakeBridge()
        fake.add_node("p1", tags=["proton-tag"])
        monkeypatch.setattr(bridge_client, "_bridge", fake)
        r = client.get("/api/photos/p1/meta", headers=_bearer(client))
        assert r.status_code == 200
        assert r.json()["proton_tags"] == ["proton-tag"]
        assert r.json()["albums_detail"] == [{"uid": "al1", "name": "Holiday"}]
        # The endpoint must resolve album names locally and never enumerate
        # the full album list from the bridge (issue #92).
        assert fake._albums_calls == 0

    def test_meta_unknown_album_uid_falls_back(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        # Album not synced locally yet: the name must fall back to the uid
        # rather than triggering a bridge album enumeration.
        _seed_done_photo("p1", albums=["al9"])
        fake = FakeBridge()
        fake.add_node("p1")
        monkeypatch.setattr(bridge_client, "_bridge", fake)
        r = client.get("/api/photos/p1/meta", headers=_bearer(client))
        assert r.status_code == 200
        assert r.json()["albums_detail"] == [{"uid": "al9", "name": "al9"}]
        assert fake._albums_calls == 0

    def test_albums(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", albums=["al1"])
        _seed_done_photo("p2", albums=["al1"])
        store.sync_albums([{"uid": "al1", "name": "Holiday"}])
        headers = _bearer(client)
        r = client.get("/api/albums", headers=headers)
        assert r.json()["albums"][0]["photo_count"] == 2
        r2 = client.get("/api/albums/al1/photos", headers=headers)
        assert len(r2.json()["photos"]) == 2

    def test_places(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", place="Paris, France", gps=(48.8584, 2.2945))
        headers = _bearer(client)
        r = client.get("/api/places", headers=headers)
        assert r.json()["places"][0]["city"] == "Paris"

    def test_map(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", place="Paris, France", gps=(48.8584, 2.2945))
        headers = _bearer(client)
        r = client.get("/api/map", headers=headers)
        marker = r.json()["markers"][0]
        assert marker["lat"] == 48.8584
        assert marker["thumb_url"].split("?")[0] == "/api/photos/p1/thumb"
        assert "sig=" in marker["thumb_url"]  # signed by default when public thumbs are off

    def test_thumb_url_stable_within_hour(self, client, password_hash):
        # Regression for signed-URL `exp` churn: two page loads within the same
        # hour must return byte-identical thumb URLs, otherwise the
        # `Cache-Control: immutable` headers on the binary endpoints are
        # defeated and every session re-downloads every thumbnail.
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", place="Paris, France", gps=(48.8584, 2.2945))
        headers = _bearer(client)
        url1 = client.get("/api/map", headers=headers).json()["markers"][0]["thumb_url"]
        url2 = client.get("/api/map", headers=headers).json()["markers"][0]["thumb_url"]
        assert url1 == url2
        assert "sig=" in url1 and "exp=" in url1


# --- faces / people -------------------------------------------------------


class TestFacesAndPeople:
    def test_people_list(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        pid = store.create_person(name="Alice", cover_uid="p1", cover_face_id=face_id)
        store.assign_face_person(face_id, pid)
        headers = _bearer(client)
        r = client.get("/api/people", headers=headers)
        assert r.status_code == 200
        people = r.json()["people"]
        assert people[0]["name"] == "Alice"
        assert people[0]["face_count"] == 1
        assert people[0]["cover_url"].split("?")[0] == "/api/people/1/cover"
        assert "sig=" in people[0]["cover_url"]  # signed by default when public thumbs are off

    def test_people_filtered(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        pid = store.create_person(name="Alice", cover_uid="p1", cover_face_id=face_id)
        store.assign_face_person(face_id, pid)
        headers = _bearer(client)
        r = client.get("/api/people", params={"q": "alic"}, headers=headers)
        assert r.json()["total"] == 1
        r2 = client.get("/api/people", params={"q": "zzz"}, headers=headers)
        assert r2.json()["total"] == 0

    def test_person_cover_thumb_fallback(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        store.create_person(name="Alice", cover_uid="p1", cover_face_id=None)
        r = client.get("/api/people/1/cover", headers=_bearer(client))
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"

    def test_person_cover_404(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        assert client.get("/api/people/999/cover", headers=_bearer(client)).status_code == 404

    def test_person_faces(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        pid = store.create_person(name="Alice", cover_uid="p1", cover_face_id=face_id)
        store.assign_face_person(face_id, pid)
        headers = _bearer(client)
        r = client.get(f"/api/people/{pid}/faces", headers=headers)
        assert r.status_code == 200
        assert r.json()["faces"][0]["id"] == face_id

    def test_person_set_cover(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        other = store.insert_face(photo_uid="p1", person_id=None, confidence=0.99,
                                  bbox=(0.2, 0.2, 0.3, 0.3), embedding=_emb(2).tobytes())
        pid = store.create_person(name="Alice", cover_uid="p1", cover_face_id=None)
        store.assign_face_person(face_id, pid)
        headers = _bearer(client)
        # The endpoint shares its /cover suffix with an auth-free binary route —
        # a valid bearer token must still be honored and a write role enforced.
        r = client.post(f"/api/people/{pid}/cover", json={"face_id": face_id}, headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["cover_face_id"] == face_id
        assert store.get_person(pid)["cover_face_id"] == face_id
        # A face belonging to another person is rejected.
        assert client.post(f"/api/people/{pid}/cover", json={"face_id": other}, headers=headers).status_code == 400
        # Anonymous writes still fail closed with 401.
        assert client.post(f"/api/people/{pid}/cover", json={"face_id": face_id}).status_code == 401

    def test_person_photos(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        pid = store.create_person(name="Alice", cover_uid="p1", cover_face_id=face_id)
        store.assign_face_person(face_id, pid)
        headers = _bearer(client)
        r = client.get(f"/api/people/{pid}/photos", headers=headers)
        assert [p["uid"] for p in r.json()["photos"]] == ["p1"]

    def test_unassigned_faces(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        headers = _bearer(client)
        r = client.get("/api/faces/unassigned", headers=headers)
        assert r.json()["faces"][0]["id"] == face_id

    def test_photo_faces(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        _seed_face("p1")
        headers = _bearer(client)
        r = client.get("/api/photos/p1/faces", headers=headers)
        assert len(r.json()["faces"]) == 1

    def test_face_crop(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1", bbox=(0.2, 0.2, 0.4, 0.4))
        r = client.get(f"/api/faces/{face_id}/crop", headers=_bearer(client))
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"


class TestBinaryEndpointAuth:
    """Binary endpoints (/thumb /full /cover /crop) are secure by default."""

    def test_thumb_requires_auth_by_default(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        r = client.get("/api/photos/p1/thumb")
        assert r.status_code == 401

    def test_full_requires_auth_by_default(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        r = client.get("/api/photos/p1/full")
        assert r.status_code == 401

    def test_thumb_public_when_opt_in(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        monkeypatch.setenv("DEMO_ALLOW_PUBLIC_THUMBS", "1")
        r = client.get("/api/photos/p1/thumb")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"

    def test_thumb_ok_with_bearer(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        r = client.get("/api/photos/p1/thumb", headers=_bearer(client))
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"

    def test_thumb_ok_with_signed_url(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        sig, exp = auth.make_signed_token("/api/photos/p1/thumb", ttl_seconds=300)
        r = client.get(f"/api/photos/p1/thumb?sig={sig}&exp={exp}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"

    def test_thumb_ok_with_signed_url_when_opt_in(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        monkeypatch.setenv("DEMO_ALLOW_PUBLIC_THUMBS", "1")
        sig, exp = auth.make_signed_token("/api/photos/p1/thumb", ttl_seconds=300)
        r = client.get(f"/api/photos/p1/thumb?sig={sig}&exp={exp}")
        assert r.status_code == 200

    def test_signed_url_via_api_sign_endpoint(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        r = client.post(
            "/api/sign",
            json={"paths": ["/api/photos/p1/thumb"]},
            headers=_bearer(client),
        )
        assert r.status_code == 200, r.text
        url = r.json()["urls"][0]
        signed = f"/api/photos/p1/thumb?sig={url['sig']}&exp={url['exp']}"
        assert client.get(signed).status_code == 200

    def test_face_assign_to_person(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        pid = store.create_person(name="Alice", cover_uid="p1", cover_face_id=None)
        headers = _bearer(client)
        r = client.post(f"/api/faces/{face_id}/person", json={"person_id": pid}, headers=headers)
        assert r.status_code == 200
        assert r.json()["person_id"] == pid
        assert store.get_person(pid)["face_count"] == 1

    def test_face_assign_by_name(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        headers = _bearer(client)
        r = client.post(f"/api/faces/{face_id}/person", json={"name": "Bob"}, headers=headers)
        assert r.status_code == 200
        assert r.json()["merged"] is False
        pid = r.json()["person_id"]
        assert store.get_person(pid)["name"] == "Bob"

    def test_face_assign_requires_body(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        headers = _bearer(client)
        assert client.post(f"/api/faces/{face_id}/person", json={}, headers=headers).status_code == 400

    def test_face_unassign(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        pid = store.create_person(name="Alice", cover_uid="p1", cover_face_id=face_id)
        store.assign_face_person(face_id, pid)
        headers = _bearer(client)
        r = client.post(f"/api/faces/{face_id}/unassign", headers=headers)
        assert r.status_code == 200
        assert store.get_person(pid)["face_count"] == 0

    def test_rename_person(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        face_id = _seed_face("p1")
        pid = store.create_person(name="Alice", cover_uid="p1", cover_face_id=face_id)
        store.assign_face_person(face_id, pid)
        headers = _bearer(client)
        r = client.post(f"/api/people/{pid}/name", json={"name": "Alicia"}, headers=headers)
        assert r.status_code == 200
        assert r.json()["merged"] is False
        assert store.get_person(pid)["name"] == "Alicia"

    def test_rename_merges_existing(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        _seed_done_photo("p2")
        fa = _seed_face("p1")
        fb = _seed_face("p2")
        pa = store.create_person(name="Alice", cover_uid="p1", cover_face_id=fa)
        pb = store.create_person(name="Alicia", cover_uid="p2", cover_face_id=fb)
        store.assign_face_person(fa, pa)
        store.assign_face_person(fb, pb)
        headers = _bearer(client)
        r = client.post(f"/api/people/{pa}/name", json={"name": "Alicia"}, headers=headers)
        assert r.status_code == 200
        assert r.json()["merged"] is True
        assert r.json()["target_id"] == pb
        assert store.get_person(pa) is None
        assert store.get_person(pb)["face_count"] == 2

    def test_merge_person(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        _seed_done_photo("p2")
        fa = _seed_face("p1")
        fb = _seed_face("p2")
        pa = store.create_person(name="Alice", cover_uid="p1", cover_face_id=fa)
        pb = store.create_person(name="Bob", cover_uid="p2", cover_face_id=fb)
        store.assign_face_person(fa, pa)
        store.assign_face_person(fb, pb)
        headers = _bearer(client)
        r = client.post(f"/api/people/{pa}/merge", json={"target_id": pb}, headers=headers)
        assert r.status_code == 200
        assert store.get_person(pa) is None
        assert store.get_person(pb)["face_count"] == 2

    def test_merge_person_self(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        fa = _seed_face("p1")
        pa = store.create_person(name="Alice", cover_uid="p1", cover_face_id=fa)
        headers = _bearer(client)
        assert client.post(f"/api/people/{pa}/merge", json={"target_id": pa}, headers=headers).status_code == 400

    def test_people_duplicates(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        fa = _seed_face("p1", emb=_emb(10))
        pa = store.create_person(name="Alice", cover_uid="p1", cover_face_id=fa)
        store.assign_face_person(fa, pa)
        headers = _bearer(client)
        r = client.get("/api/people/duplicates", headers=headers)
        assert r.status_code == 200
        assert "duplicates" in r.json()

    def test_person_similar(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        fa = _seed_face("p1", emb=_emb(10))
        pa = store.create_person(name="Alice", cover_uid="p1", cover_face_id=fa)
        store.assign_face_person(fa, pa)
        headers = _bearer(client)
        r = client.get(f"/api/people/{pa}/similar", headers=headers)
        assert r.status_code == 200
        assert r.json()["similar"] == []

    def test_merge_all(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        _seed_done_photo("p2")
        fa = _seed_face("p1")
        fb = _seed_face("p2")
        pa = store.create_person(name="Alice", cover_uid="p1", cover_face_id=fa)
        pb = store.create_person(name="Bob", cover_uid="p2", cover_face_id=fb)
        store.assign_face_person(fa, pa)
        store.assign_face_person(fb, pb)
        headers = _bearer(client)
        r = client.post(f"/api/people/{pa}/merge_all", json={"source_ids": [pb]}, headers=headers)
        assert r.status_code == 200
        assert r.json()["merged_count"] == 1
        assert store.get_person(pb) is None

    def test_merge_all_similar(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        # target + 3 look-alikes sharing identical embeddings (cosine == 1.0),
        # so one call should merge all of them above any reasonable threshold.
        for i in range(4):
            _seed_done_photo(f"p{i}")
        fa = _seed_face("p0", emb=_emb(10))
        pa = store.create_person(name="Alice", cover_uid="p0", cover_face_id=fa)
        store.assign_face_person(fa, pa)
        source_ids = []
        for i in range(1, 4):
            f = _seed_face(f"p{i}", emb=_emb(10))
            pid = store.create_person(name=None, cover_uid=f"p{i}", cover_face_id=f)
            store.assign_face_person(f, pid)
            source_ids.append(pid)
        headers = _bearer(client)
        r = client.post(f"/api/people/{pa}/merge_all_similar",
                        json={"threshold": 0.40}, headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["target_id"] == pa
        assert body["merged_count"] == 3
        assert body["face_count"] == 4
        for pid in source_ids:
            assert store.get_person(pid) is None

    def test_merge_all_similar_none(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        fa = _seed_face("p1", emb=_emb(10))
        pa = store.create_person(name="Alice", cover_uid="p1", cover_face_id=fa)
        store.assign_face_person(fa, pa)
        headers = _bearer(client)
        r = client.post(f"/api/people/{pa}/merge_all_similar",
                        json={"threshold": 0.99}, headers=headers)
        assert r.status_code == 200
        assert r.json()["merged_count"] == 0

    def test_people_duplicates_multiple(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        # 4 identical people should be reported as duplicate pairs.
        for i in range(4):
            _seed_done_photo(f"p{i}")
            f = _seed_face(f"p{i}", emb=_emb(10))
            pid = store.create_person(name=None, cover_uid=f"p{i}", cover_face_id=f)
            store.assign_face_person(f, pid)
        headers = _bearer(client)
        r = client.get("/api/people/duplicates", params={"threshold": 0.40}, headers=headers)
        assert r.status_code == 200
        dups = r.json()["duplicates"]
        assert len(dups) >= 3  # 4 identical people → C(4,2)=6 pairs, all above threshold
        assert all(d["similarity"] > 0.99 for d in dups)

    def test_face_suggest(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        _seed_done_photo("p2")
        fa = _seed_face("p1", emb=_emb(10))
        fb = _seed_face("p2", emb=_emb(10))
        pa = store.create_person(name="Alice", cover_uid="p1", cover_face_id=fa)
        store.assign_face_person(fa, pa)
        headers = _bearer(client)
        r = client.get(f"/api/faces/{fb}/suggest", headers=headers)
        assert r.status_code == 200
        assert r.json()["suggestions"][0]["person_id"] == pa


# --- search ---------------------------------------------------------------


class TestSearch:
    def test_search(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        store.insert_clip("p1", _emb(5).tobytes())
        monkeypatch.setattr(api, "embed_text", lambda text: _emb(5))
        headers = _bearer(client)
        r = client.get("/api/search", params={"q": "beach"}, headers=headers)
        assert r.status_code == 200
        results = r.json()["results"]
        assert results[0]["uid"] == "p1"
        assert results[0]["score"] > 0.9

    def test_search_empty_query(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        headers = _bearer(client)
        assert client.get("/api/search", params={"q": "  "}, headers=headers).status_code == 400

    def test_search_clip_failure(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        monkeypatch.setattr(api, "embed_text", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
        headers = _bearer(client)
        assert client.get("/api/search", params={"q": "beach"}, headers=headers).status_code == 503

    def test_search_empty_matrix(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        monkeypatch.setattr(api, "embed_text", lambda text: _emb(5))
        headers = _bearer(client)
        r = client.get("/api/search", params={"q": "beach"}, headers=headers)
        assert r.json() == {"results": [], "total": 0}

    def test_search_limit_clamped(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        for i in range(3):
            _seed_done_photo(f"p{i}")
            store.insert_clip(f"p{i}", _emb(i).tobytes())
        monkeypatch.setattr(api, "embed_text", lambda text: _emb(5))
        headers = _bearer(client)
        r = client.get("/api/search", params={"q": "beach", "limit": 10**6}, headers=headers)
        assert r.status_code == 200
        assert len(r.json()["results"]) == 3
        r = client.get("/api/search", params={"q": "beach", "limit": 0}, headers=headers)
        assert r.status_code == 200
        assert len(r.json()["results"]) == 1

    def test_search_face_limit_clamped(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        for i in range(3):
            _seed_done_photo(f"p{i}")
            _seed_face(f"p{i}", emb=_emb(i))
        monkeypatch.setattr(api, "embed_query_face", lambda bgr: _emb(5))
        headers = _bearer(client)
        r = client.post("/api/search/face", files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
                        params={"limit": 10**6}, headers=headers)
        assert r.status_code == 200
        assert len(r.json()["results"]) == 3
        r = client.post("/api/search/face", files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
                        params={"limit": 0}, headers=headers)
        assert r.status_code == 200
        assert len(r.json()["results"]) == 1

    def test_search_face(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        fa = _seed_face("p1", emb=_emb(10))
        pid = store.create_person(name="Alice", cover_uid="p1", cover_face_id=fa)
        store.assign_face_person(fa, pid)
        monkeypatch.setattr(api, "embed_query_face", lambda bgr: _emb(10))
        headers = _bearer(client)
        r = client.post("/api/search/face", files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
                        headers=headers)
        assert r.status_code == 200
        results = r.json()["results"]
        assert results[0]["uid"] == "p1"

    def test_search_face_no_face(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        monkeypatch.setattr(api, "embed_query_face", lambda bgr: None)
        headers = _bearer(client)
        r = client.post("/api/search/face", files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
                        headers=headers)
        assert r.status_code == 404


# --- admin ----------------------------------------------------------------


class TestAdmin:
    def _seed_admin(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        return _bearer(client)

    def test_admin_users_list(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        r = client.get("/api/admin/users", headers=headers)
        assert r.status_code == 200
        assert r.json()["users"][0]["username"] == "admin"

    def test_admin_users_read_role_forbidden(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_user("reader", "read", password_hash)
        headers = _bearer(client, "reader")
        assert client.get("/api/admin/users", headers=headers).status_code == 403

    def test_admin_create_user(self, client, monkeypatch, password_hash):
        headers = self._seed_admin(client, password_hash)
        monkeypatch.setattr(api, "hash_password", lambda pw: f"h:{pw}")
        r = client.post("/api/admin/users", json={"username": "bob", "password": "password123",
                                                  "role": "write"}, headers=headers)
        assert r.status_code == 200
        assert r.json()["user"]["role"] == "write"

    def test_admin_create_user_validation(self, client, monkeypatch, password_hash):
        headers = self._seed_admin(client, password_hash)
        assert client.post("/api/admin/users", json={"username": "x", "password": "password123"},
                           headers=headers).status_code == 400
        assert client.post("/api/admin/users", json={"username": "bob", "password": "short"},
                           headers=headers).status_code == 400
        assert client.post("/api/admin/users", json={"username": "bob", "password": "password123",
                                                     "role": "superuser"}, headers=headers).status_code == 400

    def test_admin_update_user(self, client, monkeypatch, password_hash):
        headers = self._seed_admin(client, password_hash)
        uid = _seed_user("bob", "read", password_hash)
        monkeypatch.setattr(api, "hash_password", lambda pw: f"h:{pw}")
        r = client.patch(f"/api/admin/users/{uid}", json={"role": "write", "disabled": True},
                         headers=headers)
        assert r.status_code == 200
        assert r.json()["user"]["disabled"] is True

    def test_admin_update_user_404(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        assert client.patch("/api/admin/users/999", json={"role": "write"}, headers=headers).status_code == 404

    def test_admin_delete_user(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        uid = _seed_user("bob", "read", password_hash)
        r = client.delete(f"/api/admin/users/{uid}", headers=headers)
        assert r.status_code == 200
        assert store.get_user_by_id(uid) is None

    def test_admin_cannot_delete_last_admin(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        admin_uid = store.get_user_by_username("admin")["id"]
        assert client.delete(f"/api/admin/users/{admin_uid}", headers=headers).status_code == 400

    def test_admin_revoke_user_tokens(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        uid = _seed_user("bob", "read", password_hash)
        token = store.issue_token(uid, "access", 3600, user_agent="t", ip="1.2.3.4")
        r = client.post(f"/api/admin/users/{uid}/logout", headers=headers)
        assert r.status_code == 200
        assert r.json()["revoked"] == 1
        assert store.lookup_token(token) is None

    def test_admin_overview(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        r = client.get("/api/admin/overview", headers=headers)
        assert r.status_code == 200
        assert "server" in r.json()

    def test_admin_backup(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        r = client.post("/api/admin/backup", headers=headers)
        assert r.status_code == 200
        assert r.json()["name"].startswith("index-")

    def test_admin_backups_list(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        admin.snapshot_backup()
        r = client.get("/api/admin/backups", headers=headers)
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_admin_backups_delete(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        res = admin.snapshot_backup()
        r = client.delete(f"/api/admin/backups/{res['name']}", headers=headers)
        assert r.status_code == 200
        assert client.get("/api/admin/backups", headers=headers).json() == []

    def test_admin_backups_prune(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        admin.snapshot_backup()
        r = client.post("/api/admin/backups/prune", json={"keep": 5}, headers=headers)
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_admin_schedule_get_set(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        r = client.put("/api/admin/schedule", json={"enabled": True, "hour": 3, "minute": 30, "keep": 7},
                       headers=headers)
        assert r.status_code == 200
        assert r.json()["enabled"] is True
        r2 = client.get("/api/admin/schedule", headers=headers)
        assert r2.json()["keep"] == 7

    def test_admin_sync(self, client, monkeypatch, password_hash):
        headers = self._seed_admin(client, password_hash)
        monkeypatch.setattr(
            api, "_indexer_proxy_json",
            lambda method, path, body=None: {"last_sync": 1, "threads": {}, "config": {}},
        )
        assert client.get("/api/admin/sync", headers=headers).status_code == 200
        assert client.post("/api/admin/sync/trigger", headers=headers).status_code == 200
        assert client.put("/api/admin/sync", json={"tip_size": 20}, headers=headers).status_code == 200

    def test_admin_checks(self, client, monkeypatch, password_hash):
        headers = self._seed_admin(client, password_hash)
        admin.snapshot_backup()
        monkeypatch.setattr(admin, "_bridge_reachability",
                            lambda: {"name": "Bridge reachable", "ok": True, "status": "ok", "detail": ""})
        monkeypatch.setattr(admin, "_bridge_cache_health",
                            lambda recent_full_res_failures=0: {"name": "Bridge cache", "ok": True,
                                                                "status": "ok", "detail": ""})
        r = client.post("/api/admin/checks", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["passed"] == body["total"]
        assert body["total"] == 8

    def test_admin_gc_empty_people(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        with store.get_conn() as conn:
            conn.execute("UPDATE people SET created=1000 WHERE id=?",
                         (store.create_person(None, None, None),))
        r = client.post("/api/admin/people/gc-empty", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["deleted"] == 1
        # idempotent: nothing left to sweep
        assert client.post("/api/admin/people/gc-empty", headers=headers).json()["deleted"] == 0

    def test_admin_gc_empty_people_read_role_forbidden(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_user("reader", "read", password_hash)
        headers = _bearer(client, "reader")
        assert client.post("/api/admin/people/gc-empty", headers=headers).status_code == 403

    def test_admin_bridge_cache(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        assert client.get("/api/admin/bridge/cache", headers=headers).status_code == 200
        assert client.post("/api/admin/bridge/cache/clear", headers=headers).status_code == 200

    def test_admin_schedule_read_role_forbidden(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_user("reader", "read", password_hash)
        headers = _bearer(client, "reader")
        assert client.get("/api/admin/schedule", headers=headers).status_code == 403


class TestDemoDisableAdminUserManagement:
    """Mirrors the demo_disable_backups pattern: 404 every /api/admin/users*
    route when the flag is on, regardless of admin auth."""

    _ENDPOINTS = [
        ("GET",   "/api/admin/users",                None),
        ("POST",  "/api/admin/users",                {"username": "alice", "password": "password123"}),
        ("PATCH", "/api/admin/users/1",              {"role": "write"}),
        ("DELETE","/api/admin/users/1",              None),
        ("POST",  "/api/admin/users/1/logout",       None),
    ]

    def _seed_admin(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        return _bearer(client)

    def test_endpoints_work_when_flag_off(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        # Sanity: with the flag at its default, the routes work as before.
        assert client.get("/api/admin/users", headers=headers).status_code == 200

    def test_endpoints_404_when_flag_on(self, client, monkeypatch, password_hash):
        # Patch the symbol on api.py — that's the reference the route body
        # captures at import time.
        monkeypatch.setattr(api, "demo_disable_admin_user_management", lambda: True)
        headers = self._seed_admin(client, password_hash)
        for method, path, json_body in self._ENDPOINTS:
            method_fn = getattr(client, method.lower())
            kwargs = {"headers": headers}
            if json_body is not None:
                kwargs["json"] = json_body
            r = method_fn(path, **kwargs)
            assert r.status_code == 404, (method, path, r.status_code, r.text)

    def test_endpoints_404_under_hardening_mode(self, client, monkeypatch, password_hash):
        # When DEMO_HARDENING_MODE is on and the env var is unset, the
        # hardening override flips the flag to its safe value (True).
        monkeypatch.delenv("DEMO_DISABLE_ADMIN_USER_MANAGEMENT", raising=False)
        monkeypatch.setenv("DEMO_HARDENING_MODE", "1")
        headers = self._seed_admin(client, password_hash)
        r = client.get("/api/admin/users", headers=headers)
        assert r.status_code == 404

    def test_explicit_opt_out_wins_over_hardening_mode(self, client, monkeypatch, password_hash):
        # An explicit DEMO_DISABLE_ADMIN_USER_MANAGEMENT=0 keeps the routes
        # accessible even when hardening mode is on.
        monkeypatch.setenv("DEMO_HARDENING_MODE", "1")
        monkeypatch.setenv("DEMO_DISABLE_ADMIN_USER_MANAGEMENT", "0")
        headers = self._seed_admin(client, password_hash)
        r = client.get("/api/admin/users", headers=headers)
        assert r.status_code == 200

    def test_self_routes_still_work_when_flag_on(self, client, monkeypatch, password_hash):
        # The fix only gates *user management*. Self-service routes like
        # /api/auth/me and /api/auth/logout must still work so an admin
        # who's already logged in can inspect their own account and sign
        # out of their own devices.
        monkeypatch.setattr(api, "demo_disable_admin_user_management", lambda: True)
        headers = self._seed_admin(client, password_hash)
        assert client.get("/api/auth/me", headers=headers).status_code == 200
        assert client.post("/api/auth/logout", headers=headers).status_code == 200


class TestCompression:
    """HTTP compression middleware: gzip JSON/UI responses, skip binary media."""

    def test_json_compressed(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        for i in range(5):
            _seed_done_photo(f"p{i}", capture_time=1700000000 + i)
        headers = _bearer(client)
        headers["Accept-Encoding"] = "gzip"
        r = client.get("/api/photos", headers=headers)
        assert r.status_code == 200
        assert r.headers.get("content-encoding") == "gzip"
        assert "accept-encoding" in r.headers.get("vary", "").lower()
        assert len(r.json()["photos"]) == 5

    def test_binary_not_compressed(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        headers = _bearer(client)
        headers["Accept-Encoding"] = "gzip"
        r = client.get("/api/photos/p1/thumb", headers=headers)
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"
        assert "content-encoding" not in r.headers

    def test_small_json_passthrough(self, client):
        r = client.get("/api/auth/limits", headers={"Accept-Encoding": "gzip"})
        assert r.status_code == 200
        assert r.json() == {"min_username": 2, "min_password": 8}
        assert "content-encoding" not in r.headers

    def test_no_accept_encoding_no_compression(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        headers = _bearer(client)
        headers["Accept-Encoding"] = "identity"
        r = client.get("/api/photos", headers=headers)
        assert r.status_code == 200
        assert "content-encoding" not in r.headers

    def test_streaming_response_compressed(self):
        """Streaming JSON responses are gzipped chunk-by-chunk."""
        import asyncio

        from compression import CompressionMiddleware

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'{"a": ' + b"1" * 2000, "more_body": True})
            await send({"type": "http.response.body", "body": b"}", "more_body": False})

        messages = []

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "headers": [(b"accept-encoding", b"gzip")]}
        asyncio.run(CompressionMiddleware(app, minimum_size=1)(scope, None, send))
        start = messages[0]
        headers = dict(start["headers"])
        assert headers[b"content-encoding"] == b"gzip"
        assert b"content-length" not in headers
        assert b"accept-encoding" in headers[b"vary"].lower()
        # Both body chunks were compressed.
        assert messages[1]["body"] != b'{"a": ' + b"1" * 2000
        assert messages[2]["body"] != b"}"

    def test_pathsend_passthrough(self):
        """pathsend responses are forwarded uncompressed."""
        import asyncio

        from compression import CompressionMiddleware

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.pathsend", "path": "/tmp/x"})

        messages = []

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "headers": [(b"accept-encoding", b"gzip")]}
        asyncio.run(CompressionMiddleware(app, minimum_size=1)(scope, None, send))
        assert messages[0]["type"] == "http.response.start"
        assert messages[1]["type"] == "http.response.pathsend"
        assert b"content-encoding" not in dict(messages[0]["headers"])


class TestTTLCacheSingleFlight:
    """TTL caches use double-checked locking: when N threads hit an expired
    cache at once, exactly one of them runs the expensive compute."""

    def _run_concurrent(self, fn, n: int = 20) -> None:
        barrier = threading.Barrier(n)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=10)
                fn()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == []

    def test_cached_stats_single_flight(self, monkeypatch):
        api._stats_cache = (time.time() - 1000, {})
        calls = 0

        def fake_stats() -> dict:
            nonlocal calls
            calls += 1
            time.sleep(0.05)
            return {"ok": True}

        monkeypatch.setattr(api, "stats", fake_stats)
        self._run_concurrent(api._cached_stats)
        assert calls == 1
        assert api._stats_cache[1] == {"ok": True}

    def test_cached_bridge_health_single_flight(self, monkeypatch):
        api._bridge_health_cache = (time.time() - 1000, (False, False))
        calls = 0

        class FakeBridge:
            def health(self):
                nonlocal calls
                calls += 1
                time.sleep(0.05)
                return {"ok": True, "loggedIn": True}

        monkeypatch.setattr(api, "get_bridge", lambda: FakeBridge())
        self._run_concurrent(api._cached_bridge_health)
        assert calls == 1
        assert api._bridge_health_cache[1] == (True, True)

    def test_people_all_cached_single_flight(self, monkeypatch):
        api._people_cache = (None, time.time() - 1000, [])
        calls = 0

        def fake_all_people(q=None):
            nonlocal calls
            calls += 1
            time.sleep(0.05)
            return []

        monkeypatch.setattr(api, "all_people", fake_all_people)
        self._run_concurrent(api._people_all_cached)
        assert calls == 1
        assert api._people_cache is not None and api._people_cache[0] is None

    def test_people_filtered_single_flight(self, monkeypatch):
        api._people_cache = ("foo", time.time() - 1000, [])
        calls = 0

        def fake_all_people(q=None):
            nonlocal calls
            calls += 1
            time.sleep(0.05)
            return []

        monkeypatch.setattr(api, "all_people", fake_all_people)
        self._run_concurrent(lambda: api.api_people(limit=10, offset=0, q="foo"))
        assert calls == 1
        assert api._people_cache is not None and api._people_cache[0] == "foo"

    def test_duplicates_single_flight(self, monkeypatch):
        api._dups_cache = (time.time() - 1000, {})
        people = [
            {"id": 1, "name": "A", "cover_uid": "p1", "cover_face_id": None,
             "face_count": 1, "photo_count": 1, "cover_url": None},
            {"id": 2, "name": "B", "cover_uid": "p2", "cover_face_id": None,
             "face_count": 1, "photo_count": 1, "cover_url": None},
        ]
        calls = 0
        monkeypatch.setattr(api, "_people_all_cached", lambda: people)

        def fake_means():
            nonlocal calls
            calls += 1
            time.sleep(0.05)
            return {1: _emb(0), 2: np.zeros(512, dtype=np.float32)}

        monkeypatch.setattr(api, "person_mean_embeddings_from_cache", fake_means)
        self._run_concurrent(api.api_people_duplicates)
        assert calls == 1
        assert api._dups_cache[1] == {"duplicates": []}

    def test_photo_duplicates_single_flight(self, monkeypatch):
        api._photo_dups_cache = None
        calls = 0

        def fake_duplicate_groups(limit=500):
            nonlocal calls
            calls += 1
            time.sleep(0.05)
            return []

        monkeypatch.setattr(api, "duplicate_groups", fake_duplicate_groups)
        self._run_concurrent(lambda: api._duplicate_groups_cached(200))
        assert calls == 1
        assert list(api._photo_dups_cache[1]) == [200]
        # page sizes are keyed separately, so a new limit recomputes once and
        # later hits are served from the cache.
        assert api._duplicate_groups_cached(500) is api._photo_dups_cache[1][500]
        assert calls == 2
        assert api._duplicate_groups_cached(200) is api._photo_dups_cache[1][200]
        assert calls == 2

    def test_clip_matrix_single_flight(self, monkeypatch):
        api._clip_cache = (time.time() - 1000, 1, ["p1"], np.zeros((1, 512), dtype=np.float32))
        sidecar_calls = 0

        def fake_clip_count() -> int:
            return 2

        def fake_sidecar():
            nonlocal sidecar_calls
            sidecar_calls += 1
            time.sleep(0.05)
            return ["p1", "p2"], np.zeros((2, 512), dtype=np.float32)

        monkeypatch.setattr(api, "clip_count", fake_clip_count)
        monkeypatch.setattr(api, "read_clip_sidecar", fake_sidecar)
        self._run_concurrent(api._get_clip_matrix)
        assert sidecar_calls == 1

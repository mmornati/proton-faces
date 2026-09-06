"""TestClient coverage of the FastAPI app in app/src/api.py.

The app has no factory: `api.app` is built at import time with a global
`require_user` dependency. We exercise it through a real TestClient against
a tmp sqlite DB, with `bridge_client._bridge` swapped for a fake bridge and
module-level ML functions (embed_text / embed_query_face) monkeypatched.
"""

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

    def close(self) -> None:
        pass


class FakeBridge:
    """Minimal in-repo fake for the Proton bridge HTTP surface."""

    def __init__(self):
        self._nodes = {}
        self._albums = []
        self._full_data = b"\xff\xd8\xfffake-jpeg"

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
        return {"albums": self._albums}

    def full_photo(self, uid, range_header=None, timeout_ms=None):
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


@pytest.fixture(scope="session")
def password_hash():
    return auth.hash_password("password123")


@pytest.fixture
def client(tmp_db, monkeypatch):
    monkeypatch.setattr(bridge_client, "_bridge", FakeBridge())
    with TestClient(api.app) as c:
        yield c


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
        r = client.get("/api/photos/p1/thumb")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"

    def test_thumb_missing(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", thumb=False)
        assert client.get("/api/photos/p1/thumb").status_code == 404

    def test_full_photo(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        r = client.get("/api/photos/p1/full")
        assert r.status_code == 200
        assert b"fake-jpeg" in r.content

    def test_full_photo_transient_error(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1")
        monkeypatch.setattr(
            bridge_client, "_bridge",
            FailingFullBridge(bridge_client.BridgeTransientError(429, "rate limited", retry_after_sec=5)),
        )
        r = client.get("/api/photos/p1/full")
        assert r.status_code == 429
        assert r.headers.get("Retry-After") == "5"

    def test_full_photo_404(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        assert client.get("/api/photos/missing/full").status_code == 404

    def test_meta(self, client, monkeypatch, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_done_photo("p1", albums=["al1"])
        fake = FakeBridge()
        fake.add_node("p1", tags=["proton-tag"])
        fake.add_album("al1", "Holiday")
        monkeypatch.setattr(bridge_client, "_bridge", fake)
        r = client.get("/api/photos/p1/meta", headers=_bearer(client))
        assert r.status_code == 200
        assert r.json()["proton_tags"] == ["proton-tag"]
        assert r.json()["albums_detail"] == [{"uid": "al1", "name": "Holiday"}]

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
        assert marker["thumb_url"] == "/api/photos/p1/thumb"


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
        assert people[0]["cover_url"] == "/api/people/1/cover"

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
        r = client.get("/api/people/1/cover")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"

    def test_person_cover_404(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        assert client.get("/api/people/999/cover").status_code == 404

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
        r = client.get(f"/api/faces/{face_id}/crop")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"

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

    def test_admin_bridge_cache(self, client, password_hash):
        headers = self._seed_admin(client, password_hash)
        assert client.get("/api/admin/bridge/cache", headers=headers).status_code == 200
        assert client.post("/api/admin/bridge/cache/clear", headers=headers).status_code == 200

    def test_admin_schedule_read_role_forbidden(self, client, password_hash):
        _seed_user(password_hash=password_hash)
        _seed_user("reader", "read", password_hash)
        headers = _bearer(client, "reader")
        assert client.get("/api/admin/schedule", headers=headers).status_code == 403

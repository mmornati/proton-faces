"""Unit coverage for app/src/api_common.py: the cross-router pure(ish)
helpers extracted from the api.py monolith (row serialization, image
sniffing, the duplicate/suggested-merge similarity math, the bridge-health
and indexer-proxy caches, and the crop pipeline).

These call api_common's functions directly rather than going through
TestClient + the DB, monkeypatching the handful of collaborators each
function actually needs (mirroring the ``api.<name>`` vs. bare-name call
convention api_common.py itself documents — some helpers resolve
dependencies through ``api.<name>`` at call time, others through their own
module-local name, and which one matters for where you patch).
"""


import httpx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

import api
import api_common

# --- _sniff_image_type ------------------------------------------------------

@pytest.mark.parametrize("data, expected", [
    (b"\xff\xd8\xff" + b"rest", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n" + b"rest", "image/png"),
    (b"GIF89a" + b"rest", "image/gif"),
    (b"GIF87a" + b"rest", "image/gif"),
    (b"RIFF____WEBPrest", "image/webp"),
    (b"____ftypheic" + b"rest", "image/heic"),
    (b"____ftypmp42" + b"rest", "video/mp4"),
    (b"\x1a\x45\xdf\xa3" + b"rest", "video/webm"),
    (b"not a media file at all", None),
])
def test_sniff_image_type(data, expected):
    assert api_common._sniff_image_type(data) == expected


# --- _row_to_dict ------------------------------------------------------------

class TestRowToDict:
    def _row(self, **overrides):
        row = {
            "uid": "p1",
            "thumb_path": None,
            "media_type": "image/jpeg",
            "favorited": 0,
            "archived": 0,
            "hidden": 0,
            "tags": None,
        }
        row.update(overrides)
        return row

    def test_drops_embedding_and_signs_thumb_url(self, monkeypatch):
        monkeypatch.setattr(api_common, "allow_public_thumbs", lambda: True)
        d = api_common._row_to_dict(self._row(thumb_path="p1.webp", embedding=b"raw"))
        assert "embedding" not in d
        assert d["thumb_url"] == "/api/photos/p1/thumb"

    def test_no_thumb_path_means_no_thumb_url(self, monkeypatch):
        monkeypatch.setattr(api_common, "allow_public_thumbs", lambda: True)
        d = api_common._row_to_dict(self._row(thumb_path=None))
        assert d["thumb_url"] is None

    @pytest.mark.parametrize("media_type, kind", [
        ("video/mp4", "video"),
        ("image/jpeg", "image"),
        ("application/pdf", "other"),
        ("", "other"),
        (None, "other"),
    ])
    def test_kind_from_media_type(self, monkeypatch, media_type, kind):
        monkeypatch.setattr(api_common, "allow_public_thumbs", lambda: True)
        d = api_common._row_to_dict(self._row(media_type=media_type))
        assert d["kind"] == kind

    def test_valid_tags_json_is_parsed(self, monkeypatch):
        monkeypatch.setattr(api_common, "allow_public_thumbs", lambda: True)
        d = api_common._row_to_dict(self._row(tags='["a", "b"]'))
        assert d["tags"] == ["a", "b"]

    def test_malformed_tags_json_falls_back_to_empty_list(self, monkeypatch):
        monkeypatch.setattr(api_common, "allow_public_thumbs", lambda: True)
        d = api_common._row_to_dict(self._row(tags="not-json"))
        assert d["tags"] == []

    def test_falsy_tags_is_empty_list(self, monkeypatch):
        monkeypatch.setattr(api_common, "allow_public_thumbs", lambda: True)
        d = api_common._row_to_dict(self._row(tags=None))
        assert d["tags"] == []

    def test_bool_coercion(self, monkeypatch):
        monkeypatch.setattr(api_common, "allow_public_thumbs", lambda: True)
        d = api_common._row_to_dict(self._row(favorited=1, archived=1, hidden=0))
        assert (d["favorited"], d["archived"], d["hidden"]) == (True, True, False)


# --- _clamp_search_limit / _topk_indices ------------------------------------

@pytest.mark.parametrize("limit, expected", [
    (-5, 1),
    (0, 1),
    (1, 1),
    (50, 50),
    (api_common.SEARCH_MAX_LIMIT, api_common.SEARCH_MAX_LIMIT),
    (api_common.SEARCH_MAX_LIMIT + 100, api_common.SEARCH_MAX_LIMIT),
])
def test_clamp_search_limit(limit, expected):
    assert api_common._clamp_search_limit(limit) == expected


class TestTopKIndices:
    def test_k_le_zero_returns_empty(self):
        out = api_common._topk_indices(np.array([3.0, 1.0, 2.0]), 0)
        assert out.tolist() == []

    def test_empty_scores_returns_empty(self):
        out = api_common._topk_indices(np.empty(0), 5)
        assert out.tolist() == []

    def test_k_greater_than_n_returns_all_sorted_desc(self):
        out = api_common._topk_indices(np.array([3.0, 1.0, 2.0]), 10)
        assert out.tolist() == [0, 2, 1]

    def test_returns_top_k_sorted_desc(self):
        scores = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
        out = api_common._topk_indices(scores, 2)
        assert out.tolist() == [1, 3]


# --- _dups_payload / _suggested_rows ----------------------------------------

def _unit(cos_with_e0: float, dim: int = 4, axis: int = 1) -> np.ndarray:
    """A unit vector whose dot product with e0=[1,0,...] is `cos_with_e0`,
    with the remaining energy placed on `axis` so two vectors built this way
    on *different* axes are themselves orthogonal."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = cos_with_e0
    v[axis] = np.sqrt(max(0.0, 1.0 - cos_with_e0 ** 2))
    return v


def _person(pid, name="unused"):
    return {"id": pid, "name": name, "cover_uid": None, "cover_face_id": None,
            "face_count": 1, "photo_count": 1, "cover_url": None}


class TestDupsPayload:
    def test_below_threshold_returns_no_duplicates(self, monkeypatch):
        people = [_person(1, "A"), _person(2, "B")]
        means = {1: _unit(1.0, axis=1), 2: _unit(0.0, axis=1)}
        monkeypatch.setattr(api, "_people_all_cached", lambda: people)
        monkeypatch.setattr(api, "person_mean_embeddings_from_cache", lambda: means)
        assert api_common._dups_payload(threshold=0.9, limit=50) == {"duplicates": []}

    def test_finds_pair_above_threshold(self, monkeypatch):
        people = [_person(1, "A"), _person(2, "B"), _person(3, "C")]
        means = {1: _unit(1.0, axis=1), 2: _unit(1.0, axis=1), 3: _unit(0.0, axis=1)}
        monkeypatch.setattr(api, "_people_all_cached", lambda: people)
        monkeypatch.setattr(api, "person_mean_embeddings_from_cache", lambda: means)
        out = api_common._dups_payload(threshold=0.9, limit=50)
        assert len(out["duplicates"]) == 1
        dup = out["duplicates"][0]
        assert dup["similarity"] == 1.0
        assert {dup["a"]["id"], dup["b"]["id"]} == {1, 2}

    def test_limit_keeps_only_the_highest_similarity_pairs(self, monkeypatch):
        # (1, 2) are exact duplicates (sim=1.0); (3, 4) are near-duplicates
        # (sim=0.95); both pairs clear the 0.9 threshold, but limit=1 must
        # keep only the strongest pair — exercising the top-k heap.
        people = [_person(i, str(i)) for i in (1, 2, 3, 4)]
        means = {
            1: np.array([1, 0, 0, 0], dtype=np.float32),
            2: np.array([1, 0, 0, 0], dtype=np.float32),
            3: np.array([0, 0, 1, 0], dtype=np.float32),
            4: np.array([0, 0, 0.95, np.sqrt(1 - 0.95 ** 2)], dtype=np.float32),
        }
        monkeypatch.setattr(api, "_people_all_cached", lambda: people)
        monkeypatch.setattr(api, "person_mean_embeddings_from_cache", lambda: means)
        out = api_common._dups_payload(threshold=0.9, limit=1)
        assert len(out["duplicates"]) == 1
        assert out["duplicates"][0]["similarity"] == 1.0

    def test_fewer_than_two_embeddings_returns_no_duplicates(self, monkeypatch):
        monkeypatch.setattr(api, "_people_all_cached", lambda: [_person(1)])
        monkeypatch.setattr(api, "person_mean_embeddings_from_cache", lambda: {1: _unit(1.0)})
        assert api_common._dups_payload(threshold=0.9, limit=50) == {"duplicates": []}


class TestSuggestedRows:
    def test_candidate_counts_and_ranking(self, monkeypatch):
        # P0 is similar to both P1 (0.7) and P2 (0.8); P1 and P2 are not
        # similar to each other (0.56, below the 0.6 threshold used here).
        p0, p1, p2 = _unit(1.0, dim=3, axis=1), _unit(0.7, dim=3, axis=1), _unit(0.8, dim=3, axis=2)
        people = [_person(0, "Z"), _person(1, "A"), _person(2, "B")]
        means = {0: p0, 1: p1, 2: p2}
        monkeypatch.setattr(api_common, "_people_all_cached", lambda: people)
        monkeypatch.setattr(api, "person_mean_embeddings_from_cache", lambda: means)
        rows = api_common._suggested_rows(threshold=0.6)
        by_id = {r["person_id"]: r for r in rows}
        assert by_id[0]["candidate_count"] == 2
        assert by_id[1]["candidate_count"] == 1
        assert by_id[2]["candidate_count"] == 1
        # Ranked by candidate_count desc, then top score desc: P0, P2 (0.8), P1 (0.7).
        assert [r["person_id"] for r in rows] == [0, 2, 1]

    def test_fewer_than_two_embeddings_returns_empty(self, monkeypatch):
        monkeypatch.setattr(api_common, "_people_all_cached", lambda: [_person(1)])
        monkeypatch.setattr(api, "person_mean_embeddings_from_cache", lambda: {1: _unit(1.0)})
        assert api_common._suggested_rows(threshold=0.4) == []


# --- _cached_bridge_health ---------------------------------------------------

class TestCachedBridgeHealth:
    def test_bridge_exception_falls_back_to_unhealthy(self, monkeypatch):
        api._bridge_health_cache = None

        def boom():
            raise RuntimeError("bridge down")

        monkeypatch.setattr(api, "get_bridge", boom)
        assert api_common._cached_bridge_health() == (False, False)
        # The (bad) result is still cached so a flapping bridge doesn't spam
        # health checks on every request within the TTL.
        assert api._bridge_health_cache[1] == (False, False)


# --- _fetch_remote_indexer_state / _indexer_proxy_json ----------------------

class _RaisingClient:
    def get(self, url, headers=None):
        raise httpx.ConnectError("connection refused")

    def request(self, method, url, json=None, headers=None):
        raise httpx.ConnectError("connection refused")


class TestFetchRemoteIndexerState:
    def test_client_error_falls_back_to_empty_state(self, monkeypatch):
        api._indexer_proxy_cache = None
        api._indexer_proxy_last_warn = 0.0
        monkeypatch.setattr(api, "_get_indexer_proxy_client", lambda: _RaisingClient())
        monkeypatch.setattr(api_common, "_cached_stats", lambda: {"photos": {"pending": 7}})
        payload = api_common._fetch_remote_indexer_state()
        assert payload["proxy_ok"] is False
        assert payload["proxy_error"] == "ConnectError"
        assert payload["pending_db"] == 7

    def test_repeated_failures_within_window_log_once(self, monkeypatch):
        api._indexer_proxy_cache = None
        api._indexer_proxy_last_warn = 0.0
        monkeypatch.setattr(api, "_get_indexer_proxy_client", lambda: _RaisingClient())
        monkeypatch.setattr(api_common, "_cached_stats", lambda: {"photos": {"pending": 0}})
        api_common._fetch_remote_indexer_state()
        first_warn = api._indexer_proxy_last_warn
        assert first_warn > 0
        # Force a second real attempt (bypass the payload cache) within the
        # same throttle window: the warn timestamp must not move again.
        api._indexer_proxy_cache = None
        api_common._fetch_remote_indexer_state()
        assert api._indexer_proxy_last_warn == first_warn


class TestIndexerProxyJson:
    def test_client_error_raises_http_exception(self, monkeypatch):
        monkeypatch.setattr(api, "_get_indexer_proxy_client", lambda: _RaisingClient())
        with pytest.raises(HTTPException) as exc_info:
            api_common._indexer_proxy_json("GET", "/status")
        assert exc_info.value.status_code == 502


# --- _face_crop_bytes ---------------------------------------------------------

class TestFaceCropBytes:
    def test_invalid_photo_uid_is_discarded(self, monkeypatch):
        monkeypatch.setattr(
            api_common, "_face_row",
            lambda fid: {"photo_uid": "bad/uid", "bbox": "[0,0,0.1,0.1]"},
        )
        assert api_common._face_crop_bytes(1) is None

    def test_missing_thumbnail_returns_none(self, app_settings, monkeypatch):
        monkeypatch.setattr(
            api_common, "_face_row",
            lambda fid: {"photo_uid": "p1", "bbox": "[0,0,0.1,0.1]"},
        )
        assert api_common._face_crop_bytes(1) is None

    def test_generates_and_caches_crop(self, app_settings, monkeypatch):
        from PIL import Image

        thumb = app_settings.thumb_dir / "p1.webp"
        Image.new("RGB", (64, 64), (10, 20, 30)).save(thumb, format="WEBP")
        monkeypatch.setattr(
            api_common, "_face_row",
            lambda fid: {"photo_uid": "p1", "bbox": "[0.1,0.1,0.4,0.4]"},
        )
        data = api_common._face_crop_bytes(42)
        assert data
        assert (app_settings.crops_dir / "42.jpg").exists()
        # A second call serves straight from the on-disk cache.
        monkeypatch.setattr(
            api_common, "_face_row",
            lambda fid: (_ for _ in ()).throw(AssertionError("should not re-query")),
        )
        assert api_common._face_crop_bytes(42) == data

    def test_cache_write_failure_still_returns_the_crop(self, app_settings, monkeypatch):
        from PIL import Image

        thumb = app_settings.thumb_dir / "p1.webp"
        Image.new("RGB", (64, 64), (10, 20, 30)).save(thumb, format="WEBP")
        monkeypatch.setattr(
            api_common, "_face_row",
            lambda fid: {"photo_uid": "p1", "bbox": "[0.1,0.1,0.4,0.4]"},
        )

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(api_common.os, "replace", boom)
        data = api_common._face_crop_bytes(43)
        assert data
        assert not (app_settings.crops_dir / "43.jpg").exists()


# --- _serve_person_cover / _serve_face_crop ----------------------------------

class TestServePersonCover:
    def test_no_cover_face_and_invalid_cover_uid_is_404(self):
        person = {"cover_face_id": None, "cover_uid": "bad/uid"}
        with pytest.raises(HTTPException) as exc_info:
            api_common._serve_person_cover(1, person)
        assert exc_info.value.status_code == 404

    def test_no_cover_face_falls_back_to_photo_thumb(self, app_settings):
        (app_settings.thumb_dir / "p1.webp").write_bytes(b"fake-webp")
        person = {"cover_face_id": None, "cover_uid": "p1"}
        resp = api_common._serve_person_cover(1, person)
        assert isinstance(resp, FileResponse)
        assert resp.media_type == "image/webp"

    def test_cached_crop_is_served_without_regenerating(self, app_settings, monkeypatch):
        cache_path = app_settings.crops_dir / "5.jpg"
        cache_path.write_bytes(b"cached-crop")
        monkeypatch.setattr(
            api_common, "_face_crop_bytes",
            lambda fid: (_ for _ in ()).throw(AssertionError("must not regenerate")),
        )
        person = {"cover_face_id": 5, "cover_uid": None}
        resp = api_common._serve_person_cover(1, person)
        assert isinstance(resp, FileResponse)
        assert str(resp.path) == str(cache_path)

    def test_crop_generation_failure_is_404(self, app_settings, monkeypatch):
        monkeypatch.setattr(api_common, "_face_crop_bytes", lambda fid: None)
        person = {"cover_face_id": 5, "cover_uid": None}
        with pytest.raises(HTTPException) as exc_info:
            api_common._serve_person_cover(1, person)
        assert exc_info.value.status_code == 404


class TestServeFaceCrop:
    def test_cached_crop_is_served(self, app_settings):
        cache_path = app_settings.crops_dir / "9.jpg"
        cache_path.write_bytes(b"cached-crop")
        resp = api_common._serve_face_crop(9, face={})
        assert isinstance(resp, FileResponse)
        assert str(resp.path) == str(cache_path)

    def test_crop_generation_failure_is_404(self, app_settings, monkeypatch):
        monkeypatch.setattr(api_common, "_face_crop_bytes", lambda fid: None)
        with pytest.raises(HTTPException) as exc_info:
            api_common._serve_face_crop(9, face={})
        assert exc_info.value.status_code == 404

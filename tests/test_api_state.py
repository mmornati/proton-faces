"""Unit coverage for app/src/api_state.py: module-level state that used to
be scattered across the api.py monolith (TTL caches, locks, the fullres
failure log, the indexer proxy client singleton) and the invalidation
helpers that mutate it.

Unlike tests/test_api.py, these tests call the module's functions directly
— no TestClient, no DB — since api_state.py is intentionally side-effect-
free at import time and its logic doesn't depend on the running app.
"""

import threading
import time

import api
import api_state


class TestFullResFailureWindow:
    """_record_full_res_failure / _recent_full_res_failures: a sliding
    15-minute window of fullres fetch failures, used by /api/admin/checks
    to surface bridge instability."""

    def test_no_failures_returns_zero(self):
        assert api_state._recent_full_res_failures() == 0

    def test_records_and_counts_within_window(self):
        api_state._record_full_res_failure()
        api_state._record_full_res_failure()
        assert api_state._recent_full_res_failures() == 2

    def test_trims_entries_older_than_the_window(self, monkeypatch):
        base = 1_000_000.0
        monkeypatch.setattr(time, "time", lambda: base)
        api_state._record_full_res_failure()
        # A second failure well inside the window must not evict the first.
        monkeypatch.setattr(time, "time", lambda: base + 60)
        api_state._record_full_res_failure()
        assert api_state._recent_full_res_failures() == 2
        # Once we're past the 15-minute window (measured from the *latest*
        # entry), a fresh record call trims the stale entries itself...
        monkeypatch.setattr(
            time, "time", lambda: base + 60 + api_state._FULL_RES_FAILURE_WINDOW_SEC + 1
        )
        api_state._record_full_res_failure()
        assert api_state._recent_full_res_failures() == 1
        # ...and _recent_full_res_failures() trims on its own too, even
        # without an intervening _record_full_res_failure() call.
        monkeypatch.setattr(
            time, "time",
            lambda: base + 60 + 2 * api_state._FULL_RES_FAILURE_WINDOW_SEC + 2,
        )
        assert api_state._recent_full_res_failures() == 0


class TestIndexerProxyClientSingleton:
    """_get_indexer_proxy_client: a lazy, double-checked-locking singleton."""

    def test_returns_the_same_client_across_calls(self):
        c1 = api_state._get_indexer_proxy_client()
        c2 = api_state._get_indexer_proxy_client()
        assert c1 is c2

    def test_singleton_is_thread_safe(self):
        clients: list = []
        barrier = threading.Barrier(20)

        def worker():
            barrier.wait()
            clients.append(api_state._get_indexer_proxy_client())

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(clients) == 20
        assert len(set(id(c) for c in clients)) == 1


class TestInvalidateCachesMutateApiNamespace:
    """Regression coverage for the #108-split bug: every reader/writer of
    these caches (api_common.py, api_routes_people.py, api_routes_photos.py)
    goes through ``api.<name>``, not this module's own attribute. An
    invalidation helper that only does ``global X; X = None`` rebinds
    api_state's own copy and silently no-ops from every caller's point of
    view — see the #108 code review for the full writeup. Parametrized so a
    future cache gets the same guard automatically.
    """

    def test_invalidate_people_cache_clears_api_namespace(self):
        api._people_cache["some-query"] = (time.time(), ["stale"])
        api_state._invalidate_people_cache()
        assert dict(api._people_cache) == {}

    def _set_and_invalidate(self, attr, invalidate):
        setattr(api, attr, (time.time(), "stale"))
        invalidate()
        assert getattr(api, attr) is None

    def test_invalidate_dups_cache_clears_api_namespace(self):
        api._suggested_cache[0.4] = (time.time(), ["stale"])
        self._set_and_invalidate("_dups_cache", api_state._invalidate_dups_cache)
        assert api._suggested_cache == {}

    def test_invalidate_photo_dups_cache_clears_api_namespace(self):
        self._set_and_invalidate(
            "_photo_dups_cache", api_state._invalidate_photo_dups_cache
        )

    def test_invalidate_clip_cache_clears_api_namespace(self):
        self._set_and_invalidate("_clip_cache", api_state._invalidate_clip_cache)

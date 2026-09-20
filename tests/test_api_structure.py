"""Structural regression guards for the api.py aggregator + api_routes_*
split (issue #108).

The split relies on two invariants that a future refactor could easily
break silently: every route handler function must still be reachable as
``api.<name>`` (tests, and some handlers themselves, call through that
namespace), and the route table itself must not gain/lose/rename an
endpoint by accident. Both were checked by hand during the #108 code
review (a manual decorator-by-decorator diff against the pre-split
monolith); these tests mechanize that check so it runs on every PR.
"""

from fastapi.routing import APIRoute, APIRouter

import api


def _iter_api_routes():
    """Flatten api.app's routes down to (methods, path, endpoint) triples,
    recursing through however FastAPI currently represents an included
    APIRouter (an internal detail that has changed across FastAPI versions)."""
    seen: set[int] = set()

    def _walk(routes):
        for route in routes:
            if id(route) in seen:
                continue
            seen.add(id(route))
            if isinstance(route, APIRoute):
                yield frozenset(route.methods or ()), route.path, route.endpoint
                continue
            inner_router = getattr(route, "original_router", None)
            if isinstance(inner_router, APIRouter):
                yield from _walk(inner_router.routes)
                continue
            nested = getattr(route, "routes", None)
            if nested:
                yield from _walk(nested)

    yield from _walk(api.app.routes)


# Locked from the route table as of the #108 split + follow-up fix PR.
# A change here should be a deliberate, reviewed edit — not a side effect
# of a module refactor silently dropping, duplicating, or renaming a route.
EXPECTED_ROUTES = {
    ("DELETE", "/api/admin/backups/{name}"),
    ("DELETE", "/api/admin/users/{user_id}"),
    ("GET", "/api/admin/backups"),
    ("GET", "/api/admin/bridge/cache"),
    ("GET", "/api/admin/overview"),
    ("GET", "/api/admin/schedule"),
    ("GET", "/api/admin/sync"),
    ("GET", "/api/admin/users"),
    ("GET", "/api/albums"),
    ("GET", "/api/albums/{album_uid}/photos"),
    ("GET", "/api/auth/limits"),
    ("GET", "/api/auth/me"),
    ("GET", "/api/bridge_health"),
    ("GET", "/api/duplicates"),
    ("GET", "/api/faces/unassigned"),
    ("GET", "/api/faces/{face_id}/crop"),
    ("GET", "/api/faces/{face_id}/suggest"),
    ("GET", "/api/health"),
    ("GET", "/api/map"),
    ("GET", "/api/memories"),
    ("GET", "/api/people"),
    ("GET", "/api/people/duplicates"),
    ("GET", "/api/people/suggested-merges"),
    ("GET", "/api/people/{person_id}/cover"),
    ("GET", "/api/people/{person_id}/faces"),
    ("GET", "/api/people/{person_id}/map"),
    ("GET", "/api/people/{person_id}/photos"),
    ("GET", "/api/people/{person_id}/similar"),
    ("GET", "/api/photos"),
    ("GET", "/api/photos/anchors"),
    ("GET", "/api/photos/archived"),
    ("GET", "/api/photos/{uid}"),
    ("GET", "/api/photos/{uid}/faces"),
    ("GET", "/api/photos/{uid}/full"),
    ("GET", "/api/photos/{uid}/meta"),
    ("GET", "/api/photos/{uid}/tags"),
    ("GET", "/api/photos/{uid}/thumb"),
    ("GET", "/api/places"),
    ("GET", "/api/search"),
    ("GET", "/api/stats"),
    ("GET", "/api/status"),
    ("GET", "/api/tags"),
    ("PATCH", "/api/admin/users/{user_id}"),
    ("PATCH", "/api/photos/{uid}"),
    ("POST", "/api/admin/backup"),
    ("POST", "/api/admin/backups/prune"),
    ("POST", "/api/admin/bridge/cache/clear"),
    ("POST", "/api/admin/checks"),
    ("POST", "/api/admin/db/compact"),
    ("POST", "/api/admin/people/gc-empty"),
    ("POST", "/api/admin/sync/trigger"),
    ("POST", "/api/admin/users"),
    ("POST", "/api/admin/users/{user_id}/2fa/disable"),
    ("POST", "/api/admin/users/{user_id}/logout"),
    ("POST", "/api/auth/2fa/confirm"),
    ("POST", "/api/auth/2fa/disable"),
    ("POST", "/api/auth/2fa/setup"),
    ("POST", "/api/auth/2fa/verify"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/logout"),
    ("POST", "/api/auth/password"),
    ("POST", "/api/auth/refresh"),
    ("POST", "/api/faces/{face_id}/person"),
    ("POST", "/api/faces/{face_id}/unassign"),
    ("POST", "/api/people/{person_id}/cover"),
    ("POST", "/api/people/{person_id}/name"),
    ("POST", "/api/people/{source_id}/merge"),
    ("POST", "/api/people/{target_id}/merge_all"),
    ("POST", "/api/people/{target_id}/merge_all_similar"),
    ("POST", "/api/search/face"),
    ("POST", "/api/sign"),
    ("PUT", "/api/admin/schedule"),
    ("PUT", "/api/admin/sync"),
    ("PUT", "/api/photos/{uid}/tags"),
}


def test_route_table_snapshot():
    actual = {
        (method, path)
        for methods, path, _ in _iter_api_routes()
        for method in methods
    }
    missing = EXPECTED_ROUTES - actual
    extra = actual - EXPECTED_ROUTES
    assert not missing, f"routes dropped or renamed: {sorted(missing)}"
    assert not extra, f"new routes not yet reviewed into EXPECTED_ROUTES: {sorted(extra)}"


def test_every_route_handler_is_reexported():
    """Every function backing an app route must be reachable as api.<name>
    — tests (and some handlers) call through that namespace, and it's the
    contract the api.py docstring advertises for the whole split."""
    unreachable = []
    for _methods, path, endpoint in _iter_api_routes():
        if endpoint is None:
            continue
        name = endpoint.__name__
        if getattr(api, name, None) is not endpoint:
            unreachable.append((path, name))
    assert not unreachable, f"route handlers not re-exported from api.py: {unreachable}"



def test_metadata_panel_escapes_values():
    """The photo metadata table must escape every value that is not an
    explicitly pre-built HTML pill (audit 2026-09: `m.name` is the Proton
    filename, controllable by whoever shared the photo)."""
    from pathlib import Path

    html = Path(__file__).resolve().parent.parent / "app" / "src" / "static" / "index.html"
    src = html.read_text()
    assert '<td class="v">${v}</td>' not in src
    assert '<td class="v">${html ? v : escv(v)}</td>' in src



def _spa_sources():
    from pathlib import Path

    static = Path(__file__).resolve().parent.parent / "app" / "src" / "static"
    return (static / "index.html").read_text(), (static / "sw.js").read_text()


def test_spa_refresh_shares_parsed_body_not_response():
    """Concurrent 401s must share one parsed refresh result (audit B-10)."""
    html, _ = _spa_sources()
    expected = "}).then(async (r) => {\n      if (!r.ok) throw new Error(\"refresh failed\");\n      return r.json();"
    assert expected in html
    assert "const r = await _refreshInflight;" not in html


def test_spa_face_search_is_authenticated():
    html, _ = _spa_sources()
    assert 'fetch("/api/search/face"' not in html
    assert 'api("/api/search/face", { method: "POST", body: fd })' in html


def test_spa_grid_listeners_are_delegated():
    html, _ = _spa_sources()
    assert "_bindGridDelegation(el)" in html
    assert 'el.querySelectorAll(".starbtn").forEach' not in html


def test_service_worker_never_caches_failed_shell():
    _, sw = _spa_sources()
    assert 'if (resp.ok && resp.type === "basic")' in sw
    assert 'const VERSION = "pf-shell-v3"' in sw

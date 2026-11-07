import http.server
import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import api
import auth
import config
import faces
import indexer
import store
import telemetry

APP_SRC = Path(__file__).resolve().parent.parent / "app" / "src"


@pytest.fixture
def client(tmp_db, monkeypatch):
    monkeypatch.setenv("SIGNING_SECRET", "test-signing-secret-of-sufficient-length")
    with TestClient(api.app) as c:
        yield c


def _bearer(client):
    password = "password123"
    store.create_user(username="admin", password_hash=auth.hash_password(password), role="admin", display_name="A")
    r = client.post("/api/auth/login", json={"username": "admin", "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _photo(uid):
    return {
        "uid": uid,
        "name": f"{uid}-secret.jpg",
        "media_type": "image/jpeg",
        "capture_time": 1700000000,
        "sha1": f"sha-{uid}",
        "albums": [],
        "size": 1000,
        "duration_sec": None,
    }


@pytest.fixture
def reader():
    r = InMemoryMetricReader()
    assert telemetry.setup("test-service", reader=r)
    return r


def _points(reader, name):
    """Data points of one metric, as a list of (attributes, point)."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    out = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    out.extend((dict(p.attributes), p) for p in metric.data.data_points)
    return out


def _all_attribute_values(reader):
    values = []
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for p in metric.data.data_points:
                    values.extend(str(v) for v in p.attributes.values())
    return values


class TestOffByDefault:
    def test_setup_without_endpoint_is_off(self, monkeypatch):
        monkeypatch.setattr(config.settings, "otel_endpoint", "")
        assert telemetry.setup(telemetry.APP_SERVICE) is False
        assert telemetry.enabled() is False

    def test_sdk_disabled_wins_over_endpoint(self, monkeypatch):
        monkeypatch.setattr(config.settings, "otel_endpoint", "http://collector:4318")
        monkeypatch.setattr(config.settings, "otel_sdk_disabled", True)
        assert telemetry.setup(telemetry.APP_SERVICE) is False

    def test_helpers_are_noops_when_off(self):
        telemetry.record_photo(1.0, True)
        telemetry.count_error("process")
        telemetry.record_model_load("clip", 1.0)
        telemetry.JobRun("sync").finish("error")
        with telemetry.job_run("cluster") as run:
            run.fail()
        with telemetry.search_timer("text", "embed"):
            pass
        assert telemetry._job_last_run == {}

    def test_sdk_is_not_imported_when_off(self, tmp_path):
        env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
        env["DATA_DIR"] = str(tmp_path)
        env["SIGNING_SECRET"] = "x" * 32
        code = (
            "import sys, api, indexer, telemetry\n"
            "assert telemetry.setup(telemetry.APP_SERVICE) is False\n"
            "assert not [m for m in sys.modules if m.startswith('opentelemetry')]\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=APP_SRC, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

    def test_setup_failure_leaves_metrics_off(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_resource", lambda _name: (_ for _ in ()).throw(ValueError("bad")))
        assert telemetry.setup("x", reader=InMemoryMetricReader()) is False
        assert telemetry.enabled() is False
        telemetry.count_error("process")


class TestResource:
    def test_default_service_name_and_instance(self, monkeypatch):
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
        attrs = telemetry._resource(telemetry.INDEXER_SERVICE).attributes
        assert attrs["service.name"] == "proton-faces-indexer"
        assert attrs["service.instance.id"] == str(os.getpid())

    def test_env_wins(self, monkeypatch):
        monkeypatch.setenv("OTEL_SERVICE_NAME", "custom")
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.instance.id=a,deployment.environment=home")
        attrs = telemetry._resource(telemetry.APP_SERVICE).attributes
        assert attrs["service.name"] == "custom"
        assert attrs["service.instance.id"] == "a"
        assert attrs["deployment.environment"] == "home"

    def test_setup_is_idempotent(self, reader):
        assert telemetry.setup("other", reader=InMemoryMetricReader()) is True
        telemetry.count_error("process")
        assert _points(reader, "proton_faces.indexer.errors")


class TestHttpMetrics:
    def test_records_route_template_not_raw_path(self, client, reader):
        uid = "SECRET-UID-123"
        store.upsert_photos([_photo(uid)])
        client.get(f"/api/photos/{uid}/thumb")
        points = _points(reader, "proton_faces.http.request.duration")
        routes = {attrs["http.route"] for attrs, _ in points}
        assert "/api/photos/{uid}/thumb" in routes
        assert not [v for v in _all_attribute_values(reader) if uid in v]

    def test_count_and_status_class(self, client, reader):
        for _ in range(3):
            client.get("/api/people")
        points = _points(reader, "proton_faces.http.request.duration")
        match = [(a, p) for a, p in points if a["http.route"] == "/api/people"]
        assert len(match) == 1
        attrs, point = match[0]
        assert attrs["http.request.method"] == "GET"
        assert attrs["http.response.status_class"] == "4xx"
        assert point.count == 3
        assert point.sum > 0

    def test_unmatched_and_static_paths_collapse_to_other(self, client, reader):
        client.get("/wp-admin/secret-scan.php")
        client.get("/.env")
        client.get("/index.html")
        routes = {a["http.route"] for a, _ in _points(reader, "proton_faces.http.request.duration")}
        assert routes == {"other"}

    def test_unknown_method_collapses_to_other(self, client, reader):
        client.request("EVILMETHOD", "/api/people")
        methods = {a["http.request.method"] for a, _ in _points(reader, "proton_faces.http.request.duration")}
        assert methods == {"OTHER"}

    def test_body_limit_rejection_is_measured(self, client, reader, monkeypatch):
        client.post("/api/auth/login", content=b"x" * (config.settings.max_json_body_bytes + 1))
        classes = {a["http.response.status_class"] for a, _ in _points(reader, "proton_faces.http.request.duration")}
        assert classes == {"4xx"}

    def test_no_series_when_off(self, client):
        assert client.get("/api/people").status_code == 401
        assert telemetry.enabled() is False

    def test_failed_route_still_measured(self, reader):
        async def boom(scope, receive, send):
            raise RuntimeError("boom")

        import asyncio

        middleware = telemetry.MetricsMiddleware(boom)
        scope = {"type": "http", "method": "GET", "path": "/x"}
        with pytest.raises(RuntimeError):
            asyncio.run(middleware(scope, None, None))
        attrs = [a for a, _ in _points(reader, "proton_faces.http.request.duration")]
        assert attrs == [{"http.request.method": "GET", "http.route": "other", "http.response.status_class": "5xx"}]


class TestSearchMetrics:
    def test_text_search_phases_without_the_query(self, client, reader, tmp_db, monkeypatch):
        import numpy as np

        monkeypatch.setattr(api, "embed_text", lambda _q: np.zeros(512, dtype=np.float32))
        monkeypatch.setattr(api, "_semantic_search", lambda *_a: {"items": []})
        query = "my-private-query"
        response = client.get(f"/api/search?q={query}", headers=_bearer(client))
        assert response.status_code == 200
        phases = {
            (a["search.type"], a["search.phase"]) for a, _ in _points(reader, "proton_faces.search.duration")
        }
        assert phases == {("text", "embed"), ("text", "match")}
        assert not [v for v in _all_attribute_values(reader) if query in v]

    def test_failed_embed_not_recorded(self, client, reader, tmp_db, monkeypatch):
        def fail(_q):
            raise RuntimeError("no model")

        monkeypatch.setattr(api, "embed_text", fail)
        assert client.get("/api/search?q=x", headers=_bearer(client)).status_code == 503
        assert _points(reader, "proton_faces.search.duration") == []

    def test_face_search_phases(self, client, reader, tmp_db, monkeypatch):
        import numpy as np
        from PIL import Image

        monkeypatch.setattr(api, "embed_query_face", lambda _img: np.zeros(512, dtype=np.float32))
        monkeypatch.setattr(api, "_face_similarity", lambda *_a: {"items": []})
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        response = client.post(
            "/api/search/face", files={"file": ("a.png", buf.getvalue(), "image/png")}, headers=_bearer(client)
        )
        assert response.status_code == 200
        phases = {
            (a["search.type"], a["search.phase"]) for a, _ in _points(reader, "proton_faces.search.duration")
        }
        assert phases == {("face", "embed"), ("face", "match")}


class TestModelLoad:
    def test_faces_load_is_timed(self, reader, monkeypatch, app_settings):
        import types

        class _Analysis:
            def __init__(self, **_kw):
                pass

            def prepare(self, **_kw):
                pass

        module = types.ModuleType("insightface.app")
        module.FaceAnalysis = _Analysis
        monkeypatch.setitem(sys.modules, "insightface", types.ModuleType("insightface"))
        monkeypatch.setitem(sys.modules, "insightface.app", module)
        monkeypatch.setattr(faces, "_app", None)
        faces._load()
        points = _points(reader, "proton_faces.model.load.duration")
        assert [a for a, _ in points] == [{"model": "insightface"}]


class TestJobs:
    def test_success_records_duration_and_timestamps(self, reader):
        before = time.time()
        with telemetry.job_run("cluster"):
            pass
        assert [a for a, _ in _points(reader, "proton_faces.indexer.job.duration")] == [
            {"job": "cluster", "outcome": "ok"}
        ]
        last_run = _points(reader, "proton_faces.indexer.job.last_run")
        assert last_run[0][0] == {"job": "cluster"}
        assert last_run[0][1].value >= before
        assert _points(reader, "proton_faces.indexer.job.last_success")
        assert _points(reader, "proton_faces.indexer.job.failures") == []

    def test_exception_counts_as_failure_and_propagates(self, reader):
        with pytest.raises(ValueError):
            with telemetry.job_run("sync"):
                raise ValueError("boom")
        assert [(a, p.value) for a, p in _points(reader, "proton_faces.indexer.job.failures")] == [
            ({"job": "sync", "kind": "error"}, 1)
        ]
        assert _points(reader, "proton_faces.indexer.job.last_success") == []
        assert _points(reader, "proton_faces.indexer.job.last_run")

    def test_albums_fetch_failure_is_a_job_failure(self, reader, tmp_db, monkeypatch):
        class _Bridge:
            def albums(self):
                raise RuntimeError("bridge down with secret-album-name")

        monkeypatch.setattr(indexer, "get_bridge", lambda: _Bridge())
        assert indexer._sync_albums_once() == 0
        assert [a for a, _ in _points(reader, "proton_faces.indexer.job.failures")] == [
            {"job": "albums", "kind": "error"}
        ]
        assert not [v for v in _all_attribute_values(reader) if "secret" in v]

    def test_full_scan_failure_is_a_job_failure(self, reader, monkeypatch):
        def fail():
            raise RuntimeError("proton down")

        monkeypatch.setattr(indexer, "_sync_once", fail)
        with pytest.raises(RuntimeError):
            indexer._run_full_scan_and_cleanup()
        assert [a for a, _ in _points(reader, "proton_faces.indexer.job.failures")] == [
            {"job": "sync", "kind": "error"}
        ]

    def test_gps_subprocess_reports_exit_code(self, monkeypatch):
        import subprocess as sp

        def run(returncode):
            monkeypatch.setattr(
                sp, "run", lambda *_a, **_k: sp.CompletedProcess([], returncode, stdout="", stderr="")
            )
            return indexer._run_gps_subprocess()

        assert run(0) is True
        assert run(1) is False


class TestPhotoMetrics:
    def _seed(self, app_settings, uid):
        from PIL import Image

        store.upsert_photos([_photo(uid)])
        store.claim_photos_for_download([uid])
        Image.new("RGB", (8, 8)).save(indexer._work_path(uid), format="WEBP")

    def test_processed_photo_counted(self, reader, tmp_db, app_settings, monkeypatch):
        import numpy as np

        monkeypatch.setattr(indexer, "detect_faces", lambda _img: [])
        monkeypatch.setattr(indexer, "embed_batch", lambda imgs: np.zeros((len(imgs), 512), dtype=np.float32))
        self._seed(app_settings, "PHOTO-UID-1")
        indexer._process_batch(["PHOTO-UID-1"])
        assert [(a, p.value) for a, p in _points(reader, "proton_faces.indexer.photos.processed")] == [
            ({"outcome": "done"}, 1)
        ]
        duration = _points(reader, "proton_faces.indexer.photo.duration")
        assert duration[0][1].count == 1
        assert not [v for v in _all_attribute_values(reader) if "PHOTO-UID" in v or "secret" in v]

    def test_missing_work_file_counted_as_error(self, reader, tmp_db, app_settings):
        store.upsert_photos([_photo("gone")])
        store.claim_photos_for_download(["gone"])
        indexer._process_batch(["gone"])
        assert [a for a, _ in _points(reader, "proton_faces.indexer.photos.processed")] == [{"outcome": "error"}]

    def test_lost_claim_counts_nothing(self, reader, tmp_db, app_settings):
        indexer._process_batch(["never-claimable"])
        assert _points(reader, "proton_faces.indexer.photos.processed") == []

    def test_thumbnail_errors_by_stage(self, reader, tmp_db, app_settings):
        store.upsert_photos([_photo("u1")])

        class _Bridge:
            def thumbnails(self, _uids):
                raise RuntimeError("boom")

        indexer._downloader_tick(_Bridge())
        assert [a for a, _ in _points(reader, "proton_faces.indexer.errors")] == [{"stage": "download"}]


class TestIndexerGauges:
    def test_queue_and_threads(self, reader, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "get_indexer_state",
            lambda: {
                "pending_in_queue": 4,
                "threads": {"sync": True, "gps": False, "worker-0": True, "worker-1": True, "worker-2": False},
            },
        )
        monkeypatch.setattr(indexer, "_fullres_backoff", {"a-uid": 1.0})
        telemetry.watch_indexer(indexer._metrics_snapshot)
        queues = {a["queue"]: p.value for a, p in _points(reader, "proton_faces.indexer.queue.depth")}
        assert queues == {"pending": 4, "fullres_backoff": 1}
        threads = {a["thread"]: p.value for a, p in _points(reader, "proton_faces.indexer.thread.alive")}
        assert threads == {"sync": 1, "gps": 0, "worker": 2}

    def test_snapshot_failure_does_not_break_collection(self, reader):
        def fail():
            raise RuntimeError("boom")

        telemetry.watch_indexer(fail)
        telemetry.count_error("process")
        assert _points(reader, "proton_faces.indexer.errors")
        assert _points(reader, "proton_faces.indexer.queue.depth") == []


class _Collector(http.server.BaseHTTPRequestHandler):
    received: list[tuple[str, bytes]] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).received.append((self.path, body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args):
        pass


class TestRealExporter:
    def _setup(self, monkeypatch, endpoint):
        monkeypatch.setattr(config.settings, "otel_endpoint", endpoint)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1000")
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
        assert telemetry.setup(telemetry.APP_SERVICE) is True

    def test_exports_otlp_http_with_service_name(self, monkeypatch):
        _Collector.received = []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self._setup(monkeypatch, f"http://127.0.0.1:{server.server_port}")
            telemetry.count_error("process")
            telemetry.shutdown()
        finally:
            server.shutdown()
        path, body = _Collector.received[-1]
        assert path == "/v1/metrics"
        request = ExportMetricsServiceRequest.FromString(body)
        resource = {a.key: a.value.string_value for a in request.resource_metrics[0].resource.attributes}
        assert resource["service.name"] == "proton-faces-app"
        names = {m.name for sm in request.resource_metrics[0].scope_metrics for m in sm.metrics}
        assert "proton_faces.indexer.errors" in names

    def test_collector_down_never_blocks_the_app(self, client, monkeypatch):
        self._setup(monkeypatch, "http://127.0.0.1:9")
        start = time.monotonic()
        assert client.get("/api/people").status_code == 401
        telemetry.count_error("process")
        telemetry.shutdown()
        assert time.monotonic() - start < telemetry._SHUTDOWN_TIMEOUT_MS / 1000 + 2


def test_batch_failure_counts_without_a_duration(reader):
    telemetry.record_photo(None, False)
    assert [a for a, _ in _points(reader, "proton_faces.indexer.photos.processed")] == [{"outcome": "error"}]
    assert _points(reader, "proton_faces.indexer.photo.duration") == []


class _StopLoop(BaseException):
    """Escapes the fullres loop, whose own handlers catch Exception."""


class TestFullresJob:
    def _run_one(self, monkeypatch, app_settings, full_photo):
        class _Bridge:
            def __init__(self):
                self.full_photo = full_photo

        store.upsert_photos([_photo("fr-uid")])
        store.set_photo_full("fr-uid")
        monkeypatch.setattr(indexer, "get_bridge", lambda: _Bridge())

        def stop(_seconds):
            raise _StopLoop

        monkeypatch.setattr(indexer.time, "sleep", stop)
        with pytest.raises(_StopLoop):
            indexer._fullres_loop()

    def test_failure(self, reader, tmp_db, app_settings, monkeypatch):
        def boom(_uid):
            raise RuntimeError("fr-uid exploded")

        self._run_one(monkeypatch, app_settings, boom)
        assert [a for a, _ in _points(reader, "proton_faces.indexer.job.failures")] == [
            {"job": "fullres", "kind": "error"}
        ]
        assert not [v for v in _all_attribute_values(reader) if "fr-uid" in v]

    def test_transient(self, reader, tmp_db, app_settings, monkeypatch):
        def limited(_uid):
            raise indexer.BridgeTransientError(429, "slow down", retry_after_sec=1.0)

        self._run_one(monkeypatch, app_settings, limited)
        assert [a for a, _ in _points(reader, "proton_faces.indexer.job.failures")] == [
            {"job": "fullres", "kind": "transient"}
        ]
        assert [a["outcome"] for a, _ in _points(reader, "proton_faces.indexer.job.duration")] == ["transient"]

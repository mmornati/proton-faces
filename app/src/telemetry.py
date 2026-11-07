"""Optional OpenTelemetry metrics, exported over OTLP/HTTP.

Off unless OTEL_EXPORTER_OTLP_ENDPOINT is set: `opentelemetry` is imported
inside setup() only, and every record helper is a no-op until then. Export runs
on the SDK's background thread, so a slow or absent collector costs a failed
POST per interval there and never touches a request or an indexing thread.

Attribute values come from fixed sets or from route templates only. Never pass
a photo uid, filename, person name, search query, URL or token here: the
collector is outside the app's trust boundary (SECURITY_HARDENING.md).
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from contextlib import contextmanager

from config import settings

log = logging.getLogger("telemetry")

APP_SERVICE = "proton-faces-app"
INDEXER_SERVICE = "proton-faces-indexer"

_SHUTDOWN_TIMEOUT_MS = 5000
_HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
_THREAD_INDEX = re.compile(r"-\d+$")

_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
_SEARCH_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
_MODEL_BUCKETS = (0.5, 1, 2.5, 5, 10, 30, 60, 120)
_PHOTO_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
_JOB_BUCKETS = (0.1, 1, 5, 15, 60, 300, 900, 1800, 3600, 7200)

_lock = threading.Lock()
_provider = None
_instruments: dict = {}
_job_last_run: dict[str, float] = {}
_job_last_success: dict[str, float] = {}
_indexer_snapshot = None


def enabled() -> bool:
    return _provider is not None


def setup(default_service_name: str, reader=None) -> bool:
    """Start exporting from this process. Idempotent; never raises.

    Call it from the process that serves (uvicorn worker or indexer), not from
    a parent that forks workers or from a one-shot CLI. `reader` replaces the
    OTLP exporter in tests.
    """
    global _provider
    if reader is None and (not settings.otel_endpoint or settings.otel_sdk_disabled):
        return False
    with _lock:
        if _provider is not None:
            return True
        try:
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            if reader is None:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

                reader = PeriodicExportingMetricReader(OTLPMetricExporter())
            provider = MeterProvider(metric_readers=[reader], resource=_resource(default_service_name))
            _instruments.update(_create_instruments(provider.get_meter("proton-faces")))
        except Exception as exc:  # noqa: BLE001 - metrics must never block boot
            log.warning("OpenTelemetry metrics disabled: %s", exc)
            _instruments.clear()
            return False
        _provider = provider
    log.info("OpenTelemetry metrics enabled (service default %s)", default_service_name)
    return True


def shutdown() -> None:
    """Flush and stop the exporter, bounded by _SHUTDOWN_TIMEOUT_MS."""
    global _provider, _indexer_snapshot
    with _lock:
        provider, _provider = _provider, None
        _instruments.clear()
    if provider is not None:
        try:
            provider.shutdown(timeout_millis=_SHUTDOWN_TIMEOUT_MS)
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenTelemetry shutdown failed: %s", exc)
    # Cleared last: the final flush above still reports these gauges.
    _job_last_run.clear()
    _job_last_success.clear()
    _indexer_snapshot = None


def _resource(default_service_name: str):
    from opentelemetry.sdk.resources import Resource

    detected = Resource.create()
    attrs: dict[str, str] = {}
    # OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES win over the per-process default.
    if detected.attributes.get("service.name", "unknown_service").startswith("unknown_service"):
        attrs["service.name"] = default_service_name
    # The SDK's default is a random UUID, which starts new series on every restart.
    # The pid keeps sibling uvicorn workers apart and is stable in a container.
    if "service.instance.id" not in os.environ.get("OTEL_RESOURCE_ATTRIBUTES", ""):
        attrs["service.instance.id"] = str(os.getpid())
    return detected.merge(Resource(attrs))


def _create_instruments(meter) -> dict:
    def seconds(name: str, description: str, buckets: tuple) -> object:
        return meter.create_histogram(
            name, unit="s", description=description, explicit_bucket_boundaries_advisory=list(buckets)
        )

    meter.create_observable_gauge(
        "proton_faces.indexer.queue.depth",
        callbacks=[_observe_queues],
        description="Items waiting in the indexer's in-memory queues, by queue.",
    )
    meter.create_observable_gauge(
        "proton_faces.indexer.thread.alive",
        callbacks=[_observe_threads],
        description="Live indexer threads, by thread role. 0 means that role died.",
    )
    meter.create_observable_gauge(
        "proton_faces.indexer.job.last_run",
        callbacks=[lambda _options: _timestamps(_job_last_run)],
        unit="s",
        description="Unix time the job last finished, whatever the outcome.",
    )
    meter.create_observable_gauge(
        "proton_faces.indexer.job.last_success",
        callbacks=[lambda _options: _timestamps(_job_last_success)],
        unit="s",
        description="Unix time the job last finished without failing.",
    )
    return {
        "http": seconds(
            "proton_faces.http.request.duration", "HTTP request duration, by route template.", _HTTP_BUCKETS
        ),
        "search": seconds("proton_faces.search.duration", "Search time, by search type and phase.", _SEARCH_BUCKETS),
        "model": seconds("proton_faces.model.load.duration", "ML model load time, by model.", _MODEL_BUCKETS),
        "photo": seconds(
            "proton_faces.indexer.photo.duration",
            "Recognition time per photo, including its share of a CLIP batch.",
            _PHOTO_BUCKETS,
        ),
        "job": seconds("proton_faces.indexer.job.duration", "Background job run time, by job.", _JOB_BUCKETS),
        "photos": meter.create_counter(
            "proton_faces.indexer.photos.processed", description="Photos through the recognition workers, by outcome."
        ),
        "errors": meter.create_counter("proton_faces.indexer.errors", description="Indexer errors, by pipeline stage."),
        "job_failures": meter.create_counter(
            "proton_faces.indexer.job.failures", description="Failed job runs, by job and kind."
        ),
    }


def _record(name: str, value: float, attributes: dict[str, str]) -> None:
    instrument = _instruments.get(name)
    if instrument is None:
        return
    try:
        instrument.record(value, attributes)
    except Exception:  # noqa: BLE001 - a metric must never break the caller
        log.debug("could not record %s", name, exc_info=True)


def _add(name: str, attributes: dict[str, str]) -> None:
    instrument = _instruments.get(name)
    if instrument is None:
        return
    try:
        instrument.add(1, attributes)
    except Exception:  # noqa: BLE001 - a metric must never break the caller
        log.debug("could not record %s", name, exc_info=True)


# --- app ---------------------------------------------------------------------


class MetricsMiddleware:
    """Times each HTTP request up to its last body byte. Pure ASGI, so streamed
    responses are not buffered. Add it last so it wraps every other middleware."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or _provider is None:
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status: int | None = None

        async def send_with_status(message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        except Exception:
            status = status or 500
            raise
        finally:
            _record("http", time.perf_counter() - start, _http_attributes(scope, status))


def _http_attributes(scope, status: int | None) -> dict[str, str]:
    method = scope.get("method", "")
    return {
        "http.request.method": method if method in _HTTP_METHODS else "OTHER",
        # Only a matched API route carries a template. The raw path is never
        # used: anything else, static files and 404 scans included, is "other".
        "http.route": getattr(scope.get("route"), "path", None) or "other",
        "http.response.status_class": f"{status // 100}xx" if status else "none",
    }


@contextmanager
def search_timer(search_type: str, phase: str):
    """Times a search phase ("text"/"face", "embed"/"match"); failures are not recorded."""
    start = time.perf_counter()
    yield
    _record("search", time.perf_counter() - start, {"search.type": search_type, "search.phase": phase})


def record_model_load(model: str, seconds: float) -> None:
    _record("model", seconds, {"model": model})


# --- indexer -----------------------------------------------------------------


def record_photo(seconds: float | None, ok: bool) -> None:
    outcome = "done" if ok else "error"
    if seconds is not None:
        _record("photo", seconds, {"outcome": outcome})
    _add("photos", {"outcome": outcome})


def count_error(stage: str) -> None:
    _add("errors", {"stage": stage})


class JobRun:
    """One run of a background job. Call finish() once, before any pacing sleep."""

    def __init__(self, job: str) -> None:
        self.job = job
        self._start = time.perf_counter()

    def finish(self, failure: str | None = None) -> None:
        if _provider is None:
            return
        now = time.time()
        _job_last_run[self.job] = now
        _record("job", time.perf_counter() - self._start, {"job": self.job, "outcome": failure or "ok"})
        if failure:
            _add("job_failures", {"job": self.job, "kind": failure})
        else:
            _job_last_success[self.job] = now


class _JobScope:
    failure: str | None = None

    def fail(self, kind: str = "error") -> None:
        self.failure = kind


@contextmanager
def job_run(job: str):
    """Measures a job run. An exception, or `scope.fail()` where the job swallows its own errors, marks it failed."""
    run = JobRun(job)
    scope = _JobScope()
    try:
        yield scope
    except Exception:
        scope.fail()
        raise
    finally:
        run.finish(scope.failure)


def watch_indexer(snapshot) -> None:
    """Registers the callable behind the indexer gauges.

    It runs on the export thread every interval and must stay cheap (no DB, no
    I/O). It returns {"queues": {name: depth}, "threads": {name: alive}}.
    """
    global _indexer_snapshot
    _indexer_snapshot = snapshot


def _read_snapshot() -> dict:
    if _indexer_snapshot is None:
        return {}
    try:
        return _indexer_snapshot()
    except Exception:  # noqa: BLE001 - an exporting gauge must not take the collection down
        log.debug("indexer snapshot failed", exc_info=True)
        return {}


def _observe_queues(_options) -> list:
    from opentelemetry.metrics import Observation

    return [Observation(depth, {"queue": name}) for name, depth in _read_snapshot().get("queues", {}).items()]


def _observe_threads(_options) -> list:
    from opentelemetry.metrics import Observation

    alive: dict[str, int] = {}
    for name, is_alive in _read_snapshot().get("threads", {}).items():
        role = _THREAD_INDEX.sub("", name)
        alive[role] = alive.get(role, 0) + int(bool(is_alive))
    return [Observation(count, {"thread": role}) for role, count in alive.items()]


def _timestamps(values: dict[str, float]) -> list:
    from opentelemetry.metrics import Observation

    return [Observation(ts, {"job": job}) for job, ts in list(values.items())]

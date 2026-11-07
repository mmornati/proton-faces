# Observability

The `app` and `indexer` processes can export OpenTelemetry metrics over OTLP/HTTP to a collector you run. It is **off by default**: without `OTEL_EXPORTER_OTLP_ENDPOINT` the SDK is not even imported, and nothing leaves the host. Demo installs are unchanged.

`/api/status` stays the source for health and photo counts. These metrics add what it can't show: latency, throughput and per-job history.

## Enable it

Set this on the `app` and `indexer` containers (`compose.yml` passes it through from `.env`):

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
```

Use the collector's **HTTP** port (4318 by convention). The gRPC port (4317) will not work. The exporter appends `/v1/metrics` to this base URL.

!!! warning "The container must reach the collector"
    In the stock `compose.yml`, `indexer` and `proton-bridge` are only on the `internal` network. Attach `indexer` to a network the collector is on, or run the collector on `internal`. An unreachable collector costs a few log lines per interval and nothing else.

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset (off) | Base URL of the collector's OTLP/HTTP receiver. |
| `OTEL_SDK_DISABLED` | `false` | `true` turns export off even with an endpoint set. |
| `OTEL_METRIC_EXPORT_INTERVAL` | `60000` | Export interval in milliseconds. |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | `10000` | Timeout per export in milliseconds. |
| `OTEL_EXPORTER_OTLP_HEADERS` | unset | Extra headers, such as an auth token for the collector. |
| `OTEL_SERVICE_NAME` | per process | Overrides `service.name`. Leave it unset unless each container gets its own value, or the two processes share one name. |
| `OTEL_RESOURCE_ATTRIBUTES` | unset | Extra resource attributes, such as `deployment.environment=home`. |

The other standard `OTEL_*` exporter variables work too, because the SDK reads them itself.

`service.name` is `proton-faces-app` or `proton-faces-indexer`. `service.instance.id` is the process id, so sibling uvicorn workers keep separate series and a restart adds none.

## Metrics

Durations are histograms in seconds. A histogram already carries the request or run count, so there is no separate counter for it. Backends that flatten names, such as Prometheus-style stores, typically show `proton_faces_http_request_duration_bucket`, `_sum` and `_count`.

### `proton-faces-app`

| Metric | Type | Attributes | Meaning |
|--------|------|------------|---------|
| `proton_faces.http.request.duration` | histogram | `http.request.method`, `http.route`, `http.response.status_class` | Request count and duration, measured up to the last body byte. |
| `proton_faces.search.duration` | histogram | `search.type` (`text`, `face`), `search.phase` (`embed`, `match`) | Time to embed the query and to match it. Failed searches are not recorded. |
| `proton_faces.model.load.duration` | histogram | `model` (`clip`, `insightface`) | Model load time, once per process. Also emitted by the indexer. |

`http.route` is the route template, such as `/api/photos/{uid}/thumb`. Static files and unmatched paths are all `other`.

### `proton-faces-indexer`

| Metric | Type | Attributes | Meaning |
|--------|------|------------|---------|
| `proton_faces.indexer.photos.processed` | counter | `outcome` (`done`, `error`) | Photos through the recognition workers. Videos skip the workers: see the `fullres` job. |
| `proton_faces.indexer.photo.duration` | histogram | `outcome` | Recognition time per photo, including its share of a CLIP batch. |
| `proton_faces.indexer.errors` | counter | `stage` (`download`, `download_transient`, `thumbnail`, `decode`, `process`, `worker`, `fullres`) | Pipeline errors. `download_transient` is the bridge answering 429/5xx, so it also shows upstream rate limiting. |
| `proton_faces.indexer.queue.depth` | gauge | `queue` (`pending`, `fullres_backoff`) | In-memory queues only. The photo backlog by status is in `/api/status`. |
| `proton_faces.indexer.thread.alive` | gauge | `thread` (`sync`, `downloader`, `fullres`, `cluster`, `gps`, `albums`, `sidecar`, `worker`) | Live threads per role; `worker` counts all workers. Alert on `0`. |
| `proton_faces.indexer.job.duration` | histogram | `job`, `outcome` (`ok`, `error`, `transient`) | Run time per job, without pacing sleeps. |
| `proton_faces.indexer.job.failures` | counter | `job`, `kind` (`error`, `transient`) | Failed runs. |
| `proton_faces.indexer.job.last_run` | gauge | `job` | Unix time of the last finished run, whatever its outcome. |
| `proton_faces.indexer.job.last_success` | gauge | `job` | Unix time of the last successful run. Stale values are the alert to use for a stuck job. |

`job` is one of `sync` (full scan), `sync_tip` (the cheap check for new uploads), `cluster`, `gps`, `albums` or `fullres` (one download). The `last_*` gauges appear once a job has run since the process started.

## What is exported

Attribute values come from the fixed lists above, or from route templates. Photo ids, filenames, person names, search queries, usernames, signed URLs and tokens are never exported, and neither are error messages. Series count stays bounded by routes × methods × status classes, and by the fixed lists for the rest.

## Failure behaviour

Export runs on a background thread, so requests and indexing threads never wait for it. With the collector down, each interval logs a few `opentelemetry.exporter` warnings, the data for that interval is dropped, and shutdown takes at most about 5 s longer. A setup failure logs a warning and leaves metrics off.

One-shot commands such as `--backfill-gps-exif` don't export.

## Cost

Off: no measurable cost, since nothing is imported and the middleware is a single check per request. The packages add about 8 MB to the image. On: about 14 MB of RSS per process and 0.2 s at startup.

## Bridge

`proton-bridge` exports nothing. It compiles inside the Proton SDK monorepo, so the OpenTelemetry JS packages would have to go into that build, and it sits on the internal network only. Upstream rate limiting is visible from the indexer through the `download_transient` error stage and `transient` `fullres` failures.

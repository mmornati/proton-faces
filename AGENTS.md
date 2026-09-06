# AGENTS.md — proton-faces

> Guidance for LLM coding agents working on this repository. Read this before
> touching code. The project is a self-hosted face / object / location search
> engine that runs on top of Proton Drive Photos; see `README.md` for the
> product overview and `docs/reference/architecture.md` for the deep dive.

---

## 1. Repository map

| Path | What lives here | Build / run command |
|---|---|---|
| `app/src/` | Python FastAPI app, indexer, store, auth, demo bridge | `python main.py` (serves uvicorn) or `python indexer_main.py` (worker) |
| `app/src/static/` | Vanilla-JS single-page app — **one** `index.html`, no bundler | Served as a static file by the FastAPI app |
| `app/requirements-core.txt`, `app/requirements-ml.txt` | Split runtime deps (ML deps are lazy-loaded) | `pip install -r app/requirements-core.txt` |
| `bridge/src/` | Bun + Proton Drive SDK wrapper; compiled into the SDK CLI in the image | `bun install` (tests only) |
| `bridge/test/` | `bun test` unit tests for the pure bridge helpers | `bun test` (run from `bridge/`) |
| `scripts/` | Operator / fixture scripts (`create-admin.sh`, `backup.sh`, `verify-hardening.sh`, perf checks, demo rebuilder) | `bash scripts/<name>.sh` or `python scripts/<name>.py` |
| `tests/` | `pytest` unit suite for the Python side | `pytest` (run from repo root) |
| `docs/` | MkDocs site (Material theme); built with `--strict` | `mkdocs serve` / `mkdocs build --strict` |
| `compose.yml` | Production compose (3 services: `proton-bridge`, `indexer`, `app`) | `docker compose up -d` |
| `compose.demo-local.yml`, `compose.demo-prod-test.yml` | Demo profile overrides | `docker compose --profile demo up -d` |
| `SECURITY_HARDENING.md` | **Authoritative** threat model + fix matrix (F-01…F-14) | Read this before touching auth, /api/status, backups, or demo defaults |

Container roles (one-liner):

- `proton-bridge` — Bun + Proton SDK, the *only* component that talks to Proton. Read-only by construction. Reachable only on the compose internal network.
- `indexer` — Recognition pipeline (sync / downloader / fullres / cluster / gps / albums) + an internal `:8091` status endpoint the `app` container proxies to populate the footer.
- `app` — FastAPI search API + static SPA. Read-only against the SQLite index except for per-user favorites / tags / metadata. Optionally runs the indexer in-process when `RUN_INDEXER=1`.

---

## 2. Python conventions (`app/src/`)

- **Python 3.11+.** Files start with `from __future__ import annotations`; use PEP 604 unions (`X | None`) freely.
- **Ruff** (`pyproject.toml`): `line-length = 120`, `target-version = "py311"`, select `E, F, I, W`. `src = ["app/src", "tests", "scripts"]`. The CI step is `ruff check app/src tests scripts` — match that locally.
- **Test layout**: `tests/test_<module>.py` mirrors `app/src/<module>.py`. Use `pytest`. Coverage gate is **60%** (`--cov-fail-under=60` in `pyproject.toml` and CI). The `_reset_module_state` autouse fixture in `tests/conftest.py:55-88` clears every module-level singleton between tests — **add any new singleton you create to it**.
- **TestClient** via `fastapi.testclient`. The shared `client`, `password_hash`, `_seed_user`, `_seed_done_photo`, `_seed_face`, `_bearer(client)` helpers live in `tests/conftest.py` — reuse them, don't reinvent.
- **Env config** is read once at import time in `app/src/config.Settings()`. To redirect settings in tests, use `monkeypatch.setattr(config.settings, "name", path)` after the `app_settings` fixture sets up `tmp_path`. Do **not** mutate `os.environ` after `config.settings` has been instantiated.
- **SQLite is the only datastore.** All queries in `app/src/store.py` are parameterized. The schema lives in the `_SCHEMA` string and is created via `CREATE TABLE IF NOT EXISTS …` — additive changes only. New queries belong in `store.py` with a focused test in `tests/test_store.py`.
- **Singletons** are module-level (`bridge_client._bridge`, `clip._sess_vision / _sess_text / _tokenizer`, `cluster._person_means*`, `faces._app`, `geocode._rg`, `api._*_cache`). The naming convention is `<thing>_cache_<stat>` for caches with TTLs. Always make them reset-cleanly (see how `store._embedding_cache = None` is paired with `store._embedding_cache_ts = 0.0`).
- **Lazy ML imports.** `clip.py`, `faces.py`, and `cluster.py` must not import `onnxruntime` / `insightface` / `PIL` heavy bits at module top level — tests run without ML deps installed.
- **Threading model.** The indexer is a bag of daemon threads started in `indexer.py::start()`. They share the SQLite DB via WAL + 30 s `busy_timeout` and per-row atomic claims (`UPDATE ... WHERE status='new'`). Do not introduce locks across processes; do not add a second writer.
- **No `shell=True` anywhere** in subprocess calls. Use array-form args (`subprocess.run([...], check=False)`).
- **Comments are rare** in this repo. Prefer self-explanatory names; only add a comment when the *why* is non-obvious.

---

## 3. Bridge / TypeScript conventions (`bridge/`)

- **Bun + strict TS**. `bridge/package.json` is private / dev-only; the Docker image compiles `bridge.ts` into the Proton SDK CLI monorepo.
- **Pure helpers in `bridge/src/helpers.ts` and `bridge/src/rateLimit.ts`.** These files **MUST NOT** import `@protontech/drive-sdk`, `./init`, or any node/bun builtins — that keeps them unit-testable with `bun test` without the SDK. SDK-touching code lives only in `bridge/src/bridge.ts`. No third category.
- **No third category of helper.** If you need new shared logic, decide which file it goes in *before* you write it.
- **Bridge tests** live in `bridge/test/<name>.test.ts`, use `import { describe, expect, test } from 'bun:test'`, and run with `bun test` from the `bridge/` directory. CI runs the same.
- **Cache file glob** (`CACHE_FILE_GLOB` in `helpers.ts`) is intentionally future-proof — when adding a new SDK cache, you should not need to change `clear_cache` logic.

---

## 4. Docs / configuration conventions

- **MkDocs Material**, configured in `mkdocs.yml`. CI builds with `mkdocs build --strict --clean` — every warning fails. No broken internal links; no missing pages.
- **`.env.example` is the operator-facing key list.** Every env var referenced by `app/src/config.Settings.__init__` must appear here with its default and a one-line comment. Group by topic using the existing `# --- group ---` banners. Add new flags to `docs/reference/configuration.md` (the table is roughly grouped by topic, not strictly alphabetical).
- **Defaults must be safe-by-default.** When in doubt, ship the conservative value and let an opt-in flag flip it on. See §6.
- **No marketing fluff** in the docs. They are operator-facing and dense.

---

## 5. Build / test / lint commands

Match what CI does (`.github/workflows/test.yml`):

```bash
# --- Python ---------------------------------------------------------------
# Lint
ruff check app/src tests scripts

# Syntax-check the operator scripts that the CI step compiles
python -m py_compile \
  scripts/find-commons-faces.py \
  scripts/augment-demo-fixture.py \
  scripts/rebuild-faces-from-commons.py \
  scripts/fetch-demo-assets.py \
  scripts/capture_screenshots.py \
  scripts/perf_check_issue5.py \
  scripts/perf_check_live.py

# Unit tests with coverage (gate is 60%)
pytest --cov=app/src --cov-fail-under=60 --cov-report=term-missing

# Run a single test file or module
pytest tests/test_auth.py -q
pytest tests/test_auth.py --cov=app/src/auth --cov-report=term-missing

# --- Bridge ---------------------------------------------------------------
(cd bridge && bun install && bun test)

# --- Docs -----------------------------------------------------------------
mkdocs build --strict --clean   # mirrors CI; warns fail the build
```

Before claiming a change is done: run `ruff check`, run the targeted `pytest` for the module(s) you touched, and run the full `pytest --cov` if you changed anything that affects coverage.

---

## 6. Security invariants — DO NOT REGRESS

These come from `SECURITY_HARDENING.md` (the F-01…F-14 fix matrix). Each is already on `main`. Reverting any of these is a **critical regression** that requires a security review before merging.

| ID | Rule | Anchor |
|---|---|---|
| **F-01** | `/thumb`, `/full`, `/cover`, `/crop` are **secure by default**. They require a bearer token OR a valid `?sig=&exp=` signed URL. World-readable only when `DEMO_ALLOW_PUBLIC_THUMBS=1` is explicitly set (the `demo` compose profile does this). | `app/src/auth.py::allow_public_thumbs`, `signed_or_token`, `_is_auth_free` |
| **F-02** | Traefik security-headers middleware (`pf-ratelimit` on `/api/auth/login`) lives outside the repo in `/data/coolify/proxy/dynamic/`. Don't remove the docs reference. | `docs/reference/security-privacy.md` |
| **F-03** | `/api/status` `config` block requires a bearer token. | `app/src/api.py::status` |
| **F-05** | `DEMO_DISABLE_BACKUPS=1` returns 404 on `/api/admin/backup*`. | `app/src/auth.py::demo_disable_backups` |
| **F-05b** | `demo_disable_admin_user_management()` gates every `/api/admin/users*` route (list/create/patch/delete/logout). The flag is already plumbed through `_hardening_overrides`; without enforcement the dead-code route would let anyone with admin creds mint a persistent account that survives demo resets (fix #35). Self-service routes (`/api/auth/me`, `/api/auth/logout`) stay open. | `app/src/auth.py::demo_disable_admin_user_management`, `app/src/api.py::api_admin_*` |
| **F-06** | `DEMO_LOGIN_LOGS=1` is the only path that logs the demo admin password. Default OFF. | `app/src/auth.py::demo_login_logs` |
| **F-08** | Logout revokes **both** the access and the refresh token. | `app/src/auth.py`, `app/src/store.py::revoke_all_tokens` |
| **F-09** | `/api/search/face` is bounded by `FACE_SEARCH_MAX_UPLOAD_BYTES` (default 8 MB) and `FACE_SEARCH_MAX_IMAGE_PIXELS` (default 50 M). Decompression-bomb guard. | `app/src/api.py::search_face` |
| **F-11** | `/api/auth/refresh` **rotates** the refresh token — the old one is revoked before a new pair is minted. | `app/src/auth.py::refresh` |

Additional invariants not on the F-list but enforced by code:

- `DEMO_HARDENING_MODE` is the master switch: when an unsafe flag is unset and hardening is on, `auth._hardening_overrides` returns the **safe** value. Explicit env vars always win.
- Bcrypt cost 12 in `auth.hash_password`. Never lower it.
- Unknown-username login paths still run `bcrypt.checkpw` against a dummy hash to keep timing comparable — keeps the username-enumeration defense in place.
- `SIGNING_SECRET` env var is **required outside DEMO_MODE**: the app fails closed at startup if it's unset (no known default, no per-boot fallback in prod). Only `DEMO_MODE` may use a per-boot ephemeral secret (with a WARN log). Do not silently switch to a stable default.
- `_AUTH_FREE_PATHS` and `_AUTH_FREE_BINARY_SUFFIXES` (`app/src/auth.py:221-240`) are the only auth-free allowlists. **Mutating requests** on a binary suffix are not auth-free even when public thumbs are on (`_is_auth_free` short-circuits on `method not in ("GET", "HEAD")`).

### Adding a new binary-suffix endpoint

Either:

1. Use `signed_or_token` (bearer-or-signed-URL) as the FastAPI dependency — matches `/thumb /full /cover /crop` today.
2. Or add the new suffix to `_AUTH_FREE_BINARY_SUFFIXES`, **and** gate it behind `allow_public_thumbs()` in the route (not in `_is_auth_free` — `_is_auth_free` only checks GET/HEAD), **and** add a test in `tests/test_api.py::TestBinaryEndpointAuth` covering the four cases: no auth (401), bearer (200), signed URL (200), public-thumbs-on (200).

Never "just remove the auth check" to make a test pass. The `TestBinaryEndpointAuth` class is the canonical reference for the expected behavior.

### Adding a new demo-mode flag

Thread it through `auth._hardening_overrides` so `DEMO_HARDENING_MODE=1` flips it to the safe value when unset. Add a test in `tests/test_auth.py::TestAllowPublicThumbs` style that covers: default safe, explicit opt-in, hardening-mode-on, hardening-mode-on + explicit opt-in.

---

## 7. Indexing / ML invariants

- `WORKERS` ≤ `CPU cores - 1` (operator guidance; not enforced in code). Default in `compose.yml` is `3` on a 4-core box.
- `indexer` service is capped at `cpus: "2.0"` in `compose.yml`. Don't raise without a perf justification.
- Sync safety: `PROTON_SYNC_GRACE_CYCLES` (default `2`) prevents false deletions when Proton returns a partial timeline. Do not set to `0` without a hard reason.
- `FULLRES_RETRY_AFTER_SEC` and `FULLRES_BACKOFF_SEC` (defaults 600 / 900) protect the bridge SDK queue from being saturated by a stuck fullres loop. `FULLRES_DRAIN_INTERVAL_SEC=15` paces backlog drain to protect home bandwidth. Don't remove.
- `MIN_CLUSTER_SIZE=3` and `MIN_SAMPLES=2` are the post-noise-fix defaults; existing people rows are NOT retroactively affected when these change.
- `clip.py`, `faces.py`, `cluster.py` use lazy model loading. Don't import `onnxruntime` / `insightface` / heavy `PIL` paths at module top.
- `bridge_client.get_bridge()` is a process-local lazy singleton. Tests reset it via `bridge_client._bridge = None` in the autouse fixture.
- `BridgeTransientError` (raised in `bridge_client.py`) propagates 429 + `Retry-After` to the client. The fullres loop distinguishes transient vs permanent failures — preserve that.

---

## 8. Common pitfalls — how to add a feature

| Task | Where it goes | What to also do |
|---|---|---|
| New HTTP route that returns JSON | `app/src/api.py` | Pick the right dep: `require_user` (any authed user), `require_role("read" \| "write" \| "admin")`. Add a test class in `tests/test_api.py`. |
| New HTTP route that returns binary data | `app/src/api.py` | See §6. Use `signed_or_token`, or `_AUTH_FREE_BINARY_SUFFIXES` + gating + `TestBinaryEndpointAuth` test. |
| New env var | `app/src/config.Settings.__init__` | Document in `.env.example` (with default + comment) and `docs/reference/configuration.md` if user-visible. |
| New store query | `app/src/store.py` | Parameterize, add a focused test in `tests/test_store.py`. |
| New bridge helper | `bridge/src/helpers.ts` (pure) or `bridge/src/bridge.ts` (SDK-touching) | Add a `bridge/test/<name>.test.ts` if the pure side. |
| New demo-mode flag | `app/src/auth.py::_hardening_overrides` | Add to `_hardening_overrides`'s allowlist, add a test in `tests/test_auth.py`. |
| New singleton | anywhere | Add a reset line to `tests/conftest.py::_reset_module_state`. |
| New docs page | `docs/<area>/<page>.md` | Add to `mkdocs.yml` `nav:`, ensure internal links resolve, run `mkdocs build --strict`. |
| New compose service / env | `compose.yml` | `mem_limit`, `pids_limit`, `cap_drop: ["ALL"]`, `security_opt: ["no-new-privileges:true"]` are the baseline. Justify changes in a comment. |
| New ML model or heavy dep | `app/requirements-ml.txt` | Lazy-load in the using module. Confirm it doesn't get imported at top of `app/src/config.py`, `app/src/main.py`, or `tests/conftest.py`. |

---

## 9. Git / PR conventions

- **Conventional commit subjects**: `feat(scope):`, `fix(scope):`, `perf:`, `test:`, `docs:`, `refactor:`, `chore:`. Recent history uses lowercase scope: `people`, `bridge`, `indexer`, `deploy`, `demo`, `fullres`, `compose`, `ui`, `docker`, `people-merge`.
- **Branch prefix matches type**: `feat/`, `fix/`, `perf/`, `feature/`, `docs/`. Examples from recent history: `feat/test-infra`, `fix/auth-free-binary-endpoints-default`, `perf/suggest-embeddings-cache`.
- **PRs are squash-merged** (look at `git log --merges`). Keep commit history linear inside a PR; do not mix squashed and unsquashed commits.
- **Don't commit secrets.** `credentials/`, `.env`, `auth-session.json` are git-ignored. Never log plaintext passwords, never commit a Proton session, never paste a token in a test.
- **Don't commit unless the user explicitly asks.** Open a PR-ready state; let the user drive the merge.
- **Don't force-push, amend published commits, skip hooks, or run interactive git.** If a commit fails a hook, fix the underlying issue and create a new commit.
- **`.gitignore` covers** `.env`, `.coverage`, `.pytest_cache`, `.ruff_cache`, `.venv`, `__pycache__`. If you generate a new artifact class, add it to `.gitignore`.

---

## 10. Things an LLM must never do

1. **Edit `SECURITY_HARDENING.md` to mark a finding as resolved** without an actual code fix merged and the corresponding tests added.
2. **Bypass auth in a test by removing `require_user` from a route.** Use `monkeypatch.setenv("DEMO_ALLOW_PUBLIC_THUMBS", "1")` (only for tests of the public-thumbs code path) or pass `headers=_bearer(client)` instead.
3. **Flip `DEMO_ALLOW_PUBLIC_THUMBS` to default `1`.** It must stay default `0`; only the `demo` compose profile sets it to `1`.
4. **Lower bcrypt cost** below 12.
5. **Remove the `SIGNING_SECRET` warning** or switch `_signing_secret` to a stable default without an env-var override.
6. **Add a new module-level singleton** without also resetting it in `tests/conftest.py::_reset_module_state`.
7. **Raise `compose.yml` service resource limits** (`mem_limit`, `pids_limit`, `cpus: "2.0"` for `indexer`) without a perf-justified reason documented in the commit body.
8. **Mutate a new env var at runtime** in `app/src/main.py` / `api.py` — `Settings` is read once at import. If you need a runtime-tunable, add it to `indexer.get_sync_config()` / `set_sync_config()` (the one runtime-overridable config surface) or document it as restart-required.
9. **Use `shell=True` in subprocess calls** anywhere in `app/src/` or `scripts/`. Array-form args only.
10. **Force-push, amend, or skip hooks.** If the user didn't ask, don't.
11. **Invent a new branch / PR / issue** without the user explicitly asking. If the work is in-flight on a branch, continue on that branch; do not create parallel work.

---

## 11. Quick reference — common commands

```bash
# run a single test file
pytest tests/test_auth.py -q
# with coverage on a module
pytest tests/test_auth.py --cov=app/src/auth --cov-report=term-missing

# regenerate the demo fixture faces from Wikimedia Commons
python scripts/rebuild-faces-from-commons.py

# fetch demo assets
python scripts/fetch-demo-assets.py

# check hardening against a running stack
bash scripts/verify-hardening.sh

# start the public demo locally
docker compose --profile demo up -d

# create the first admin in a fresh deploy
ADMIN_PASSWORD='…' docker compose exec app python main.py --create-admin <username>
```

---

## 12. Key file anchors

When in doubt, these are the right files to read first:

- `README.md` — product overview, capabilities matrix, links to the docs site.
- `SECURITY_HARDENING.md` — threat model + the F-01…F-14 fix matrix. **Read before touching anything auth- or security-adjacent.**
- `docs/reference/architecture.md` — the 30-second architecture diagram and the "why three containers" rationale.
- `app/src/main.py:39-76` — `--create-admin` / `--reset-password` one-shot admin commands.
- `app/src/indexer_main.py:34-99` — indexer process entry, signal handling, status server.
- `app/src/indexer.py:810-862` — `start()` and the thread layout of the indexer.
- `app/src/api.py` — single-file FastAPI app. Routes are roughly grouped: auth, photos, faces/people, search, admin, status, sign.
- `app/src/auth.py:84-128` — `_hardening_overrides` and the secure-by-default contract.
- `app/src/auth.py:221-260` — `_AUTH_FREE_PATHS`, `_AUTH_FREE_BINARY_SUFFIXES`, `_is_auth_free`.
- `app/src/auth.py:262-334` — `signed_or_token` + `require_user`. Copy this pattern for any new binary route.
- `app/src/store.py:14-56` — `_SCHEMA` (additive changes only).
- `app/src/config.py:37-127` — every env var in one place.
- `app/src/demo.py` — the `DEMO_MODE=1` substitute for the Proton bridge.
- `tests/conftest.py:55-88` — the singleton-reset pattern to mimic when you add a new one.
- `bridge/src/bridge.ts:31-60` — the boundary where SDK-touching code is allowed.
- `bridge/src/helpers.ts:1-7` — the "pure helpers only" rule.
- `compose.yml` — three-service layout, resource limits, `RUN_INDEXER` fallback.
- `pyproject.toml` — ruff config, pytest config, coverage gate.
- `.github/workflows/test.yml` — the CI commands an LLM should mirror locally.

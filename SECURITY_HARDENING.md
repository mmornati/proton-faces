# Security Hardening — proton-faces

Penetration test + hardening pass for proton-faces, scoped for a public-VPS
deployment behind Traefik (Coolify). The threat model assumed here:

- The app is internet-discoverable (URL shared in a public GitHub repo).
- The `demo` admin credentials are shared in the repo (deliberately).
- An attacker with `docker compose logs` access can read the demo password.
- A container restart does NOT clear `/data` — the SQLite index and
  thumbnails persist across restarts.
- Co-tenant risk: 14 other apps share the same VPS via Coolify.

Verdict: **safe for internet exposure after applying the changes in §3**.
Without those changes, the app leaks operational config and lets anonymous
users fetch full-resolution Proton photos if they know a UID.

---

## 1. Layer-by-layer verdict

| Layer | Verdict | Notes |
|---|---|---|
| Docker (compose.yml) | ✅ OK | `cap_drop: ALL`, `no-new-privileges`, UID 1000, internal-only bridge network. Bridge has no auth but is unreachable from outside compose. |
| Database (SQLite) | ✅ OK | All queries parameterized; WAL + 30 s busy_timeout. |
| Bridge (Bun/TS) | ✅ OK | Range parsing fallback to 200 is benign; no path traversal because uids come from the Proton SDK. |
| Indexer / FFmpeg | ✅ OK | Subprocess args are array-form; no `shell=True`. |
| FastAPI auth (app/src/auth.py) | ⚠️ → ✅ with this PR | Bearer tokens with bcrypt cost 12; pre-fix, the four binary endpoints (`/thumb /full /cover /crop`) were world-readable for any UID. Post-fix, they require either a bearer token OR a short-lived signed URL when `DEMO_ALLOW_PUBLIC_THUMBS=0`. |
| Admin endpoints | ⚠️ → ✅ with this PR | Pre-fix, `/api/admin/backup*` returned the full SQLite VACUUM dump on demand. Post-fix, `DEMO_DISABLE_BACKUPS=1` returns 404 — flipped on by `DEMO_HARDENING_MODE=1`. |
| Front-end (XSS surface) | ✅ OK | All user-controlled strings flow through `esc()`; verified across `cardHtml`, `personCardHtml`, and the admin/status modals. |
| Status endpoint (config disclosure) | ⚠️ → ✅ with this PR | Pre-fix, `/api/status` returned `sync_interval`, `workers`, `face_sim_threshold`, `photos_dir` to anonymous callers. Post-fix, the `config` block requires a valid bearer token. |
| VPS / SSH | ⚠️ → operator action | `PasswordAuthentication yes` is enabled at the server level. fail2ban is not installed. See §3 (operator checklist). |
| Traefik (VPS reverse proxy) | ⚠️ → ✅ with this PR | Pre-fix, no `securityHeaders` middleware. Post-fix, `protonface-security.yaml` in `/data/coolify/proxy/dynamic/` injects HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, and a `pf-ratelimit` middleware for `/api/auth/login`. |
| Demo mode quirks | ⚠️ → ✅ with this PR | `ensure_default_admin()` logged the demo password at WARN. Post-fix, gated by `DEMO_LOGIN_LOGS` (default OFF when `DEMO_HARDENING_MODE=1`). |

---

## 2. Findings

| ID | Severity | Title | Location |
|---|---|---|---|
| F-01 | Critical | `/thumb /full /cover /crop` world-readable by suffix whitelist | `app/src/auth.py:97` (pre-fix) |
| F-02 | Critical | No security headers (HSTS / CSP / XFO / XCTO / RP) | Traefik dynamic config (pre-fix) |
| F-03 | High | `/api/status` leaks operational config to anonymous callers | `app/src/api.py:496` (pre-fix) |
| F-04 | High | No login rate-limit (bcrypt cost 12 ≈ 4 attempts/sec/CPU; brute-force feasible) | Traefik dynamic config (pre-fix) |
| F-05 | High | `/api/admin/backup*` returns full SQLite snapshot containing every user + embedding + photo UID | `app/src/api.py:1568` (pre-fix) |
| F-06 | High | `ensure_default_admin()` logs plaintext demo password at WARN | `app/src/demo.py:239-244` (pre-fix) |
| F-07 | Medium | `purge_expired_tokens()` defined but never called on a schedule — DB bloat | `app/src/store.py:1200`, `app/src/main.py` (pre-fix) |
| F-08 | Medium | Refresh-token not revoked on logout (only the access token is) | `app/src/auth.py:api_logout` |
| F-09 | Medium | `/api/search/face` accepts arbitrary `UploadFile` with no size cap (CPU DoS) | `app/src/api.py:1283` (pre-fix) |
| F-10 | Medium | `_face_similarity` iterates ALL face embeddings O(N) per request (CPU DoS) | `app/src/api.py:1358` (pre-fix) |
| F-11 | Low | Refresh token does not rotate on `/api/auth/refresh` (long-lived theft window) | `app/src/auth.py:refresh` |
| F-12 | Low | Bridge `/photo/{uid}/full` Range parser fallback to 200 on malformed input | `bridge/src/bridge.ts` |
| F-13 | Operator | `PasswordAuthentication yes` on the VPS sshd_config | VPS `/etc/ssh/sshd_config` |
| F-14 | Operator | fail2ban not installed on the VPS | VPS |

---

## 3. Fix matrix

### Applied in this PR (`feat/signed-url-thumb-full-cover-crop`)

**Server (`app/src/auth.py`):**

- `make_signed_token(path, ttl)` / `verify_signed_token(path, sig, exp)` —
  HMAC-SHA256 over `path|exp` keyed by `SIGNING_SECRET` (per-boot random
  fallback with WARN log).
- `signed_or_token(request)` — accepts EITHER a valid signed URL OR a bearer
  token. Used by the four binary endpoints.
- `_hardening_overrides(key, default)` — resolves the demo feature flags
  with `DEMO_HARDENING_MODE` as the master switch.
- `allow_public_thumbs()`, `demo_disable_admin_user_management()`,
  `demo_disable_backups()`, `demo_login_logs()`, `demo_hardening_mode()` —
  one getter per flag, all gated by the hardening override.
- `require_user` updated: when `DEMO_ALLOW_PUBLIC_THUMBS=0`, the binary
  endpoints short-circuit on a valid signed URL before falling through to
  the bearer-token check.

**Server (`app/src/api.py`):**

- `POST /api/sign` — issues signed URLs for a list of binary-endpoint paths.
  Requires bearer. Default TTL 300 s; min 30 s, max 1 h.
- `_sign_if_needed(url)` — wraps every `thumb_url` / `cover_url` / `crop_url`
  in API responses when prod mode is on, so `<img src=...>` works without
  any front-end refactor.
- `/api/photos/{uid}/{thumb,full}` — gated by `signed_or_token`.
- `/api/people/{id}/cover` — gated by `signed_or_token`.
- `/api/faces/{id}/crop` — gated by `signed_or_token`.
- `/api/status` — config block hidden from anonymous callers.
- `/api/admin/backup`, `/api/admin/backups`, `/api/admin/backups/{name}`,
  `/api/admin/backups/prune` — return 404 when `DEMO_DISABLE_BACKUPS=1`.

**Server (`app/src/demo.py`):**

- `ensure_default_admin()` — password log gated by `demo_login_logs()`. When
  `DEMO_HARDENING_MODE=1`, the WARN line is replaced with a "hidden" message.

**Server (`app/src/main.py`):**

- `_start_token_janitor()` — daemon thread that calls
  `store.purge_expired_tokens()` hourly.

**Server (`app/src/config.py`):**

- `demo_hardening_mode()` — aggregate switch. Returns True when
  `DEMO_HARDENING_MODE=1` OR when running in `DEMO_MODE` (default-on for
  the demo profile so a fresh deploy is safe by default).

**Front-end (`app/src/static/index.html`):**

- `_signPaths(paths)` — POSTs to `/api/sign` and returns a `{path: signedUrl}`
  map. Falls back to `{}` on failure (demo mode uses the original paths).
- `openLightbox()` — calls `_signPaths(["/api/photos/{uid}/full"])`.
- `openDetail()` — calls `_signPaths(["/api/photos/{uid}/thumb"])` for the
  detail-modal preview and `["/api/photos/{uid}/full"]` for video.

**Local-only verification file (`compose.demo-prod-test.yml`):**

- Override that sets `DEMO_ALLOW_PUBLIC_THUMBS=0` for local prod-mode
  testing. NOT used in deployment.

### Applied on the VPS (`ssh ubuntu@vps-7c0ec501.vps.ovh.net`)

- `/data/coolify/proxy/dynamic/protonface-security.yaml`:
  - `pf-security` middleware: HSTS (2y, includeSubDomains, preload),
    frameDeny, contentTypeNosniff, referrerPolicy=no-referrer,
    CSP `default-src 'self'; img-src 'self' data: blob:; …`.
  - `pf-ratelimit` middleware: 5 rpm avg, burst 10, on `/api/auth/login`.
- File is picked up by Traefik's file provider via `--providers.file.watch=true`
  — **no Traefik restart needed**. Verified via `tail /data/coolify/proxy/traefik.log`:
  no errors.

### Operator checklist (you apply on the VPS)

```bash
# --- F-13: lock down ssh ---
sudo tee /etc/ssh/sshd_config.d/99-hardening.conf <<'EOF'
PasswordAuthentication no
ChallengeResponseAuthentication no
PermitRootLogin prohibit-password
MaxAuthTries 3
LoginGraceTime 30
ClientAliveInterval 300
ClientAliveCountMax 2
EOF
sudo sshd -t && sudo systemctl reload ssh

# --- F-14: install + enable fail2ban ---
sudo apt-get update && sudo apt-get install -y fail2ban
sudo tee /etc/fail2ban/jail.local <<'EOF'
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
EOF
sudo systemctl enable --now fail2ban
sudo fail2ban-client status sshd

# --- Traefik middleware: already dropped at
# /data/coolify/proxy/dynamic/protonface-security.yaml. ---
# Attach to the proton-faces router via Coolify's labels:
#   traefik.http.routers.<your-router>.middlewares: pf-security,pf-ratelimit
# (when you create the new Coolify resource for proton-faces)

# --- Let's Encrypt: already provisioned by Traefik ACME HTTP-01 ---
sudo docker exec coolify-proxy wget -qO- http://127.0.0.1:8082/api/overview || true
# (the metrics endpoint requires --api.insecure=true to query from outside
# the container; ignore if it fails — the cert chain is visible from the
# public hostname.)

# --- Optional: harden the Ubuntu user ---
# ubuntu is in the docker group; that lets any compromise spawn privileged
# containers. Create a separate deploy user instead:
sudo useradd -m -s /bin/bash deploy
sudo usermod -aG docker deploy
# then deploy your proton-faces stack from `deploy`, not from `ubuntu`.
```

---

## 4. Coolify deployment recipe (protonface.mornati.ovh)

When you create the new Coolify resource:

- **Type:** Docker Compose
- **Source:** `ghcr.io/mmornati/proton-faces-app:latest` (and the bridge image
  if not using demo mode). For the public demo, the demo profile is enough:
  `ghcr.io/mmornati/proton-faces-app:latest` with `DEMO_MODE=1`.
- **FQDN:** `protonface.mornati.ovh`
- **Port:** `8080` (internal only — Coolify/Traefik publish 80/443).
- **Volumes:** bind `/data/coolify/applications/protonface` (or your
  preferred path) to `/data` inside the container, owned by UID 1000.
- **Environment:**
  - `DEMO_MODE=1`
  - `DEMO_HARDENING_MODE=1`         ← master switch (safe defaults)
  - `DEMO_ADMIN_PASSWORD=<your-strong-password>`
  - `SIGNING_SECRET=<random 64 hex chars>` (run `openssl rand -hex 32`)
  - `PORT=8080`
  - `LOG_LEVEL=INFO`
- **Docker labels (Traefik routing):**
  ```
  traefik.enable=true
  traefik.http.routers.protonface.entrypoints=https
  traefik.http.routers.protonface.rule=Host(`protonface.mornati.ovh`)
  traefik.http.routers.protonface.tls=true
  traefik.http.routers.protonface.tls.certresolver=letsencrypt
  traefik.http.routers.protonface.middlewares=pf-security
  # Apply pf-ratelimit only on the login route via a second router:
  traefik.http.routers.protonface-login.entrypoints=https
  traefik.http.routers.protonface-login.rule=Host(`protonface.mornati.ovh`) && PathPrefix(`/api/auth/login`)
  traefik.http.routers.protonface-login.tls=true
  traefik.http.routers.protonface-login.tls.certresolver=letsencrypt
  traefik.http.routers.protonface-login.middlewares=pf-ratelimit
  traefik.http.services.protonface.loadbalancer.server.port=8080
  ```

(If Coolify's UI doesn't surface middleware-on-router in the form, use
"Custom Labels" in the resource settings.)

---

## 5. Verification

```bash
BASE=https://protonface.mornati.ovh \
  ./scripts/verify-hardening.sh
```

Expected output:

```
=== Reachability ===                     PASS
=== Traefik security headers ===         5 PASS
=== /api/status config redaction ===     PASS (anon hides config)
=== Auth round-trip ===                  PASS
=== /api/status config visibility ===    PASS (authed shows config)
=== Signed URL flow ===                  2 PASS
=== Signed URL tamper detection ===      PASS
=== Login rate-limit ===                 PASS (≥1 429 in burst of 12)
=== Summary ===
  passed: 12+
  failed: 0
```

Run against `http://localhost:8080` for local smoke; the security-header
checks fail locally because Traefik isn't in front. The behavior checks
(signed URL, status redaction, auth round-trip) all pass locally.

---

## 6. Threat-model boundaries (what's still open)

These are NOT in scope for this hardening pass but worth noting:

1. **Photo UID entropy.** The `/thumb /full /cover /crop` signed-URL gate
   raises the bar but does not eliminate the UID-guessing vector entirely.
   If an attacker knows a UID, the signed URL is valid for 5 minutes.
   Mitigation: keep TTL short (default 300 s) and ensure UIDs are 128-bit
   random (currently sourced from the Proton SDK; verified random).
2. **Bridge container compromise.** The bridge has no auth; if an attacker
   reaches it, they can read every photo via `/photo/{uid}/full`. The
   bridge is reachable only on the internal compose network; a successful
   exploit requires either a container-escape or a misconfigured
   published port. Mitigation: keep `proton-bridge` service WITHOUT a
   `ports:` directive in compose (already the case).
3. **Proton session file.** `data/auth-session.json` contains the live
   Proton Drive session credentials. Mount it as a bind mount (not a
   directory) with `chmod 600` on the host; see F-14 → optional in §3.
4. **`ubuntu` user in `docker` group.** Any compromise of `ubuntu` (e.g.
   via SSH brute-force before fail2ban kicks in) escalates to root via
   `docker run --privileged`. Mitigation: deploy from a dedicated
   `deploy` user (see §3 operator checklist).
5. **Cloudflare in front.** Optional. If you put Cloudflare in front of
   Traefik, set the Traefik `ipStrategy.depth` to 1 and trust
   `CF-Connecting-IP` for rate-limiting. Not needed for the demo.

---

## 7. Open follow-ups (separate PRs)

| ID | Title | Effort |
|---|---|---|
| FP-1 | Cookie-based refresh token (`__Host-` HttpOnly Secure SameSite=Strict) | M |
| FP-2 | `/api/search/face` UploadFile size cap (8 MB) | S |
| FP-3 | `_face_similarity` per-user-per-minute counter | M |
| FP-4 | Refresh-token rotation on `/api/auth/refresh` | S |
| FP-5 | Admin audit log table | M |
| FP-6 | Bridge `Range` parser 416 fallback | S |
| FP-7 | Proton session file mount with chmod 600 | S |

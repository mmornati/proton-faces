# Security & privacy

Proton Faces is **privacy-first by design**: no telemetry, no cloud APIs, and the only network calls go to Proton's servers via the bridge container. All ML runs locally through ONNX Runtime on CPU.

## What's on disk

| Path | Content | Sensitivity |
|---|---|---|
| `DATA_DIR/thumbs/` | 512px WebP thumbnails | Low. These are downscaled previews; not full-res. |
| `DATA_DIR/index.sqlite3` | The local index (photos, faces, clips, people, users, tokens) | Medium. Photo metadata + your local edits + bcrypt password hashes + bearer tokens (random hex). |
| `DATA_DIR/crops/` | Face-crop JPEGs (`{face_id}.jpg`) | Low. Cached subset of the thumbs. |
| `DATA_DIR/_backups/` | `index-<UTC-stamp>.sqlite3` snapshots | Same as the live DB. |
| `DATA_DIR/gps_sha1_cache.json` | `sha1 → (lat, lng)` map | Low. |
| `DATA_DIR/admin_config.json` | Auto-backup schedule | None. |
| `DATA_DIR/models/` | InsightFace buffalo_l + CLIP ViT-B/32 | None. Public ML model weights. |
| `credentials/auth-session.json` | Your Proton session | **High.** Treat like a password. |

Nothing else is written to disk. No temporary caches, no logs of photo content, no analytics files.

## What's NOT on disk

- **Full-resolution photos.** The bridge downloads them on demand when you click **View full resolution**; the bytes stream through the FastAPI response and are discarded by the client. The fullres loop downloads HEIC / videos once, generates a 512px thumbnail, and discards the full-res bytes immediately.
- **Uploaded face-search images.** The face-search-by-example endpoint reads the multipart upload into memory, embeds it, and discards it. The image is never written to disk or sent to Proton.
- **Telemetry, analytics, or any outbound calls other than Proton.** There are no analytics SDKs, no Sentry/Datadog calls, no remote logs.

## Network surface

| From | To | What |
|---|---|---|
| Browser → `app:8080` | localhost / your reverse proxy | HTTPS / JSON / static files |
| `app:8080` → `proton-bridge:8090` | compose internal network | `timeline`, `timeline_ids`, `nodes`, `albums`, `thumbnails`, `photo/{uid}/full` |
| `app:8080` → `indexer:8091` | compose internal network (no host port mapping) | `/status` (live state for the footer) |
| `proton-bridge` → Proton Drive | HTTPS to `drive-api.proton.me` | Auth + timeline + thumbnail fetch (read-only) |

The bridge's only outbound destination is Proton's API. Nothing else leaves the host.

If you put a reverse proxy in front of `:8080` (Traefik, Caddy, nginx), it's your responsibility to terminate TLS and gate the public side.

## The bridge is read-only

`proton-bridge` is built inside the [Proton Drive SDK monorepo](https://github.com/mmornati/proton-drive-sdk) and pins a specific release tag. It exposes only `GET /timeline`, `POST /nodes`, `GET /albums`, `POST /thumbnails`, `GET /photo/{uid}/full`, `GET /health`.

There is no `POST`/`PUT`/`DELETE` for write operations. There is no `upload`, no `delete`, no `move`. The session file is mounted read-only. The data dir is shared with the indexer for thumbnail caching only — not for write-back.

## Multi-user auth

Proton doesn't publish an OAuth/OIDC provider, so each family member gets a **local account**:

- **Username + bcrypt password.** Cost 12, ~250 ms per hash on a modern CPU.
- **Opaque bearer tokens** — 32 random bytes, hex-encoded. Not JWT.
- **Access tokens** — TTL `AUTH_ACCESS_TTL` (default 8 h).
- **Refresh tokens** — TTL `AUTH_REFRESH_TTL` (default 30 d).
- Tokens are stored in `auth_tokens` (SQLite), not in cookies. The browser keeps them in `localStorage` under `pf.auth`.
- Failed username probes run a dummy bcrypt hash to keep wall time comparable to a real failure (defense against username enumeration).
- Password reset (`--reset-password`) **revokes every active token** for that user.

### Roles

| Role | Can browse | Can edit metadata | Admin area |
|------|:-:|:-:|:-:|
| `read` | ✅ | ❌ | ❌ |
| `write` | ✅ | ✅ | ❌ |
| `admin` | ✅ | ✅ | ✅ |

A `write` user inherits everything a `read` user can do. An `admin` inherits everything a `write` user can do.

### What can be edited by `write`+ users

- Tag, untag, archive, unarchive, hide, unhide, favorite, unfavorite any photo.
- Rename a person / merge two people.
- Assign a face to a person / create a new named person.
- Unassign a face.
- Run face search by example.

### What only `admin` can do

- Create / edit / delete users.
- Reset another user's password.
- Sign another user out of every device.
- Read the admin overview, run health checks.
- Create / delete / prune backups, change the auto-backup schedule.

## Bearer token validation

Every API request (except the auth-free ones listed in the [API reference](api.md)) requires `Authorization: Bearer <token>`. The server:

1. Looks up the token in `auth_tokens`.
2. Checks it's an `access` token (refresh tokens only work at `/api/auth/refresh`).
3. Checks it's not expired.
4. Checks the user isn't disabled.
5. Returns a 401 otherwise — with a `WWW-Authenticate: Bearer` header.

There is no automatic token rotation; the access token stays valid until expiry or explicit revocation (`POST /api/auth/logout` or `POST /api/admin/users/{id}/logout`).

## CORS / CSRF

There is no CORS layer in the app — the web UI is served from the same origin as the API. If you put a reverse proxy in front, configure it to **not** add CORS headers; the same-origin model is intentional.

## Threat model — what you should worry about

| Threat | Mitigation |
|---|---|
| Someone reads your disk | The DB has bcrypt password hashes + opaque bearer tokens. Photos aren't on disk; only thumbs. Revoke all tokens + change passwords if you suspect compromise. |
| Someone sniffs your network | All traffic to Proton is TLS. The compose-internal traffic is plaintext HTTP but on a private network. Put a TLS-terminating reverse proxy in front of `:8080` for remote access. |
| Someone accesses `localhost:8080` while you're away | Bearer tokens are required for every non-binary endpoint. The `?` overlay shows who's signed in. Sign out from another device via the admin **Users** tab. |
| Bug in the bridge | The bridge is read-only by construction; even a bug can't write back to Proton. Pin the SDK tag in your fork if you're paranoid. |
| Malicious auth-session.json | Only the bridge reads it. It authenticates to Proton and streams the encrypted timeline — it can't be used to delete or upload. Revoke the session on Proton's website to invalidate. |

## What we don't do

- We do not upload anything to Proton (read-only bridge).
- We do not upload anything to any third party (no telemetry).
- We do not store full-resolution photos.
- We do not log photo bytes to disk (the `processed_at` field is the only timestamp).
- We do not require GPU access.
- We do not require any cloud account other than your Proton Drive session.

## Reporting a vulnerability

Open an issue on GitHub, or email the maintainer (see the GitHub profile). Please don't disclose security issues publicly until a fix is out.

---

**Next:** [FAQ](faq.md) answers the common questions.

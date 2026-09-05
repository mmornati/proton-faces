# REST API

Every endpoint exposed by the FastAPI app. All routes require authentication unless noted. Roles: `read` < `write` < `admin`.

Authentication is via `Authorization: Bearer <token>`. Get a token from `POST /api/auth/login` (username + password). The bearer token TTL is `AUTH_ACCESS_TTL` (8 h default). Refresh via `POST /api/auth/refresh` with a refresh token (TTL `AUTH_REFRESH_TTL`, 30 d default).

## Conventions

- All responses are JSON unless the endpoint is documented as binary.
- Timestamps are epoch seconds (integer) unless noted.
- Photo `uid` is a string (Proton's node uid).
- Person `id` is an integer (SQLite autoincrement).
- Errors are `{"detail": "<message>"}` with the appropriate HTTP status.

## Auth

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/auth/login` | none | `{username, password}` → `{access_token, refresh_token, expires_in, user}` |
| POST | `/api/auth/refresh` | none | `{refresh_token}` → new `{access_token, user}` |
| POST | `/api/auth/logout` | bearer | Revoke the token used for this request |
| GET | `/api/auth/me` | bearer | Current user (`{id, username, display_name, role}`) |
| GET | `/api/auth/limits` | none | Public; UI uses this for the login screen |

## Public status

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | none | `{ok: true, bridge: {reachable, loggedIn}}` |
| GET | `/api/status` | none | Aggregated status (footer + `?` overlay data) |

## Photos

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/photos?limit=200&offset=0` | read | Page of indexed photos (`{photos: [...]}`) |
| GET | `/api/photos?place=Lille` | read | Photos from a place |
| GET | `/api/photos?before=1716240000` | read | Photos captured before an epoch timestamp |
| GET | `/api/photos?only_favorites=true` | read | Your favorites |
| GET | `/api/photos?tag=dog` | read | Photos carrying a tag |
| GET | `/api/photos?include_archived=false` | read | Hide archived from the grid |
| GET | `/api/photos/archived` | read | Every archived photo |
| GET | `/api/memories?month=&day=` | read | "On this day" photos (`{month, day, photos}`) |
| GET | `/api/duplicates?limit=200` | read | Groups by sha1 (`{groups: [...]}`) |
| GET | `/api/photos/{uid}` | read | Single photo row |
| GET | `/api/photos/{uid}/meta` | read | Full metadata + live Proton node info + album names |
| GET | `/api/photos/{uid}/thumb` | none (binary) | 512px WebP, immutable cache |
| GET | `/api/photos/{uid}/full` | none (binary) | Full-resolution stream (HTTP Range supported) |
| PATCH | `/api/photos/{uid}` | write | Toggle `favorited` / `archived` / `hidden` (any subset) |
| GET | `/api/photos/{uid}/faces` | read | All face rows for this photo |
| GET | `/api/photos/{uid}/tags` | read | User tags |
| PUT | `/api/photos/{uid}/tags` | write | Replace the tag list (`{tags: [...]}`) |
| GET | `/api/photos/anchors` | read | Year-month anchors for the date rail |

## Tags

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/tags` | read | `{tags: [{name, count}, ...]}` |

## Albums

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/albums` | read | `{albums: [{uid, name, photo_count, start_ts, end_ts, cover_url}]}` |
| GET | `/api/albums/{album_uid}/photos` | read | Photos in an album |

## Places

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/places?limit=500` | read | `{places: [{place, city, count}]}` |
| GET | `/api/map?limit=1000` | read | `{markers: [{place, city, count, lat, lng, thumb_url}]}` |

## People & faces

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/people?q=&limit=200&offset=0` | read | List clusters, optional name filter |
| GET | `/api/people/{id}/cover` | none (binary) | Face-crop cover JPEG, immutable cache |
| GET | `/api/people/{id}/faces` | read | Every face of a person, with crop URLs (cover picker) |
| POST | `/api/people/{id}/cover` | write | Set cover photo `{face_id}` |
| GET | `/api/people/{id}/photos` | read | Photos containing this person |
| GET | `/api/people/{id}/map` | read | Per-person map markers |
| GET | `/api/people/duplicates?threshold=0.40&limit=50` | read | Suggested merges |
| POST | `/api/people/{id}/name` | write | Rename (auto-merges if name exists) |
| POST | `/api/people/{src}/merge` | write | Explicit merge `{target_id}` |
| GET | `/api/faces/unassigned?limit=500` | read | The unassigned queue |
| GET | `/api/faces/{id}/crop` | none (binary) | Face-crop JPEG |
| GET | `/api/faces/{id}/suggest?limit=5` | read | Ranked "who might this be" people (face→person-mean-embedding similarity) |
| POST | `/api/faces/{id}/person` | write | Assign (`{person_id}` or `{name}`); propagation kicks in |
| POST | `/api/faces/{id}/unassign` | write | Unassign from person |

## Search

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/search?q=dog&limit=100` | read | CLIP text search |
| POST | `/api/search/face` (multipart upload) | read | Face search by example |

## Admin: users

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/admin/users` | admin | List all users |
| POST | `/api/admin/users` | admin | Create a user (`{username, password, role, display_name?}`) |
| PATCH | `/api/admin/users/{id}` | admin | Edit display_name / role / disabled / password |
| DELETE | `/api/admin/users/{id}` | admin | Delete a user (refuses to delete the last admin) |
| POST | `/api/admin/users/{id}/logout` | admin | Revoke all that user's tokens |

## Admin: server ops

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/admin/overview` | admin | Server / disk / last backup summary |
| POST | `/api/admin/backup` | admin | Trigger a manual snapshot now |
| GET | `/api/admin/backups` | admin | List snapshots |
| DELETE | `/api/admin/backups/{name}` | admin | Delete a snapshot |
| POST | `/api/admin/backups/prune` | admin | Prune `{keep}` |
| GET | `/api/admin/schedule` | admin | Read the auto-backup schedule |
| PUT | `/api/admin/schedule` | admin | Update the schedule (`{enabled, hour, minute, keep}`) |
| POST | `/api/admin/checks` | admin | Run the seven health checks on demand |

## Static

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | none | The single-page web UI (vanilla JS) |
| GET | `/assets/*` | none | JS / CSS / images |

## Example: search by face

```bash
curl -X POST http://localhost:8080/api/search/face \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/face.jpg"
```

Response:

```json
{
  "results": [
    { "uid": "…", "thumb_url": "…", "score": 0.81, "favorited_by_me": false, "tags": [], "archived": false, "hidden": false, "kind": "image" },
    ...
  ],
  "total": 17
}
```

## Example: rename a person

```bash
curl -X POST http://localhost:8080/api/people/7/name \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice"}'
```

Response (newly created):

```json
{ "ok": true, "merged": false }
```

Or, when the name already exists:

```json
{ "ok": true, "merged": true, "target_id": 3, "photo_count": 42, "face_count": 18 }
```

---

**Next:** [Security & privacy](security-privacy.md) explains what's on disk, what's not, and how the auth model works.

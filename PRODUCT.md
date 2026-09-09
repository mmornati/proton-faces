# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary:** a self-hosting hobbyist who keeps personal photos in Proton Drive and wants private, searchable access without uploading any byte to a third party. They already run Docker on their own hardware, value control over their data, and tinker. Operates solo or with family on one install.
- **Secondary:** anyone evaluating the project before committing to self-host — the README demo profile (zero Proton credentials, fixture photos) is the evaluation path.

## Product Purpose

Proton Faces turns an end-to-end-encrypted Proton Photos library into a fully searchable archive — people, places, objects, and free-text — that runs entirely on the user's own machine. Nobody but the owner can see the photo if it stays in Proton; that same privacy is the reason searching must happen locally. Success: a user finds any photo ("who is this?", "where was it?", "when?", "what's in it?") in seconds, forever, with no third-party cloud touching the content.

## Positioning

A Google-Photos-style search experience over Proton Photos that is 100% private and offline-by-construction: the index and every ML model run on the owner's hardware, nothing is ever written back to Proton Drive, and there is no telemetry. The search quality is the product (RetinaFace + ArcFace face embeddings, HDBSCAN clustering, CLIP zero-shot text search, GPS reverse-geocoding) — not just an image viewer with filters.

## Operating Context

- Runs in Docker Compose with three services: `proton-bridge` (the only component that talks to Proton, read-only), `indexer` (recognition pipeline), and `app` (FastAPI search API + SPA). The `app` and `indexer` can also run in-process for a single-container demo.
- Self-hosters administer via env vars (`compose.yml` / `.env.example`), one-shot `--create-admin` / `--reset-password` commands, and an admin API with role-based access (`read` / `write` / `admin`).
- Demo mode runs the entire product against a fixture of CC0 photos with zero Proton credentials — the primary evaluation and development path.
- Photo grids, face tagging, places map (Leaflet + OSM), free-text search, albums, tags, duplicates, and a "who is this?" search-by-example flow.
- iPhone (HEIC) photos are decoded locally because Proton serves no preview.
- Browsed on desktop and mobile; the SPA is now a PWA (installable, offline app shell) with light and dark themes.

## Capabilities and Constraints

- **Capabilities:** people clustering and naming, search-by-example face matching, free-text CLIP search, GPS places with an interactive world map, albums, tags, unassigned-face queue, duplicates review, video indexing (detected, hidden from grids), favorites/archive per user, demo mode.
- **Constraints:** SQLite is the only datastore (single writer, WAL); indexer is a bag of daemon threads with per-row atomic claims; ML deps are lazy-loaded so the core app runs without them; every env var has a safe-by-default value.
- **Security posture is a product constraint:** binary photo endpoints are auth-gated (bearer or signed URL) unless the demo profile explicitly opens them; admin user management is disabled in demo mode; bcrypt cost 12; login rate-limited; refresh tokens rotate. See `SECURITY_HARDENING.md` (fix matrix F-01…F-14) for the authoritative threat model.
- **Terminology:** "fullres" = full-resolution preview; "bridge" = the Proton SDK boundary; "sync grace cycles" guard against false deletions on partial Proton timelines; "cluster" = HDBSCAN person grouping.

## Brand Commitments

- Name and wordmark: **proton·faces** (SPA title and PWA manifest) / "Proton Faces" (README). The pin badge `pill-local` reinforces the local-first identity.
- Voice in product copy and docs: direct, operator-facing, no marketing fluff, no telemetry claims made lightly (the "no telemetry" badge is a factual claim).
- Visual identity (established May 2025 rework, direction "Nocturne"): a dark-first cinematic world that follows the OS scheme with a persistent manual override; near-black graphite neutrals, one green accent, system font stack, mono accents for metadata. Light theme is a full first-class surface, not a fallback.
- The "demo is the door" experience: `demo` / `proton-faces` fixture login is the first thing a new user meets.

## Evidence on Hand

- `README.md` — product overview, capability matrix, demo video, Docker recipe.
- `SECURITY_HARDENING.md` — the authoritative threat model and fix matrix.
- `docs/` — operator-facing MkDocs site (quickstart, demo mode, face tagging, API, security & privacy, configuration).
- `docs/demo.mp4` — demo video (also hosted on Cloudinary).
- `AGENTS.md` — repository conventions and invariants for maintainers.
- No user testimonials, case studies, pricing, or press exist; future work must not fabricate them.

## Product Principles

1. **Privacy is the product, not a feature flag.** Nothing ever leaves the owner's hardware; auth, telemetry-absence, and signed URLs are treated as release-blocking invariants (F-01…F-14).
2. **Search is the job.** Careful about anything that buries finding-faster: clustering, tagging, querying, and browsing stay front and center.
3. **Safe by default.** Conservative defaults, opt-in for anything risky, demo mode as the honest evaluation path.
4. **Operate mode, not marketing.** Dense, scannable, operator-facing surfaces for people who own their software.
5. **Works where the owner is.** Real mobile support (not a shrunk desktop), offline-capable shell, and system-aware theming.

## Accessibility & Inclusion

- The SPA targets WCAG-usable contrast in both themes (verified during the Nocturne rework for text-on-background legibility).
- Light mode and dark mode are both complete, with the OS scheme as default and a persistent manual toggle; theme color meta matches each scheme.
- No product-specific requirement beyond this was confirmed.
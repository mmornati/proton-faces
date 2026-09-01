# User guide

The web UI is a single-page vanilla-JS application. There are nine top-level views plus the gear-icon admin modal, the "Search by example" modal, and the bottom status bar.

<div class="pf-cards" markdown>

<div class="pf-card" markdown>
### 📷 [Photos](photos.md)
The main infinite-scroll grid with date rail, "On this day", and the full-res detail panel.
</div>

<div class="pf-card" markdown>
### 🔍 [Search](search.md)
Zero-shot CLIP text search and "find photos of a person from a photo" face search.
</div>

<div class="pf-card" markdown>
### 👥 [People](people.md)
Auto-clustered persons, naming, merging, the per-person map, and the people duplicates finder.
</div>

<div class="pf-card" markdown>
### 🏷️ [Face tagging](face-tagging.md)
Clickable face boxes, naming one face auto-tags look-alikes, the unassigned queue.
</div>

<div class="pf-card" markdown>
### 🗺️ [Places](places.md)
Leaflet world map with clustered city markers and a city list.
</div>

<div class="pf-card" markdown>
### 🖼️ [Albums & tags](albums-tags.md)
Proton albums (read-only) and your free-form lowercase tags.
</div>

<div class="pf-card" markdown>
### ⭐ [Favorites & archive](favorites-archive.md)
Star, archive, hide, content-hash duplicates.
</div>

<div class="pf-card" markdown>
### ⚙️ [Admin area](admin.md)
Gear-icon modal: server info, health checks, backups, schedule, users.
</div>

<div class="pf-card" markdown>
### 🔎 [Status & diagnostics](status.md)
Bottom status bar + `?` overlay — every view shows live indexer state.
</div>

</div>

## Top-level navigation

The header bar has:

- **App title** (clickable, takes you to Photos)
- **View tabs** — Photos · ★ Favorites · Archive · People · Places · Albums · Tags · Duplicates · Unassigned
- **Search bar** — `<input id="q">` + **Search** button + **Search by example** button
- **⚙ gear** (admin only) — opens the admin modal

The bottom status bar shows:

- Bridge reachability (`Proton: connected` / `Proton: offline`)
- Indexer stats (`X/Y photos indexed · … people · … faces`)
- Last sync timestamp
- Press `?` for details

## Roles

The login modal signs you in as one of three roles:

| Role | Browse, search, map, albums, places, memories | Per-user ★ favorites | Tags, archive, hide, face naming, person rename/merge |
|------|:-:|:-:|:-:|
| **read** | ✅ | ✅ | ❌ |
| **write** | ✅ | ✅ | ✅ |
| **admin** | ✅ | ✅ | ✅ + admin area (users, backups, schedule, health) |

Admins are the only ones who see the gear icon and the **Users** tab inside People.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| <kbd>Enter</kbd> (in the search bar) | Run the current query |
| <kbd>Esc</kbd> | Close any open modal (photo detail, face popover, admin modal, status overlay) |
| `?` (clickable in the footer) | Open the **Status & diagnostics** overlay |

The UI is otherwise mouse/touch driven.

---

Pick a view from the cards above to dive in.

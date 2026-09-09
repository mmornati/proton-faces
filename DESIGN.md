# Design System: proton·faces — "Nocturne"

Impeccable design-system record, written from the shipped artifact (`app/src/static/index.html` May 2025). Tokens are normative; prose explains application.

---
name: proton-faces
description: Dark-first cinematic photo theatre — private search over Proton Photos
colors:
  bg: "#101114"
  bg2: "#15161a"
  panel: "#17181d"
  panel2: "#1e2026"
  text: "#f2f3f5"
  muted: "#9aa0ab"
  faint: "#5c616d"
  border: "#26282f"
  border2: "#32353e"
  accent: "#5ac272"
  accent-strong: "#7edb93"
  accent-soft: "rgba(90,194,114,.14)"
  on-accent: "#04140a"
  ok: "#5ac272"
  bad: "#f0645a"
  text-light: "#17191d"
  bg-light: "#f4f5f7"
  panel-light: "#ffffff"
  accent-light: "#1f9d4d"
  shadow:
    - "0 1px 2px rgba(0,0,0,.35), 0 16px 34px -18px rgba(0,0,0,.6)"
    - "0 1px 2px rgba(16,17,20,.08), 0 16px 34px -20px rgba(16,17,20,.25)"
  shadow-lg:
    - "0 2px 4px rgba(0,0,0,.4), 0 34px 80px -30px rgba(0,0,0,.75)"
    - "0 2px 4px rgba(16,17,20,.1), 0 34px 80px -34px rgba(16,17,20,.3)"
typography:
  display:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    fontSize: "20px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-.02em"
  title:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 680
    lineHeight: 1.2
    letterSpacing: "-.01em"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "'SF Mono', ui-monospace, 'JetBrains Mono', Menlo, Consolas, monospace"
    fontSize: "10.5px"
    fontWeight: 400
    letterSpacing: ".03em"
rounded:
  xs: "6px"
  sm: "8px"
  md: "12px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.sm}"
    padding: "8px 14px"
    fontSize: "13px"
    fontWeight: 620
  button-secondary:
    backgroundColor: "{colors.panel2}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: "8px 14px"
    border: "1px solid {colors.border}"
  button-danger:
    backgroundColor: "{colors.bad}"
    textColor: "#fff"
    rounded: "{rounded.sm}"
  card:
    backgroundColor: "{colors.bg2}"
    rounded: "{rounded.sm}"
  input:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    border: "1px solid {colors.border}"
    padding: "7px 12px"
  nav-link:
    textColor: "{colors.muted}"
    rounded: "{rounded.sm}"
    padding: "7px 10px"
  nav-link-active:
    backgroundColor: "{colors.accent-soft}"
    textColor: "{colors.accent-strong}"
    rounded: "{rounded.sm}"
  modal:
    backgroundColor: "{colors.panel}"
    rounded: "16px"
    border: "1px solid {colors.border}"
---

# Design System: proton·faces

## Overview

**Creative North Star: "The Photographic Theatre"**

Nocturne treats the browser as a private darkroom: near-black surfaces recede so photographs carry the light, a single signal-green accent marks exactly where the owner acts, and every piece of machine metadata is set in monospace like a contact sheet. The interface is a quiet projection surface for one person rifling through their own archive — dense but calm, cinematic but never decorative. It follows the OS light/dark preference by default with a persistent manual override, so it feels native whether the owner opens it at midnight or at noon.

The system is deliberately restrained: no gradients on surfaces, no colorful charts, no vendor flourishes. Depth is cheap here — tonal layering, hairline borders, one diffused shadow. Expression is reserved for the photography itself.

**Key Characteristics:**
- Dark-first but theme-complete: both themes are full first-class surfaces with equivalent contrast and structure.
- Single green accent with a strict nesting hierarchy (soft → base → strong) and one documented off-accent.
- Monospace metadata: every score, count, hash, timestamp, pill, and status reads as data.
- Edge-to-edge 4:3 photo density, tiles touching in mobile, generous in desktop.
- No decorative motion; the only moving parts are state responses (hover, selection, hover-reveal play badge).

## Colors

The palette is a graphite night spectrum plus one green accent; the light theme is its exact negative. Colors are tokenized as CSS custom properties under `:root` / `:root[data-theme="light"]` with a `prefers-color-scheme: light` intermediary for system-following devices.

### Primary
- **Signal Green** (`#5ac272` dark, `#1f9d4d` light): the single accent. Owns primary buttons, active nav state, focus borders, the status "ok" dot, local tags, and selection rings. Its rarity is the point.

### Secondary
- **Signal Green Strong** (`#7edb93` dark, `#17803e` light): the "on-accent" bright variant used for active nav labels, link-buttons, accent text tails on dark photo frosted chips, and highlighted mono values.

### Tertiary
- **Signal Green Soft** (`rgba(90,194,114,.14)` / `rgba(31,157,77,.12)`): the fill for selected rows, active nav, and hover autocomplete items — color presence without text contrast responsibility.

### Neutral
- **Night** (`#101114`): page background (dark).
- **Night-2** (`#15161a`): card-image slate / recessed backdrop.
- **Panel** (`#17181d`): modal, user-chip, statusbar-tint, input surfaces (dark).
- **Panel-2** (`#1e2026`): secondary buttons, disabled-ish fills (dark).
- **Bone** (`#f4f5f7`): page background (light).
- **Snow** (`#ffffff`): cards/panels (light).
- **Fog** / **Slate**: text triplet `#f2f3f5`/`#9aa0ab`/`#5c616d` (dark) and `#17191d`/`#6b7078`/`#9aa0ab` (light): primary text, secondary text, tertiary/faint text.
- **Hairstroke** (`#26282f`/`#e3e5ea`): the standard 1px structural border.

### Named Rules
**The One-Accent Rule.** Green appears on any single field as one role at one intensity, never both. A selected row uses accent-soft fill + accent-strong text; it does not also get a green border. A primary button is green-filled; its sibling is neutral panel. Two green elements on one screen are always the same element in two states.

**The Darkroom Rule.** In dark mode the photo is the light. Cards are one step darker than the page so image content glows, never competes with surrounding neutral fills.

## Typography

**Display Font:** System sans (SF Pro / Segoe UI / Roboto stack)
**Body Font:** Same system stack — one voice, size-contrast hierarchy.
**Label/Mono Font:** `"SF Mono", ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace`

One typeface family keeps the operator surface fast and native; type hierarchy is carried by weight, size and mono contrast, not by adding faces.

### Hierarchy
- **Display** (`700`, 20px, 1.2, `-.02em`): the topbar view title.
- **Title** (`680`, 15px, 1.2, `-.01em`): section heads, brand name, person names.
- **Body** (`400`, 14px, 1.45): default content; max `65ch` in prose blocks.
- **Label** (`400`, 10.5–12px, mono): the data layer — counts, scores, dates, hashes, status pills, kbd hints, uppercase section labels (`letter-spacing .06em`).

### Named Rules
**The Data Speaks Mono Rule.** Any scalar that is machine-produced — a similarity score, a photo count, a duration, a pixel hash, an uptime — is set in monospace, lowercase, tabular-friendly. If it isn't data, it's proportional. This is the single most identifying habit of the system.

## Layout

- Desktop shell is a **224px rail** + fluid main column (`grid-template-columns: 224px minmax(0,1fr)`). The rail is sticky and full-height with nav, a rail-spacer, and the owner chip pinned at bottom.
- Content column: sticky topbar (frosted `color-mix` + backdrop-blur, 84% page tint) with view title left, search right.
- Photo grid: `repeat(auto-fill, minmax(200px, 1fr))` with 8px gaps — dense but legible; each tile is `aspect-ratio: 4/3`.
- People grid: `minmax(164px, 1fr)`; places `minmax(150px, 1fr)`; albums `minmax(200px, 1fr)`; all 12px gaps.
- Detail/meta panel is fixed-width `320px` beside the image, wrapping under it on narrow widths.
- **Single breakpoint `940px`** flips to mobile: rail becomes a bottom nav (backdrop-blur, safe-area bottom padding), grids go 3-up at 2px gutter (people 2-up, places/albums 2-up), detail becomes a stacked sheet with a sticky bar, statusbar and date rail are hidden, toast floats at `bottom: 84px`.
- Spacing rhythm: 4/8/12/16/24px, `main` padded `18px 24px 64px` desktop, `12px 10px 84px` mobile.

## Elevation & Depth

A hybrid: **tonal layering** is the primary depth cue; one soft elevation shadow exists for raised chrome. There are no colored or exotic shadows.

### Shadow Vocabulary
- **Raised** (`0 1px 2px rgba(0,0,0,.35), 0 16px 34px -18px rgba(0,0,0,.6)` dark; 8%/25% black light): cards-over-content, meta panel, mpitems.
- **Elevated** (`-lg`: second stop, up to `80px -30px`): modals, login box, lightbox image, toasts, popover. Surfaces that must read as "drawn above everything."
- Frosted chrome (topbar, bottom nav, statusbar, detail scrim) uses `color-mix(...-transparent)` + `backdrop-filter: blur(14px)` instead of an opaque panel — depth through translucency over the scrolling photo grid.

## Shapes

- Radius vocabulary: `6px` (xs) controls/inputs small, `8px` (sm) cards/buttons/inputs/chips, `12px` (md) person cards/place cards/map, `16px` modals, `999px` pills/tags/badges/dots.
- Photos are always rectangular — `4/3` cards, square faces in people grids and avatars (circular).
- Avatar/logomark: circular / small-radius with a green diagonal gradient (`accent-strong`→`accent`) and a hairline white inner ring.
- Modals: `radius 16px`, `max-width 520px` (760 wide), frosted full-screen scrim.
- Dashed borders denote "incomplete" states (unassigned facebox, dropzone).

## Components

### Buttons
- **Shape:** `8px` radius, 1px transparent border reserved for accent states.
- **Primary:** `--accent` fill, `--on-accent` text, 620 weight; hover `filter: brightness(1.06)`, active `.96`. Hover is a luminance lift, never a color change.
- **Secondary:** `panel2` fill, text color, hairline border; hover promotes to `panel` + lighter border.
- **Danger:** `--bad` fill, white text.
- **Small** (`4px 9px`, `12px`), **Icon button** (square, centered glyph), **Link button** (text-only, `accent-strong`, underline on hover).

### Chips / Tags / Pills
- **Info pill:** mono `10.5px`, panel fill, hairline border, `999px`. `.ok`/`.bad` variants inherit `--ok`/`--bad` text + `color-mix` border at 45–50% opacity.
- **Tag chip (action):** panel fill, muted text; hover warms to `--text`, focus warms border. **Local tag:** green border + green text — the "this is mine" marker.
- **Face/score badge on photos:** dark frosted chip (`rgba(8,9,11,.6)`, backdrop-blur 4px), white text, mono; face badge turns green on hover.

### Cards / Containers
- **Photo card:** `--bg2` fill, `8px` radius desktop / `4px` mobile, `aspect-ratio: 4/3`, overflow hidden. Hover lifts saturation +1.05 on the image. Favorited = 2px green inset ring; archived = 50–55% opacity.
- **Person / Place card:** `--panel` fill + hairline border + `12px` radius + `translateY(-1px)` hover. Circular 84px portrait.
- **Album:** 4:3 imagery with a bottom gradient caption scrim and white title.

### Inputs & Search
- **Text/password/select:** `--panel` (dark) / white (light) fill, hairline border, `8px` radius, `13px` text; focus swaps border to `--accent` with no ring. Placeholder at faint.
- **Checkbox/range:** `accent-color: var(--accent)` — native control tinted green.
- **Search query box:** flex-capped `340px` right-aligned in the topbar; mobile full-width row 3.

### Navigation
- **Rail link:** muted text, `8px` radius, 11px icon gap; hover brightens text; **active** = accent-soft fill + accent-strong text, 640 weight.
- **Mobile nav:** equal-width column links with 21px glyphs, `9px` label, active same green pair. Bottom-mounted, frosted, `overflow-x: auto`, hairline top border.

### Modals & Overlays
- Scrims are always `color-mix(in srgb, var(--bg) 80%, rgba(6,7,9,.7))` + `backdrop-filter: blur(12px)`.
- **Login/Modal:** centered `max-width 520px` panel, `16px` radius, `-lg` shadow, body scroll within `86vh`.
- **Detail (photo sheet):** full-viewport scrim `z-index 100`, image (max-height 72vh) + sticky 320px meta panel. Mobile: stacked column, image `52vh`, sticky top bar.
- **Lightbox:** near-black (`rgba(6,7,9,.92)`) full-viewport, image `max-height 92vh` with `-lg` shadow, circular ghost close button top-right.

### Status & Feedback
- **Toast:** centered bottom pill (`bottom: 46px`, mobile `84px`), panel fill, hairline border, `-lg` shadow.
- **Inline status dots:** 7px circles, `.ok` = `--ok`, `.bad` = `--bad`, ambient hairline inset ring.
- **Loader:** full-viewport `color-mix` scrim + 22px spinner (green top arc over border2 track, `.7s` linear).
- **Popover (autocomplete/suggest):** anchored panel, `-lg` shadow, `8px` items; hover/`hl` = accent-soft fill + accent-strong text with muted mono meta.

### Map (Leaflet)
Fully re-skinned to the system: popups/tips panel-fill with hairline borders, attribution frosted, marker clusters green (`--accent` fill on `color-mix` ring), controls panel-toned.

## Do's and Don'ts

- **Do** let the photograph carry the light; keep surfaces neutral, recessive, one accent.
- **Do** set every scalar in mono; if a human wouldn't have typed it, it's data.
- **Do** use tonal layering and hairline borders first, shadows only for raised chrome, blur only for over-photo chrome.
- **Do** extend the token pair (`--bg` … `--bad`, plus radii/spacing) when adding a surface; never hardcode a new hex.
- **Don't** introduce a second accent or a colorful status mode; `--ok`/`--bad` (green/red) are the only semantic colors beyond the accent.
- **Don't** add decorative gradients, animations, or glass-chrome effects; motion is state response only.
- **Don't** break the light theme: both tokensets must stay equal citizens (F-01-compatible, contrast-checked).
- **Don't** hardcode dark values in components that must work in light mode — reference the variables.
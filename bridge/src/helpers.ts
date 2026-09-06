/**
 * Pure, dependency-free helpers extracted from bridge.ts so they can be
 * unit-tested with `bun test` without pulling in the Proton Drive SDK.
 *
 * This module must NOT import anything (no `./init`, no `@protontech/drive-sdk`,
 * no node/bun builtins) — keep it a plain TypeScript module.
 *
 * @module helpers
 */

/** Glob pattern matching stale work files (e.g. `<uid>-<uuid>.full`). */
export const STALE_WORK_FILE_GLOB = /^[A-Za-z0-9_-]+-[A-Fa-f0-9-]+\.full$/;

/** Tags Proton attaches to photos; map the numeric tag id to a human name. */
export const PHOTO_TAGS = ['Favorites', 'Screenshots', 'Videos', 'LivePhotos', 'MotionPhotos', 'Selfies', 'Portraits', 'Bursts', 'Panoramas', 'Raw'];

/**
 * Minimal structural shape of a Proton Drive SDK PhotoNode — enough for
 * `nodeToJson`. Structurally compatible with the real SDK type at build time.
 */
export interface PhotoNodeLike {
    uid: string;
    name: { value?: string; key: string };
    mediaType: string;
    photo?: {
        captureTime?: Date;
        albums?: Array<{ nodeUid: string }>;
        tags?: Array<number>;
        mainPhotoNodeUid?: string;
        relatedPhotoNodeUids?: string[];
    };
    activeRevision?: {
        claimedDigests?: { sha1?: string };
        claimedSize?: number;
        storageSize?: number;
    };
    creationTime?: Date;
    modificationTime?: Date;
}

/** Map a PhotoNode to the JSON shape the Python indexer consumes. */
export function nodeToJson(node: PhotoNodeLike): Record<string, unknown> {
    return {
        uid: node.uid,
        name: node.name.value ?? node.name.key,
        mediaType: node.mediaType,
        captureTime: node.photo?.captureTime ? node.photo.captureTime.toISOString() : null,
        albums: node.photo?.albums?.map((a) => a.nodeUid) ?? [],
        sha1: node.activeRevision?.claimedDigests?.sha1 ?? null,
        size: node.activeRevision?.claimedSize ?? node.activeRevision?.storageSize ?? null,
        creationTime: node.creationTime ? node.creationTime.toISOString() : null,
        modificationTime: node.modificationTime ? node.modificationTime.toISOString() : null,
        tags: node.photo?.tags?.map((t) => PHOTO_TAGS[t] ?? String(t)) ?? [],
        mainPhotoNodeUid: node.photo?.mainPhotoNodeUid ?? null,
        relatedPhotoNodeUids: node.photo?.relatedPhotoNodeUids ?? [],
    };
}

/** SDK cache files the "clear cache" endpoint unlinks (WAL/SHM siblings too). */
export const CACHE_FILE_GLOB = /^cache-.*\.sqlite(-(shm|wal))?$/i;

/**
 * Sweep stale work files from a directory. Removes any file matching
 * {@link STALE_WORK_FILE_GLOB} that is older than `maxAgeMs` milliseconds.
 * Returns the list of removed filenames.
 *
 * This is a pure function that takes a list of directory entries and a
 * current-time callback so it can be unit-tested without touching the
 * filesystem. The bridge calls it at startup to clean up orphaned temp
 * files left by a previous crash.
 */
export function sweepStaleWorkFiles(
    entries: Array<{ name: string; mtimeMs: number }>,
    nowMs: number,
    maxAgeMs: number,
): string[] {
    const cutoff = nowMs - maxAgeMs;
    const removed: string[] = [];
    for (const entry of entries) {
        if (STALE_WORK_FILE_GLOB.test(entry.name) && entry.mtimeMs < cutoff) {
            removed.push(entry.name);
        }
    }
    return removed;
}

/** Maximum number of uids accepted in a single /nodes or /thumbnails request. */
export const MAX_UID_BATCH = 5000;

/**
 * Validate a Proton photo uid before it is used to build filesystem paths,
 * temp filenames, or response headers. Uids are opaque base64url-ish
 * identifiers; anything else (path separators, `..`, whitespace, control
 * chars, non-strings) is rejected so a request can never escape the work dir
 * or probe arbitrary files.
 */
export function isValidUid(uid: unknown): uid is string {
    return typeof uid === 'string' && /^[A-Za-z0-9_-]{1,128}$/.test(uid);
}

/** A successfully parsed byte range. */
export interface ParsedRange {
    start: number;
    end: number;
    length: number;
}

/** Result of parsing a Range header for a body of the given `size`. */
export interface RangeResult {
    status: number;
    /** Present only when the request is unsatisfiable (416). */
    contentRange?: string;
    /** Present only on a satisfiable 206 range request. */
    range?: ParsedRange;
}

/**
 * Parse a single `Range: bytes=start-end | start- | -suffix` header so the
 * browser can seek. Returns `null` for no header or an unsupported form (the
 * caller then serves the full body with status 200), matching common static
 * server behavior. A `start >= size` request yields `{status: 416}` with the
 * `Content-Range` header value the caller should return (bytes `*` slash size).
 */
export function parseRange(rangeHeader: string | null | undefined, size: number): RangeResult | null {
    if (!rangeHeader) return null;
    const m = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader.trim());
    if (!m || (!m[1] && !m[2])) return null;
    let start = 0;
    let end = size - 1;
    if (m[1]) {
        start = parseInt(m[1], 10);
        if (m[2]) end = parseInt(m[2], 10);
    } else {
        start = Math.max(0, size - parseInt(m[2], 10));
    }
    if (start >= size) {
        return { status: 416, contentRange: `bytes */${size}` };
    }
    end = Math.min(end, size - 1);
    return { status: 206, range: { start, end, length: end - start + 1 } };
}

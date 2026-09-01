/**
 * proton-faces bridge
 *
 * A small HTTP service that wraps the Proton Drive SDK and exposes the three
 * operations the Python indexer needs:
 *
 *   GET  /health                 →  {"ok": true, "loggedIn": bool}
 *   GET  /timeline               →  array of photo nodes (uid, name, captureTime, sha1, mediaType)
 *   POST /thumbnails             →  body {"uids": [...]} → downloads Type1 (512px) thumbnails
 *                                   into DATA_DIR/work/<uid>.webp, returns results
 *   GET  /photo/{uid}/full       →  streams the full-resolution photo (on-demand, read-only)
 *
 * It is compiled as the "entry point" of the Proton Drive CLI repository, so it
 * reuses the CLI's own `init()` machinery (auth, crypto, cache, feature flags).
 */

import { init } from './init';
import type { PhotoNode } from '@protontech/drive-sdk';
import { ThumbnailType } from '@protontech/drive-sdk';
import { mkdir, writeFile } from 'node:fs/promises';
import { openSync, fsyncSync, closeSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { createRateLimiter, extractRetryAfter, type TokenBucket } from './rateLimit';

const PORT = Number(process.env.PORT ?? 8090);
const DATA_DIR = process.env.DATA_DIR ?? '/data';
const FULL_RES_TIMEOUT_MS = Number(process.env.PROTON_BRIDGE_FULL_RES_TIMEOUT_MS ?? 5 * 60_000);

async function ensureLoggedIn(ctx: Awaited<ReturnType<typeof init>>): Promise<Response> {
    const loggedIn = ctx.auth.isLoggedIn();
    if (!loggedIn) {
        return Response.json(
            { ok: false, error: 'Not logged in. Create a session file first (see README).' },
            { status: 401 },
        );
    }
    return Response.json({ ok: true, loggedIn });
}

const PHOTO_TAGS = ['Favorites', 'Screenshots', 'Videos', 'LivePhotos', 'MotionPhotos', 'Selfies', 'Portraits', 'Bursts', 'Panoramas', 'Raw'];

function nodeToJson(node: PhotoNode): Record<string, unknown> {
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

async function fetchTimeline(ctx: Awaited<ReturnType<typeof init>>, limiter: TokenBucket, url?: URL, idsOnly = false): Promise<Response> {
    const limit = url ? Number(url.searchParams.get('limit') ?? 0) : 0;

    // A full library timeline can take many minutes to paginate AND to
    // decrypt node keys. We must stream the result as newline-delimited JSON
    // so bytes keep flowing on the connection the whole time — otherwise Bun's
    // idleTimeout closes it mid-fetch (clients see 'Server disconnected').
    //
    // Lines starting with '#' are progress/keep-alive comments; the client
    // skips them. Every node is one JSON line.
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
            const send = (line: string) => controller.enqueue(encoder.encode(`${line}\n`));
            try {
                const uids: string[] = [];
                let lastPing = Date.now();
                await limiter.acquire();
                for await (const item of ctx.photosSdk.iterateTimeline()) {
                    if (limit > 0 && uids.length >= limit) {
                        break;
                    }
                    if (idsOnly) {
                        send(JSON.stringify({ uid: item.nodeUid, captureTime: item.captureTime.toISOString() }));
                    }
                    uids.push(item.nodeUid);
                    if (Date.now() - lastPing > 15000) {
                        send(`# progress: collected ${uids.length} uids`);
                        lastPing = Date.now();
                    }
                }

                if (!idsOnly) {
                    let count = 0;
                    await limiter.acquire();
                    for await (const node of ctx.photosSdk.iterateNodes(uids)) {
                        if ('missingUid' in node) {
                            send(JSON.stringify({ uid: node.missingUid, missing: true }));
                        } else {
                            send(JSON.stringify(nodeToJson(node as PhotoNode)));
                        }
                        count++;
                        if (Date.now() - lastPing > 15000) {
                            send(`# progress: ${count}/${uids.length} nodes`);
                            lastPing = Date.now();
                        }
                    }
                }
                controller.close();
            } catch (error) {
                const ra = extractRetryAfter(error);
                if (ra !== null) limiter.noteRetryAfter(ra);
                controller.error(error);
            }
        },
    });

    return new Response(stream, { headers: { 'Content-Type': 'application/x-ndjson' } });
}

async function fetchNodes(ctx: Awaited<ReturnType<typeof init>>, limiter: TokenBucket, body: unknown): Promise<Response> {
    const { uids } = (body ?? {}) as { uids?: string[] };
    if (!Array.isArray(uids) || uids.length === 0) {
        return Response.json({ ok: false, error: 'Expected {"uids": [...]}' }, { status: 400 });
    }

    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
            const send = (line: string) => controller.enqueue(encoder.encode(`${line}\n`));
            try {
                let count = 0;
                await limiter.acquire();
                for await (const node of ctx.photosSdk.iterateNodes(uids)) {
                    if ('missingUid' in node) {
                        send(JSON.stringify({ uid: node.missingUid, missing: true }));
                    } else {
                        send(JSON.stringify(nodeToJson(node as PhotoNode)));
                    }
                    count++;
                }
                controller.close();
            } catch (error) {
                controller.error(error);
            }
        },
    });

    return new Response(stream, { headers: { 'Content-Type': 'application/x-ndjson' } });
}

async function fetchAlbums(ctx: Awaited<ReturnType<typeof init>>, limiter: TokenBucket): Promise<Response> {
    const albums: { uid: string; name: string }[] = [];
    await limiter.acquire();
    for await (const node of ctx.photosSdk.iterateAlbums()) {
        if ('missingUid' in node) continue;
        const a = node as PhotoNode;
        albums.push({ uid: a.uid, name: a.name.value ?? a.name.key });
    }
    return Response.json({ ok: true, albums });
}

async function fetchThumbnails(ctx: Awaited<ReturnType<typeof init>>, limiter: TokenBucket, body: unknown): Promise<Response> {
    const { uids } = (body ?? {}) as { uids?: string[] };
    if (!Array.isArray(uids) || uids.length === 0) {
        return Response.json({ ok: false, error: 'Expected {"uids": [...]}' }, { status: 400 });
    }

    const workDir = path.join(DATA_DIR, 'work');
    await mkdir(workDir, { recursive: true });

    const results: { uid: string; ok: boolean; error?: string }[] = [];
    const pending: string[] = [];
    for (const uid of uids) {
        const dest = path.join(workDir, `${uid}.webp`);
        // Skip if we already have this thumbnail (resumability).
        if (await Bun.file(dest).exists()) {
            results.push({ uid, ok: true });
        } else {
            pending.push(uid);
        }
    }

    await limiter.acquire();
    for await (const result of ctx.photosSdk.iterateThumbnails(pending, ThumbnailType.Type1)) {
        if (result.ok) {
            const dest = path.join(workDir, `${result.nodeUid}.webp`);
            await writeFile(dest, result.thumbnail);
            results.push({ uid: result.nodeUid, ok: true });
        } else {
            results.push({ uid: result.nodeUid, ok: false, error: result.error });
        }
    }

    return Response.json({ ok: true, results });
}

async function streamFullPhoto(ctx: Awaited<ReturnType<typeof init>>, limiter: TokenBucket, url: URL, request: Request): Promise<Response> {
    const uid = url.pathname.split('/')[2];
    if (!uid) {
        return Response.json({ ok: false, error: 'Missing photo uid' }, { status: 400 });
    }

    // Resolve the real MIME type from the node (preferred) so the browser
    // picks the right codec instead of guessing from .webp/.jpg extensions.
    let mediaType: string | null = null;
    try {
        await limiter.acquire();
        for await (const node of ctx.photosSdk.iterateNodes([uid])) {
            if (!('missingUid' in node)) {
                mediaType = (node as PhotoNode).mediaType ?? null;
            }
            break;
        }
    } catch {
        // fall through to a safe default; the client will still get bytes.
    }

    const isVideo = mediaType?.startsWith('video/') ?? false;
    const contentType = mediaType ?? (isVideo ? 'application/octet-stream' : 'image/jpeg');

    const downloader = await ctx.photosSdk.getFileDownloader(uid, AbortSignal.timeout(FULL_RES_TIMEOUT_MS));

    if (!isVideo) {
        // Images stream live straight from Proton — no temp file on disk. This
        // avoids the clobber/partial-write races on a shared temp path that
        // broke concurrent opens (and is the regression this replaces) and adds
        // only download latency, no disk round-trip. <img> doesn't need Range.
        //
        // NOTE: we deliberately do NOT send Content-Length — getClaimedSizeInBytes()
        // can differ from the actually-decrypted byte count, and a mismatched
        // Content-Length makes clients think the stream was truncated. Chunked
        // transfer avoids that.
        const stream = new ReadableStream<Uint8Array>({
            async start(controller) {
                const writable = new WritableStream<Uint8Array>({
                    write(chunk) {
                        controller.enqueue(chunk);
                    },
                    close() {
                        controller.close();
                    },
                    abort(err) {
                        controller.error(err);
                    },
                });

                try {
                    const dlController = downloader.downloadToStream(writable);
                    await dlController.completion();
                    controller.close();
                } catch (err) {
                    controller.error(err);
                }
            },
        });

        return new Response(stream, {
            headers: {
                'Content-Type': contentType,
                'Cache-Control': 'no-store',
                'X-Photo-Uid': uid,
            },
        });
    }

    // Videos need HTTP Range for seeking (HTML5 <video> requires it — without
    // Range support the browser must download the whole file before play). We
    // buffer the decrypted bytes to a per-request temp file and stream it back,
    // honoring `Range` so the player only pulls the bytes it needs.
    //
    // The path is unique per request (randomUUID) so concurrent opens of the same
    // uid never clobber each other. We fsync + size-check before serving so a
    // partially-written file (client abort) is never handed out, and we unlink
    // the temp file once the response body finishes (or the client disconnects).
    const workDir = path.join(DATA_DIR, 'work');
    await mkdir(workDir, { recursive: true });
    const tmp = path.join(workDir, `${uid}-${randomUUID()}.full`);

    try {
        await downloader.downloadToPath(tmp);

        const ff = openSync(tmp, 'r');
        try {
            fsyncSync(ff);
        } finally {
            closeSync(ff);
        }

        const file = Bun.file(tmp);
        const size = (await file.stat()).size;
        if (size === 0) {
            throw new Error('downloaded video is empty');
        }

        // Parse a single `Range: bytes=start-end | start- | -suffix` header so
        // the browser can seek. Unsupported forms fall back to serving the full
        // body (200), matching common static-server behavior.
        let start = 0;
        let end = size - 1;
        let status = 200;
        const rangeHeader = request.headers.get('range');
        if (rangeHeader) {
            const m = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader.trim());
            if (m && (m[1] || m[2])) {
                if (m[1]) {
                    start = parseInt(m[1], 10);
                    if (m[2]) end = parseInt(m[2], 10);
                } else {
                    start = Math.max(0, size - parseInt(m[2], 10));
                }
                if (start >= size) {
                    await file.unlink();
                    return new Response(null, {
                        status: 416,
                        headers: { 'Content-Range': `bytes */${size}` },
                    });
                }
                end = Math.min(end, size - 1);
                status = 206;
            }
        }

        const length = end - start + 1;
        const headers: Record<string, string> = {
            'Cache-Control': 'no-store',
            'X-Photo-Uid': uid,
            'Content-Type': contentType,
            'Content-Length': String(length),
            'Accept-Ranges': 'bytes',
        };
        if (status === 206) headers['Content-Range'] = `bytes ${start}-${end}/${size}`;

        if (request.method === 'HEAD') {
            await file.unlink();
            return new Response(null, { status, headers });
        }

        const body = new ReadableStream<Uint8Array>({
            async start(controller) {
                const reader = file.slice(start, end + 1).stream().getReader();
                try {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        controller.enqueue(value);
                    }
                    controller.close();
                } catch (err) {
                    controller.error(err);
                } finally {
                    await file.unlink().catch(() => {});
                }
            },
        });

        return new Response(body, { status, headers });
    } catch (err) {
        await Bun.file(tmp).unlink().catch(() => {});
        return Response.json({ ok: false, error: String(err) }, { status: 502 });
    }
}

async function main(): Promise<void> {
    const ctx = await init({
        clientUidPrefix: 'sdk-js-cli',
        appVersion: 'cli-drive@0.8.0',
        sdkVersion: 'js@0.21.0',
        enablePersistedEvents: false,
        enableConsoleLog: true,
        enableMetrics: false,
        flags: {
            DriveCryptoEncryptBlocksWithPgpAead: true,
            DriveSmallFileUpload: true,
        },
    });

    console.log(`[bridge] init complete; session logged in: ${ctx.auth.isLoggedIn()}`);

    const limiter = createRateLimiter();

    Bun.serve({
        port: PORT,
        // /timeline of a large library takes a while to paginate; /photo/*/full streams.
        idleTimeout: 255,
        async fetch(request: Request) {
            const url = new URL(request.url);
            try {
                if (url.pathname === '/health') {
                    return await ensureLoggedIn(ctx);
                }
                if (url.pathname === '/timeline') {
                    return await fetchTimeline(ctx, limiter, url, false);
                }
                if (url.pathname === '/timeline/ids') {
                    return await fetchTimeline(ctx, limiter, url, true);
                }
                if (url.pathname === '/nodes' && request.method === 'POST') {
                    return await fetchNodes(ctx, limiter, await request.json());
                }
                if (url.pathname === '/albums') {
                    return await fetchAlbums(ctx, limiter);
                }
                if (url.pathname === '/thumbnails' && request.method === 'POST') {
                    return await fetchThumbnails(ctx, limiter, await request.json());
                }
                if (url.pathname.startsWith('/photo/') && url.pathname.endsWith('/full')) {
                    return await streamFullPhoto(ctx, limiter, url, request);
                }
                return Response.json({ ok: false, error: 'Not found' }, { status: 404 });
            } catch (error) {
                console.error('[bridge] error:', error);
                const ra = extractRetryAfter(error);
                if (ra !== null) limiter.noteRetryAfter(ra);
                return Response.json(
                    { ok: false, error: error instanceof Error ? error.message : String(error) },
                    { status: 500 },
                );
            }
        },
    });

    console.log(`[bridge] listening on :${PORT}`);
}

main().catch((error) => {
    console.error('[bridge] fatal:', error);
    process.exit(1);
});
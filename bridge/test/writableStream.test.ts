import { describe, expect, test } from 'bun:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

/**
 * Regression test for the video download path (bridge/src/bridge.ts): we wrap a
 * Bun FileSink in a real `WritableStream` and feed it to the SDK's
 * `downloadToStream()`. The SDK calls `stream.getWriter()`, `writer.write(chunk)`
 * repeatedly, then `writer.releaseLock()` on success — it never calls
 * `writer.close()`. This exercises that exact sequence against a real stream
 * and asserts the wrapped FileSink receives every chunk and is finalized via
 * `sink.end()`.
 *
 * This does not import the SDK — it reproduces the SDK's calling pattern so the
 * bridge wrapper is validated without the SDK installed.
 */

// A FileSink-backed WritableStream identical to the one in bridge.ts.
function fileSinkWritable(sink: Bun.FileSink): WritableStream<Uint8Array> {
    return new WritableStream<Uint8Array>({
        write(chunk) {
            sink.write(chunk);
        },
        close() {
            sink.end();
        },
        abort(err) {
            sink.end(err instanceof Error ? err : new Error(String(err)));
        },
    });
}

describe('FileSink-backed WritableStream (SDK downloadToStream pattern)', () => {
    test('getWriter -> write -> releaseLock -> close writes every chunk to disk', async () => {
        const dir = await mkdtemp(join(tmpdir(), 'pf-ws-'));
        const tmp = join(dir, 'out.bin');
        try {
            const sink = Bun.file(tmp).writer();
            const writable = fileSinkWritable(sink);

            const writer = writable.getWriter();
            // The SDK checks `'releaseLock' in writer` — a real writer must have it.
            expect('releaseLock' in writer).toBe(true);

            const chunks: Uint8Array[] = [
                new Uint8Array([1, 2, 3]),
                new Uint8Array([4, 5]),
                new Uint8Array([6, 7, 8, 9]),
            ];
            for (const chunk of chunks) {
                await writer.write(chunk);
            }

            // SDK success path: releaseLock() without close() must not throw.
            expect(() => writer.releaseLock()).not.toThrow();

            // Bridge finalizes the sink via close() after completion().
            await writable.close();

            const bytes = new Uint8Array(await Bun.file(tmp).arrayBuffer());
            expect(bytes).toEqual(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });

    test('releaseLock after writes leaves the stream closeable', async () => {
        const dir = await mkdtemp(join(tmpdir(), 'pf-ws-'));
        const tmp = join(dir, 'out.bin');
        try {
            const sink = Bun.file(tmp).writer();
            const writable = fileSinkWritable(sink);

            const writer = writable.getWriter();
            await writer.write(new Uint8Array([9, 8, 7]));
            await writer.releaseLock();

            await expect(writable.close()).resolves.toBeUndefined();
            const bytes = new Uint8Array(await Bun.file(tmp).arrayBuffer());
            expect(bytes).toEqual(new Uint8Array([9, 8, 7]));
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });

    test('abort() and close() throw while the writer is locked (why the bridge catch uses sink.end)', async () => {
        const dir = await mkdtemp(join(tmpdir(), 'pf-ws-'));
        const tmp = join(dir, 'out.bin');
        try {
            const sink = Bun.file(tmp).writer();
            const writable = fileSinkWritable(sink);
            const writer = writable.getWriter();
            await writer.write(new Uint8Array([1, 2]));

            // The SDK throws on failure without releasing the lock; the bridge
            // therefore must not call writable.abort()/close() here. Assert the
            // stream rejects them while locked, which is why the bridge's catch
            // path calls sink.end(err) directly instead.
            await expect(writable.abort(new Error('boom'))).rejects.toThrow();
            await expect(writable.close()).rejects.toThrow();
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });
});

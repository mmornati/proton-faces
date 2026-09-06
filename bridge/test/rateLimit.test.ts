import { describe, expect, test } from 'bun:test';
import { TokenBucket, createRateLimiter, extractRetryAfter } from '../src/rateLimit';

describe('TokenBucket', () => {
    test('rate <= 0 disables the bucket', () => {
        const b = new TokenBucket(0);
        expect(b.isEnabled()).toBe(false);
        const b2 = new TokenBucket(-5);
        expect(b2.isEnabled()).toBe(false);
    });

    test('positive rate enables the bucket with a default burst', () => {
        const b = new TokenBucket(5);
        expect(b.isEnabled()).toBe(true);
    });

    test('acquire returns immediately while tokens remain', async () => {
        const b = new TokenBucket(100, 10);
        for (let i = 0; i < 5; i++) {
            await b.acquire();
        }
    });

    test('acquire waits for refill once tokens are exhausted', async () => {
        const b = new TokenBucket(20, 1); // 1 token, refills 1 per 50ms
        const t0 = Date.now();
        await b.acquire();
        await b.acquire();
        await b.acquire();
        const elapsed = Date.now() - t0;
        // two refills after the initial token => ~100ms of waiting
        expect(elapsed).toBeGreaterThanOrEqual(80);
    });

    test('noteRetryAfter pushes resumeAt into the future', async () => {
        const b = new TokenBucket(10, 1);
        b.noteRetryAfter(2);
        const t0 = Date.now();
        await b.acquire();
        const elapsed = Date.now() - t0;
        expect(elapsed).toBeGreaterThanOrEqual(1500);
    });

    test('noteRetryAfter ignores a zero/short retry when the min interval dominates', () => {
        const b = new TokenBucket(10, 1); // min backoff = 100ms
        b.noteRetryAfter(0);
        b.noteRetryAfter(0.05);
    });
});

describe('createRateLimiter', () => {
    test('defaults to disabled', () => {
        const prev = process.env.PROTON_BRIDGE_RATE_LIMIT;
        delete process.env.PROTON_BRIDGE_RATE_LIMIT;
        try {
            expect(createRateLimiter().isEnabled()).toBe(false);
        } finally {
            if (prev !== undefined) process.env.PROTON_BRIDGE_RATE_LIMIT = prev;
        }
    });

    test('enables from env', () => {
        const prev = process.env.PROTON_BRIDGE_RATE_LIMIT;
        process.env.PROTON_BRIDGE_RATE_LIMIT = '3';
        try {
            expect(createRateLimiter().isEnabled()).toBe(true);
        } finally {
            if (prev !== undefined) process.env.PROTON_BRIDGE_RATE_LIMIT = prev;
        }
    });
});

function makeResponse(status: number, retryAfter?: string): Response {
    const headers = new Headers();
    if (retryAfter !== undefined) headers.set('retry-after', retryAfter);
    return new Response(null, { status, headers });
}

describe('extractRetryAfter', () => {
    test('returns null for non-objects', () => {
        expect(extractRetryAfter(null)).toBeNull();
        expect(extractRetryAfter(undefined)).toBeNull();
        expect(extractRetryAfter('string')).toBeNull();
    });

    test('returns null when there is no response', () => {
        expect(extractRetryAfter({ status: 429 })).toBeNull();
    });

    test('ignores non-429/503 statuses', () => {
        expect(extractRetryAfter({ response: makeResponse(500) })).toBeNull();
        expect(extractRetryAfter({ response: makeResponse(200) })).toBeNull();
    });

    test('extracts a numeric retry-after', () => {
        expect(extractRetryAfter({ response: makeResponse(429, '5') })).toBe(5);
        expect(extractRetryAfter({ response: makeResponse(503, '12') })).toBe(12);
    });

    test('defaults to 1 when the header is missing', () => {
        expect(extractRetryAfter({ response: makeResponse(429) })).toBe(1);
    });

    test('defaults to 1 for an unparseable header', () => {
        expect(extractRetryAfter({ response: makeResponse(429, 'not-a-date') })).toBe(1);
    });

    test('parses an HTTP-date retry-after in the future', () => {
        const future = new Date(Date.now() + 30_000).toUTCString();
        const got = extractRetryAfter({ response: makeResponse(429, future) });
        expect(got).not.toBeNull();
        expect(got!).toBeGreaterThan(20);
    });

    test('falls back to cause.response', () => {
        expect(extractRetryAfter({ cause: { response: makeResponse(429, '7') } })).toBe(7);
    });
});

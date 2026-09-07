import { describe, expect, test } from 'bun:test';

/**
 * Unit tests for the bridge auth logic.
 *
 * These test the constant-time comparison and the isAuthorized logic
 * by reimplementing the same functions here (they're not exported from
 * bridge.ts). The integration test (401 without token, 200 with token)
 * is covered by the Python-side tests that exercise the full bridge.
 */

function timingSafeEqual(a: string, b: string): boolean {
    if (a.length !== b.length) return false;
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        result |= a.charCodeAt(i) ^ b.charCodeAt(i);
    }
    return result === 0;
}

describe('timingSafeEqual', () => {
    test('equal strings return true', () => {
        expect(timingSafeEqual('abc', 'abc')).toBe(true);
        expect(timingSafeEqual('', '')).toBe(true);
        expect(timingSafeEqual('token-123', 'token-123')).toBe(true);
    });

    test('different strings return false', () => {
        expect(timingSafeEqual('abc', 'xyz')).toBe(false);
        expect(timingSafeEqual('token-123', 'token-124')).toBe(false);
    });

    test('different lengths return false', () => {
        expect(timingSafeEqual('abc', 'abcd')).toBe(false);
        expect(timingSafeEqual('abcd', 'abc')).toBe(false);
    });

    test('empty vs non-empty returns false', () => {
        expect(timingSafeEqual('', 'a')).toBe(false);
        expect(timingSafeEqual('a', '')).toBe(false);
    });
});

describe('isAuthorized logic', () => {
    function isAuthorized(token: string, authHeader: string | null, pathname: string): boolean {
        // /health is always open
        if (pathname === '/health') return true;
        // No token configured = auth disabled
        if (!token) return true;
        const header = authHeader ?? '';
        return timingSafeEqual(header, `Bearer ${token}`);
    }

    test('/health is always open regardless of token', () => {
        expect(isAuthorized('secret', null, '/health')).toBe(true);
        expect(isAuthorized('secret', 'Bearer wrong', '/health')).toBe(true);
        expect(isAuthorized('', null, '/health')).toBe(true);
    });

    test('no token configured = auth disabled', () => {
        expect(isAuthorized('', null, '/timeline')).toBe(true);
        expect(isAuthorized('', 'anything', '/timeline')).toBe(true);
    });

    test('correct bearer token is accepted', () => {
        expect(isAuthorized('my-token', 'Bearer my-token', '/timeline')).toBe(true);
        expect(isAuthorized('my-token', 'Bearer my-token', '/photo/uid/full')).toBe(true);
        expect(isAuthorized('my-token', 'Bearer my-token', '/cache/clear')).toBe(true);
    });

    test('wrong bearer token is rejected', () => {
        expect(isAuthorized('my-token', 'Bearer wrong-token', '/timeline')).toBe(false);
        expect(isAuthorized('my-token', 'Bearer ', '/timeline')).toBe(false);
    });

    test('missing authorization header is rejected when token is set', () => {
        expect(isAuthorized('my-token', null, '/timeline')).toBe(false);
    });

    test('malformed header is rejected', () => {
        expect(isAuthorized('my-token', 'Basic dXNlcjpwYXNz', '/timeline')).toBe(false);
        expect(isAuthorized('my-token', '', '/timeline')).toBe(false);
    });
});
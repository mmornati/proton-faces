import { describe, expect, test } from 'bun:test';
import { CACHE_FILE_GLOB, isValidUid, MAX_UID_BATCH, nodeToJson, parseRange, STALE_WORK_FILE_GLOB, sweepStaleWorkFiles, type PhotoNodeLike } from '../src/helpers';

function makeNode(overrides: Partial<PhotoNodeLike> = {}): PhotoNodeLike {
    return {
        uid: 'n1',
        name: { value: 'photo.jpg', key: 'photo-key' },
        mediaType: 'image/jpeg',
        photo: {
            captureTime: new Date('2024-01-15T10:30:00Z'),
            albums: [{ nodeUid: 'al1' }, { nodeUid: 'al2' }],
            tags: [0, 3],
            mainPhotoNodeUid: 'n1',
            relatedPhotoNodeUids: ['n2'],
        },
        activeRevision: {
            claimedDigests: { sha1: 'abc123' },
            claimedSize: 2048,
        },
        creationTime: new Date('2024-01-15T10:31:00Z'),
        modificationTime: new Date('2024-01-15T10:32:00Z'),
        ...overrides,
    };
}

describe('nodeToJson', () => {
    test('maps a fully populated node', () => {
        const j = nodeToJson(makeNode());
        expect(j.uid).toBe('n1');
        expect(j.name).toBe('photo.jpg');
        expect(j.mediaType).toBe('image/jpeg');
        expect(j.captureTime).toBe('2024-01-15T10:30:00.000Z');
        expect(j.albums).toEqual(['al1', 'al2']);
        expect(j.sha1).toBe('abc123');
        expect(j.size).toBe(2048);
        expect(j.tags).toEqual(['Favorites', 'LivePhotos']);
        expect(j.mainPhotoNodeUid).toBe('n1');
        expect(j.relatedPhotoNodeUids).toEqual(['n2']);
    });

    test('falls back to name.key when value is missing', () => {
        const j = nodeToJson(makeNode({ name: { key: 'fallback-key' } }));
        expect(j.name).toBe('fallback-key');
    });

    test('handles a node with no photo (captureTime null, empty collections)', () => {
        const j = nodeToJson(makeNode({ photo: undefined, activeRevision: undefined }));
        expect(j.captureTime).toBeNull();
        expect(j.albums).toEqual([]);
        expect(j.sha1).toBeNull();
        expect(j.size).toBeNull();
        expect(j.tags).toEqual([]);
        expect(j.relatedPhotoNodeUids).toEqual([]);
    });

    test('maps unknown tag ids to their string form', () => {
        const j = nodeToJson(makeNode({ photo: { ...makeNode().photo, tags: [99] } }));
        expect(j.tags).toEqual(['99']);
    });

    test('size falls back to storageSize', () => {
        const j = nodeToJson(makeNode({ activeRevision: { storageSize: 512 } }));
        expect(j.size).toBe(512);
    });

    test('nulls capture/creation/modification when absent', () => {
        const j = nodeToJson(makeNode({ photo: { tags: [] }, creationTime: undefined, modificationTime: undefined }));
        expect(j.captureTime).toBeNull();
        expect(j.creationTime).toBeNull();
        expect(j.modificationTime).toBeNull();
    });
});

describe('isValidUid', () => {
    const valid = [
        'abc123',
        'ABC_123-def',
        'a'.repeat(128),
        '0',
        'photo-uid_1',
        'PNR_abc==~def==',
        'uid_with=padding==',
        'a~b',
    ];
    for (const uid of valid) {
        test(`accepts ${JSON.stringify(uid.length > 20 ? uid.slice(0, 20) + '…' : uid)}`, () => {
            expect(isValidUid(uid)).toBe(true);
        });
    }

    const invalid: Array<[unknown, string]> = [
        ['../../etc/passwd', 'path traversal'],
        ['..', 'dotdot'],
        ['a/b', 'forward slash'],
        ['a\\b', 'backslash'],
        ['a b', 'whitespace'],
        ['a\nb', 'newline'],
        ['a\u0000b', 'nul byte'],
        ['', 'empty string'],
        ['a'.repeat(129), 'too long'],
        [42, 'non-string number'],
        [null, 'null'],
        [undefined, 'undefined'],
        [['abc'], 'array'],
        [{ uid: 'abc' }, 'object'],
    ];
    for (const [uid, label] of invalid) {
        test(`rejects ${label}`, () => {
            expect(isValidUid(uid)).toBe(false);
        });
    }
});

describe('MAX_UID_BATCH', () => {
    test('is a positive bounded cap', () => {
        expect(MAX_UID_BATCH).toBeGreaterThan(0);
        expect(MAX_UID_BATCH).toBeLessThanOrEqual(5000);
    });
});

describe('CACHE_FILE_GLOB', () => {
    const cases: Array<[string, boolean]> = [
        ['cache-abc.sqlite', true],
        ['cache-abc.sqlite-shm', true],
        ['cache-abc.sqlite-wal', true],
        ['cache-123.sqlite-shm', true],
        ['CACHE-X.SQLITE', true],
        ['auth-session.json', false],
        ['not-a-cache.txt', false],
        ['cache-abc.db', false],
        ['cache.sqlite', false],
    ];
    for (const [name, expected] of cases) {
        test(`${name} => ${expected}`, () => {
            expect(CACHE_FILE_GLOB.test(name)).toBe(expected);
        });
    }
});

describe('STALE_WORK_FILE_GLOB', () => {
    const cases: Array<[string, boolean]> = [
        ['abc123-def456.full', true],
        ['photo_uid-550e8400-e29b-41d4-a716-446655440000.full', true],
        ['a-b.full', true],
        ['0-1.full', true],
        ['abc123.full', false], // missing uuid part
        ['abc123-def456.tmp', false], // wrong extension
        ['abc123-def456.full.bak', false], // extra suffix
        ['../etc.full', false], // path chars
        ['', false],
    ];
    for (const [name, expected] of cases) {
        test(`${name} => ${expected}`, () => {
            expect(STALE_WORK_FILE_GLOB.test(name)).toBe(expected);
        });
    }
});

describe('sweepStaleWorkFiles', () => {
    const now = 1_000_000_000_000;
    const maxAge = 5 * 60 * 1000; // 5 minutes
    const cutoff = now - maxAge; // 999_700_000_000

    test('removes stale work files, keeps fresh ones', () => {
        const entries = [
            { name: 'abc123-550e8400-e29b-41d4-a716-446655440000.full', mtimeMs: cutoff - 1 }, // stale
            { name: 'def456-660e8400-e29b-41d4-a716-446655440000.full', mtimeMs: now }, // fresh
            { name: 'ghi789-770e8400-e29b-41d4-a716-446655440000.full', mtimeMs: cutoff + 1 }, // fresh (just at edge)
        ];
        const removed = sweepStaleWorkFiles(entries, now, maxAge);
        expect(removed).toEqual(['abc123-550e8400-e29b-41d4-a716-446655440000.full']);
    });

    test('ignores non-work files', () => {
        const entries = [
            { name: 'auth-session.json', mtimeMs: 0 },
            { name: 'cache-abc.sqlite', mtimeMs: 0 },
            { name: 'some-other-file.txt', mtimeMs: 0 },
        ];
        const removed = sweepStaleWorkFiles(entries, now, maxAge);
        expect(removed).toEqual([]);
    });

    test('returns empty for empty input', () => {
        expect(sweepStaleWorkFiles([], now, maxAge)).toEqual([]);
    });

    test('removes all stale work files when all are old', () => {
        const entries = [
            { name: 'a-1111.full', mtimeMs: 0 },
            { name: 'b-2222.full', mtimeMs: 1000 },
            { name: 'c-3333.full', mtimeMs: cutoff - 100 },
        ];
        const removed = sweepStaleWorkFiles(entries, now, maxAge);
        expect(removed).toEqual(['a-1111.full', 'b-2222.full', 'c-3333.full']);
    });

    test('exact cutoff boundary: equal to cutoff is NOT removed', () => {
        const entries = [
            { name: 'a-1111.full', mtimeMs: cutoff }, // exactly at cutoff — not stale
            { name: 'b-2222.full', mtimeMs: cutoff - 1 }, // just below — stale
        ];
        const removed = sweepStaleWorkFiles(entries, now, maxAge);
        expect(removed).toEqual(['b-2222.full']);
    });
});

describe('parseRange', () => {
    test('returns null when there is no header', () => {
        expect(parseRange(null, 100)).toBeNull();
        expect(parseRange(undefined, 100)).toBeNull();
        expect(parseRange('', 100)).toBeNull();
    });

    test('returns null for an unsupported form', () => {
        expect(parseRange('bytes=0-1,2-3', 100)).toBeNull();
        expect(parseRange('items=0-1', 100)).toBeNull();
    });

    test('full open range start- serves to end', () => {
        const r = parseRange('bytes=10-', 100)!;
        expect(r.status).toBe(206);
        expect(r.range).toEqual({ start: 10, end: 99, length: 90 });
    });

    test('closed range', () => {
        const r = parseRange('bytes=10-19', 100)!;
        expect(r.status).toBe(206);
        expect(r.range).toEqual({ start: 10, end: 19, length: 10 });
    });

    test('clamps end to size-1', () => {
        const r = parseRange('bytes=50-200', 100)!;
        expect(r.range).toEqual({ start: 50, end: 99, length: 50 });
    });

    test('suffix range -N', () => {
        const r = parseRange('bytes=-20', 100)!;
        expect(r.range).toEqual({ start: 80, end: 99, length: 20 });
    });

    test('suffix larger than size starts at 0', () => {
        const r = parseRange('bytes=-500', 100)!;
        expect(r.range).toEqual({ start: 0, end: 99, length: 100 });
    });

    test('start at or beyond size yields 416 with content-range', () => {
        const r = parseRange('bytes=100-', 100)!;
        expect(r.status).toBe(416);
        expect(r.contentRange).toBe('bytes */100');
        expect(r.range).toBeUndefined();
    });
});

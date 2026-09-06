import { describe, expect, test } from 'bun:test';
import { CACHE_FILE_GLOB, nodeToJson, parseRange, type PhotoNodeLike } from '../src/helpers';

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

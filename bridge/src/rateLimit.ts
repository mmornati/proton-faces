/**
 * Optional outbound rate limiter for Proton HTTP calls, in two layers.
 *
 * Layer 1 — operation starts (createRateLimiter): one token per bridge
 * operation (timeline sync, node listing, album sync, thumbnail batch,
 * full-res download). Tight default burst (compose: 3 req/s burst 2) keeps
 * the automation inside Proton's comfort zone.
 *
 * Layer 2 — every upstream HTTPS request (createHttpRateLimiter): the SDK
 * fans one operation out into hundreds-to-thousands of paginated/block
 * requests, so gating only operation starts left those unthrottled. Each
 * HTTP call through the patched HTTPClient acquires a token from this bucket.
 *
 * The bridge process is single-instance, so a per-process token bucket is
 * sufficient. Default is OFF (PROTON_BRIDGE_RATE_LIMIT=0 for layer 1, and
 * layer 1 × 10 for layer 2); opt in with a positive number of
 * requests-per-second (sustained), e.g. "5".
 *
 * Why opt-in only: Proton's public API tolerates a comfortable client rate
 * for normal use (a full library sync fits in minutes), and the SDK already
 * retries on 429/5xx. The limiter exists as a circuit-breaker for setups
 * where the bridge sits behind a small home router/CPE whose NAT table
 * gets saturated by many short-lived HTTPS connections, dropping traffic
 * for the entire LAN. If you don't see that problem, leave the default.
 *
 * Honors `Retry-After` from upstream 429 responses: a caller that observes a
 * 429 can call `noteRetryAfter(ms)` to push the bucket's next-available time
 * out by `max(retryAfterMs, 1000/rate)`.
 */

export class TokenBucket {
    private tokens: number;
    private lastRefill: number;
    private ratePerMs: number;
    readonly capacity: number;
    private resumeAt = 0;
    private waiters: { resolve: () => void; reject: (err: Error) => void }[] = [];
    private timer: ReturnType<typeof setTimeout> | null = null;

    constructor(ratePerSec: number, burst?: number) {
        if (ratePerSec <= 0) {
            this.ratePerMs = 0;
            this.capacity = 0;
            this.tokens = 0;
            this.lastRefill = 0;
            return;
        }
        this.ratePerMs = ratePerSec / 1000;
        this.capacity = burst ?? Math.max(1, Math.ceil(ratePerSec * 2));
        this.tokens = this.capacity;
        this.lastRefill = Date.now();
    }

    isEnabled(): boolean {
        return this.ratePerMs > 0;
    }

    private refill(now: number): void {
        if (!this.isEnabled()) return;
        const elapsed = now - this.lastRefill;
        if (elapsed <= 0) return;
        this.tokens = Math.min(this.capacity, this.tokens + elapsed * this.ratePerMs);
        this.lastRefill = now;
    }

    /**
     * Parked waiters beyond this depth are rejected with a 503 instead of
     * queued, so a fan-out storm (the SDK can open hundreds of concurrent
     * upstream calls) can't grow the queue without bound.
     */
    static readonly MAX_PARKED_WAITERS = 100;

    /**
     * Schedule the single refill timer for the earliest moment a token can
     * be granted. No-op when a timer is already pending.
     */
    private scheduleWake(): void {
        if (this.timer !== null) return;
        const now = Date.now();
        this.refill(now);
        const wakeAt = Math.max(now + (1 - this.tokens) / this.ratePerMs, this.resumeAt);
        this.timer = setTimeout(() => this.wakeHead(), Math.max(1, wakeAt - now));
    }

    /**
     * Drop the pending timer and re-schedule it against the current state.
     * Used when noteRetryAfter pushes resumeAt past the pending wake target.
     */
    private rescheduleWake(): void {
        if (this.timer !== null) {
            clearTimeout(this.timer);
            this.timer = null;
        }
        this.scheduleWake();
    }

    private wakeHead(): void {
        this.timer = null;
        const now = Date.now();
        this.refill(now);
        if (this.waiters.length > 0 && now >= this.resumeAt && this.tokens >= 1) {
            // Decrement in the same synchronous tick as the grant decision —
            // no await between refill and decrement, so the token math stays
            // race-free.
            this.tokens -= 1;
            this.waiters.shift()!.resolve();
        }
        if (this.waiters.length > 0) this.scheduleWake();
    }

    async acquire(): Promise<void> {
        if (!this.isEnabled()) return;
        const now = Date.now();
        this.refill(now);
        if (now >= this.resumeAt && this.tokens >= 1) {
            this.tokens -= 1;
            return;
        }
        if (this.waiters.length >= TokenBucket.MAX_PARKED_WAITERS) {
            throw new RateLimitQueueFullError();
        }
        return new Promise<void>((resolve, reject) => {
            this.waiters.push({ resolve, reject });
            this.scheduleWake();
        });
    }

    noteRetryAfter(retryAfterSeconds: number): void {
        const ms = Math.max(0, retryAfterSeconds * 1000);
        // Minimum backoff = time to refill one token = 1000/rate ms
        // (ratePerMs = rate/1000, so this is 1/ratePerMs).
        const minMs = this.isEnabled() ? 1 / this.ratePerMs : 0;
        const backoff = Math.max(ms, minMs);
        this.resumeAt = Math.max(this.resumeAt, Date.now() + backoff);
        if (this.waiters.length > 0) this.rescheduleWake();
    }
}

/**
 * Thrown by acquire() when the parked-waiter queue is full. The bridge's
 * top-level handler translates this into a 503 + Retry-After so the Python
 * client treats it as a transient, retryable condition.
 */
export class RateLimitQueueFullError extends Error {
    constructor() {
        super('rate limiter queue full — too many concurrent requests parked');
        this.name = 'RateLimitQueueFullError';
    }
}

export function createRateLimiter(): TokenBucket {
    const rate = Number(process.env.PROTON_BRIDGE_RATE_LIMIT ?? 0);
    const burst = Number(process.env.PROTON_BRIDGE_RATE_BURST ?? Math.max(1, Math.ceil(rate * 2)));
    const limiter = new TokenBucket(rate, Number.isFinite(burst) && burst > 0 ? burst : undefined);
    if (limiter.isEnabled()) {
        console.log(`[bridge] rate limit enabled: ${rate} req/s sustained, burst ${limiter.capacity}`);
    }
    return limiter;
}

/**
 * HTTP-transport bucket for the patched SDK HTTPClient (patches/httpClient.ts).
 *
 * Applies to every upstream HTTPS call the Proton Drive SDK makes, not just
 * operation starts: paginated listings, thumbnail batches, full-res blocks.
 *
 * `PROTON_BRIDGE_RATE_LIMIT_HTTP` overrides the rate directly; an explicit
 * "0" disables this layer. When the variable is unset or empty, the rate is
 * derived as `PROTON_BRIDGE_RATE_LIMIT × 10` (a single operation start
 * typically fans out into up to ~10 short-lived HTTPS calls), and stays
 * disabled when the operation layer is disabled.
 */
export function createHttpRateLimiter(): TokenBucket {
    const rawExplicit = process.env.PROTON_BRIDGE_RATE_LIMIT_HTTP;
    const hasExplicit = rawExplicit !== undefined && rawExplicit !== '';
    const explicitRate = hasExplicit ? Number(rawExplicit) : Number.NaN;

    let rate: number;
    if (hasExplicit && Number.isFinite(explicitRate)) {
        rate = explicitRate;
    } else if (hasExplicit) {
        // Non-numeric explicit value: treat as disabled rather than silently
        // falling back to a derived rate the operator did not ask for.
        rate = 0;
    } else {
        const opsRate = Number(process.env.PROTON_BRIDGE_RATE_LIMIT ?? 0);
        rate = opsRate > 0 ? opsRate * 10 : 0;
    }

    if (!Number.isFinite(rate) || rate <= 0) {
        return new TokenBucket(0);
    }
    const burst = Math.max(1, Math.ceil(rate));
    const limiter = new TokenBucket(rate, burst);
    console.log(`[bridge] http rate limit enabled: ${rate} req/s sustained, burst ${burst} (per upstream HTTPS call)`);
    return limiter;
}

export function extractRetryAfter(err: unknown): number | null {
    if (!err || typeof err !== 'object') return null;
    const anyErr = err as { response?: Response; status?: number; cause?: { response?: Response; status?: number } };
    const resp = anyErr.response ?? anyErr.cause?.response ?? null;
    if (!resp) return null;
    if (resp.status !== 429 && resp.status !== 503) return null;
    const h = resp.headers?.get?.('retry-after');
    if (!h) return 1;
    const asNum = Number(h);
    if (Number.isFinite(asNum) && asNum >= 0) return asNum;
    const asDate = Date.parse(h);
    if (Number.isFinite(asDate)) return Math.max(1, (asDate - Date.now()) / 1000);
    return 1;
}

/**
 * Feed a `Retry-After` from an upstream error into the bucket, when present.
 *
 * Shared by every SDK-consuming handler so a 429/503 observed on any path
 * (timeline, nodes, albums, thumbnails) pauses the bucket instead of only the
 * ones that remember to call `extractRetryAfter` themselves.
 */
export function noteRetryAfterIfPresent(limiter: TokenBucket, error: unknown): void {
    const ra = extractRetryAfter(error);
    if (ra !== null) limiter.noteRetryAfter(ra);
}
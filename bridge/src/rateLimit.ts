/**
 * Optional outbound rate limiter for Proton HTTP calls.
 *
 * The bridge process is single-instance, so a per-process token bucket is
 * sufficient. Default is OFF (PROTON_BRIDGE_RATE_LIMIT=0); opt in with a
 * positive number of requests-per-second (sustained), e.g. "5".
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
    private capacity: number;
    private resumeAt = 0;

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

    async acquire(): Promise<void> {
        if (!this.isEnabled()) return;
        while (true) {
            const now = Date.now();
            this.refill(now);
            const waitUntil = Math.max(now + (1 - this.tokens) / this.ratePerMs, this.resumeAt);
            if (now >= waitUntil && this.tokens >= 1) {
                this.tokens -= 1;
                return;
            }
            const sleepMs = Math.max(1, waitUntil - now);
            await new Promise((r) => setTimeout(r, sleepMs));
        }
    }

    noteRetryAfter(retryAfterSeconds: number): void {
        const ms = Math.max(0, retryAfterSeconds * 1000);
        const minMs = this.isEnabled() ? 1000 / this.ratePerMs : 0;
        const backoff = Math.max(ms, minMs);
        this.resumeAt = Math.max(this.resumeAt, Date.now() + backoff);
    }
}

export function createRateLimiter(): TokenBucket {
    const rate = Number(process.env.PROTON_BRIDGE_RATE_LIMIT ?? 0);
    const burst = Number(process.env.PROTON_BRIDGE_RATE_BURST ?? Math.max(1, Math.ceil(rate * 2)));
    const limiter = new TokenBucket(rate, Number.isFinite(burst) && burst > 0 ? burst : undefined);
    if (limiter.isEnabled()) {
        console.log(
            `[bridge] rate limit enabled: ${rate} req/s sustained, burst ${limiter.isEnabled() ? Math.ceil(rate * 2) : 0}`,
        );
    }
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
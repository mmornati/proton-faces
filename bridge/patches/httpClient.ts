import { ProtonDriveHTTPClientBlobRequest, ProtonDriveHTTPClientJsonRequest } from '@protontech/drive-sdk';

import { ApiClient } from 'proton-drive-sdk-account';

import { createHttpRateLimiter, extractRetryAfter } from '../rateLimit';

// proton-faces overlay on cli/src/api/httpClient.ts.
//
// The Proton Drive SDK routes EVERY upstream HTTPS request through this one
// injected transport (sdkDependencies.httpClient), yet the bridge's old token
// bucket only throttled operation starts — one (or two) tokens per timeline
// sync / node listing / album sync / thumbnail batch / full-res download. A
// single operation fans out into hundreds-to-thousands of unthrottled
// paginated listings, thumbnail requests and block downloads.
//
// The module-level bucket below paces each of those upstream calls so the
// built-in NAT-table circuit breaker actually covers every connection the
// bridge opens. `PROTON_BRIDGE_RATE_LIMIT_HTTP` (or the derived ops×10
// default) is read once at startup; see bridge/src/rateLimit.ts.
//
// Two details keep the throttle authoritative:
//  - `retry: 0` is passed to ky. ky would otherwise internally retry 429/5xx
//    responses INSIDE authenticatedRequest, bypassing pace() — exactly when
//    pacing matters most. The SDK's own retry loop (DriveAPIService.fetch)
//    lives above this transport and re-acquires the bucket per attempt.
//  - 429/503 responses call noteRetryAfter() (same semantics as the
//    operation-level bucket) so Retry-After pauses this layer too.

const httpLimiter = createHttpRateLimiter();
let lastPacedLogAt = 0;

async function pace(): Promise<void> {
    if (!httpLimiter.isEnabled()) return;
    const t0 = Date.now();
    await httpLimiter.acquire();
    const waitedMs = Date.now() - t0;
    if (waitedMs > 25 && Date.now() - lastPacedLogAt > 10_000) {
        lastPacedLogAt = Date.now();
        console.log(`[bridge] http rate limit: paced upstream Proton request by ${waitedMs} ms`);
    }
}

function noteResponseThrottle(response: Response): void {
    if (!httpLimiter.isEnabled() || (response.status !== 429 && response.status !== 503)) return;
    const retryAfter = extractRetryAfter({ response });
    if (retryAfter !== null) {
        httpLimiter.noteRetryAfter(retryAfter);
    }
}

export class HTTPClient {
    constructor(private readonly apiClient: ApiClient) {}

    async fetchJson(options: ProtonDriveHTTPClientJsonRequest): Promise<Response> {
        await pace();
        const response = await this.apiClient.authenticatedRequest(options.url, {
            method: options.method,
            ...(options.json !== undefined ? { json: options.json } : {}),
            ...(options.body !== undefined && options.json === undefined ? { body: options.body } : {}),
            headers: options.headers,
            timeout: options.timeoutMs,
            signal: options.signal,
            throwHttpErrors: false,
            retry: 0,
        });
        noteResponseThrottle(response);
        return response;
    }

    async fetchBlob(options: ProtonDriveHTTPClientBlobRequest): Promise<Response> {
        await pace();
        const response = await this.apiClient.authenticatedRequest(options.url, {
            method: options.method,
            body: options.body,
            headers: options.headers,
            timeout: options.timeoutMs,
            signal: options.signal,
            throwHttpErrors: false,
            retry: 0,
        });
        noteResponseThrottle(response);
        return response;
    }
}
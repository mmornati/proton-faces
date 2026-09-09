/* proton·faces service worker — offline app shell for the SPA.
 *
 * Cache policy is deliberately small and safe:
 *  - Only the static app shell (/, manifest, icons) is ever cached.
 *  - Every /api/* request and every binary /thumb,/full,/cover,/crop request
 *    goes straight to the network, untouched, so per-user data and signed
 *    URLs are never intercepted or replayed.
 */
const VERSION = "pf-shell-v1";
const SHELL_ASSETS = [
  "./",
  "./manifest.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/apple-touch-icon.png",
];

const BINARY_ENDPOINTS = ["/thumb", "/full", "/cover", "/crop"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(VERSION).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  // Never touch API or binary endpoints — network only, no cache interference.
  if (url.pathname.startsWith("/api/")) return;
  if (BINARY_ENDPOINTS.some((e) => url.pathname.endsWith(e))) return;

  // App shell: stale-while-revalidate so updates land on the next load.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(VERSION).then((cache) => cache.put("./", copy));
          return resp;
        })
        .catch(() => {
          return caches.match("./");
        })
    );
    return;
  }

  // Matching static assets (manifest, icons and anything else in-scope).
  event.respondWith(
    caches.match(request).then((cached) => {
      const freshen = fetch(request)
        .then((resp) => {
          if (resp.ok) {
            const copy = resp.clone();
            caches.open(VERSION).then((cache) => cache.put(request, copy));
          }
          return resp;
        })
        .catch(() => cached);
      return cached || freshen;
    })
  );
});
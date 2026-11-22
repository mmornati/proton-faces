/* proton·faces service worker — offline app shell for the SPA.
 *
 * Cache policy is deliberately small and safe:
 *  - The static app shell (/, manifest, icons) is cached.
 *  - Photo thumbnails (/api/photos/<uid>/thumb) are cached by path, ignoring
 *    the ?sig=&exp= query: signed URLs change every hour, which used to make
 *    the browser re-download every thumbnail it already had. Only a request
 *    that carries an unexpired signature is answered from this cache, only
 *    200 responses enter it, it is capped at THUMB_CACHE_MAX entries and
 *    logout deletes it with every other cache.
 *  - Every other /api/* request and /full,/cover,/crop go straight to the
 *    network, untouched.
 */
const VERSION = "pf-shell-v8";
const THUMB_CACHE = "pf-thumbs-v1";
const THUMB_CACHE_MAX = 4000;
const THUMB_PATH = /^\/api\/photos\/[A-Za-z0-9_=~-]{1,512}\/thumb$/;
const SHELL_ASSETS = [
  "./",
  "./manifest.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/apple-touch-icon.png",
  "./css/tokens.css",
  "./css/app.css",
  "./css/design-system.css",
  "./qrcode.js",
  "./vendor/leaflet/leaflet.js",
  "./vendor/leaflet/leaflet.css",
  "./vendor/markercluster/leaflet.markercluster.js",
  "./vendor/markercluster/MarkerCluster.css",
  "./vendor/markercluster/MarkerCluster.Default.css",
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
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION && k !== THUMB_CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  if (THUMB_PATH.test(url.pathname) && _signedAndUnexpired(url)) {
    event.respondWith(_thumb(request, url));
    return;
  }

  // Never touch other API or binary endpoints — network only.
  if (url.pathname.startsWith("/api/")) return;
  if (BINARY_ENDPOINTS.some((e) => url.pathname.endsWith(e))) return;

  // App shell: stale-while-revalidate so updates land on the next load.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((resp) => {
          // Only a real 200 from this origin may become the offline shell:
          // a 502 from the reverse proxy during a restart would otherwise
          // be served as "the app" on every offline load afterwards.
          // …and only the app's own entry point: any other same-origin
          // navigation (a vendor file opened in a tab) would otherwise
          // replace the offline shell.
          if (resp.ok && resp.type === "basic" && url.pathname === "/") {
            const copy = resp.clone();
            caches.open(VERSION).then((cache) => cache.put("./", copy));
          }
          return resp;
        })
        .catch(() => {
          return caches.match("./");
        })
    );
    return;
  }

  // Static assets (styles, scripts, manifest, icons): network first, cache
  // only as the offline fallback. Serving the cached copy first paired a
  // freshly deployed page with the previous stylesheet until the next load.
  event.respondWith(
    fetch(request)
      .then((resp) => {
        if (resp.ok) {
          const copy = resp.clone();
          caches.open(VERSION).then((cache) => cache.put(request, copy));
        }
        return resp;
      })
      .catch(() => caches.match(request))
  );
});
function _signedAndUnexpired(url) {
  const exp = Number(url.searchParams.get("exp"));
  return Boolean(url.searchParams.get("sig")) && Number.isFinite(exp) && exp * 1000 > Date.now();
}

let _thumbPuts = 0;

async function _thumb(request, url) {
  const key = url.origin + url.pathname;
  let cache;
  try {
    cache = await caches.open(THUMB_CACHE);
    const hit = await cache.match(key);
    if (hit) return hit;
  } catch (e) {
    return fetch(request);
  }
  const resp = await fetch(request);
  if (resp.ok && resp.status === 200 && resp.type === "basic") {
    const copy = resp.clone();
    cache.put(key, copy).then(() => {
      if (++_thumbPuts % 200 === 0) return _trimThumbs(cache);
    }).catch(() => {});
  }
  return resp;
}

async function _trimThumbs(cache) {
  const keys = await cache.keys();
  const extra = keys.length - THUMB_CACHE_MAX;
  // keys() lists entries in insertion order: drop the oldest.
  for (let i = 0; i < extra; i++) await cache.delete(keys[i]);
}

# Vendored front-end libraries

Served from the app's own origin so the Content-Security-Policy needs no CDN
allow-list and a CDN compromise cannot inject script into the SPA.

| Library | Version | License | Source |
|---|---|---|---|
| Leaflet | 1.9.4 | BSD-2-Clause | https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/ (`leaflet.js`, `leaflet.css`, `images/`) |
| Leaflet.markercluster | 1.5.3 | MIT | https://cdn.jsdelivr.net/npm/leaflet.markercluster@1.5.3/dist/ (`leaflet.markercluster.js`, `MarkerCluster.css`, `MarkerCluster.Default.css`) |

Files are byte-for-byte copies of the published `dist/` artifacts. To upgrade,
replace the files, update the version here and in `index.html`, and bump the
service-worker `VERSION` in `sw.js`.

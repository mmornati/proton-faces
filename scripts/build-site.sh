#!/usr/bin/env bash
# Build the GitHub Pages site: hand-built landing + film at the root, MkDocs under /docs/.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf site
mkdocs build --strict --clean --site-dir site/docs
cp -R website/. site/
echo "site/ ready: landing at site/index.html, docs at site/docs/"

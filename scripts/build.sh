#!/usr/bin/env bash
# Build the two images locally (e.g. on the server).
#
# The recommended deploy path is to PULL prebuilt images from GHCR instead:
#   docker compose pull && docker compose up -d
# That does zero building on the server and is the option we recommend.
#
# This script is for when you must build from source. Building is CPU-heavy
# (the bridge compiles a Bun binary, the app installs ML packages). True build
# parallelism is controlled by BuildKit, not by docker build flags; to keep the
# machine responsive during a local build, pin the BuildKit builder to a subset
# of cores once:
#
#   # one-time setup: a builder limited to 2 CPUs
#   docker buildx create --name capped \
#     --config <(printf '[worker.oci]\n  max-parallelism = 2\n  cpu-quota = 200000\n') \
#     --use
#   # then build with
#   docker buildx build --builder capped -t proton-faces/proton-bridge:0.1.0 ./bridge
#
# Usage:
#   scripts/build.sh            # build both images
#   scripts/build.sh app        # build only one image (bridge|app)

set -euo pipefail

TARGET="${1:-}"

build() {
  local name="$1" context="$2"
  echo "==> building ${name} from ${context}"
  echo "    (tip: use 'docker buildx create --name capped ...' to limit CPU, see header)"
  docker build -t "proton-faces/${name}:0.1.0" "${context}"
}

case "$TARGET" in
  bridge) build proton-bridge ./bridge ;;
  app)    build app ./app ;;
  "")     build proton-bridge ./bridge; build app ./app ;;
  *)      echo "unknown target: $TARGET (bridge|app)" >&2; exit 1 ;;
esac
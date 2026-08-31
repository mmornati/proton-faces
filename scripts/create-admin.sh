#!/usr/bin/env bash
# Create the first admin user for proton-faces.
#
# Usage:
#   scripts/create-admin.sh                            # prompts for username
#   scripts/create-admin.sh mom                        # creates user 'mom'
#   scripts/create-admin.sh mom --display-name "Mom"
#   ADMIN_PASSWORD=... scripts/create-admin.sh mom     # non-interactive password
#
# Must be run from the host against a running compose stack so the script
# can `docker compose exec` into the app container (where the index DB lives).
# The password is read from stdin (or ADMIN_PASSWORD); it never lands on disk.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found on PATH" >&2
  exit 1
fi
if ! docker compose ps app >/dev/null 2>&1; then
  echo "no running 'app' service. Start the stack first: docker compose up -d" >&2
  exit 1
fi

DISPLAY_NAME_FLAG=()
PREV=""
for arg in "$@"; do
  case "$PREV" in
    --display-name) DISPLAY_NAME_FLAG=(--display-name "$arg"); PREV=""; continue ;;
  esac
  case "$arg" in
    --display-name) PREV="$arg"; continue ;;
    *) ;;
  esac
done

docker compose exec -T app python main.py --create-admin "$@" "${DISPLAY_NAME_FLAG[@]}"

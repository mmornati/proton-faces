#!/usr/bin/env bash
# Export the Proton Drive CLI session into a plaintext file the bridge can use.
#
# The bridge uses PROTON_DRIVE_CREDENTIALS_STORE=unsafe_file and looks for the
# session at DATA_DIR/auth-session.json (here: ./data/auth-session.json), which
# is the same file the official `proton-drive` CLI reads/writes when launched
# with that credential store. If you already have a session from the CLI
# (e.g. stored in `pass`), copy it into place manually.
#
# Usage:
#   scripts/export-session.sh
#   # or point PROTON_DRIVE_CREDENTIALS_STORE at the CLI's own store.
#
# Requires the proton-drive CLI on PATH and a logged-in session.
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
CRED_STORE="${PROTON_DRIVE_CREDENTIALS_STORE:-unsafe_file}"

mkdir -p "$DATA_DIR"

if command -v proton-drive >/dev/null 2>&1; then
  echo "=> Generating auth session with the proton-drive CLI..."
  if [ "$CRED_STORE" = "unsafe_file" ]; then
    export PROTON_DRIVE_CREDENTIALS_STORE=unsafe_file
    export PROTON_DRIVE_CACHE_DIR="$DATA_DIR"
  fi
  # Any command that forces a refresh of the cached session works. Login
  # happens on first use if the user isn't authenticated yet.
  proton-drive auth login >/dev/null 2>&1 || true
  echo "=> auth session should now be at $DATA_DIR/auth-session.json"
  ls -l "$DATA_DIR/auth-session.json" 2>/dev/null || echo "   (not found — run 'proton-drive auth login' interactively first)"
else
  echo "proton-drive CLI not found on PATH."
  echo "Place a session file at $DATA_DIR/auth-session.json manually."
  echo "Format (JSON): {\"session\":{\"uid\":\"...\",\"accessToken\":\"...\",\"refreshToken\":\"...\"},\"userKeyPassword\":\"...\",\"cachePassword\":\"...\"}"
fi
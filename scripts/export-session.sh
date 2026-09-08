#!/usr/bin/env bash
# Export the Proton Drive CLI session into a plaintext file the bridge can use.
#
# The bridge uses PROTON_DRIVE_CREDENTIALS_STORE=unsafe_file and looks for the
# session at /data/auth-session.json inside its container. That path is
# bind-mounted from the host file produced by this script (default:
# ./credentials/auth-session.json, overridable via AUTH_SESSION_OUT).
#
# The session file MUST live outside the shared `data` volume — see issue #32.
# compose.yml only mounts the session file into the proton-bridge container,
# so the internet-facing app/indexer containers cannot read the
# userKeyPassword that decrypts your Drive.
#
# If you already have a session from the CLI (e.g. stored in `pass`), copy
# it into place manually.
#
# Usage:
#   scripts/export-session.sh
#
# Requires the proton-drive CLI on PATH and a logged-in session.
set -euo pipefail

# Host-side destination for the exported session. Defaults to the path
# compose.yml mounts into the bridge (AUTH_SESSION_MOUNT). Override when you
# keep the file elsewhere on disk — just update AUTH_SESSION_MOUNT in .env
# to match.
AUTH_SESSION_OUT="${AUTH_SESSION_OUT:-./credentials/auth-session.json}"

mkdir -p "$(dirname "$AUTH_SESSION_OUT")"

if command -v proton-drive >/dev/null 2>&1; then
  echo "=> Generating a fresh session with the proton-drive CLI..."
  echo "   A browser will open (or a sign-in URL will be printed) for your Proton account."
  echo "   Keep the terminal open until the CLI reports the login completed."
  # Force the unsafe_file store + a throwaway cache dir so the fresh session
  # lands in a predictable file we can copy to $AUTH_SESSION_OUT. A real
  # keychain / pass store is never touched.
  export PROTON_DRIVE_CREDENTIALS_STORE=unsafe_file
  export PROTON_DRIVE_CACHE_DIR="$(mktemp -d)"
  trap 'rm -rf "$PROTON_DRIVE_CACHE_DIR"' EXIT
  if proton-drive auth login; then
    if [ -f "$PROTON_DRIVE_CACHE_DIR/auth-session.json" ]; then
      cp "$PROTON_DRIVE_CACHE_DIR/auth-session.json" "$AUTH_SESSION_OUT"
      chmod 600 "$AUTH_SESSION_OUT"
      echo "=> auth session secured at $AUTH_SESSION_OUT (chmod 600)"
    else
      echo "   (no auth-session.json produced — did the CLI login complete?)" >&2
    fi
  else
    echo "   (proton-drive auth login did not complete; see the output above)" >&2
  fi
else
  echo "proton-drive CLI not found on PATH."
  echo "Place a session file at $AUTH_SESSION_OUT manually."
  echo "Format (JSON): {\"session\":{\"uid\":\"...\",\"accessToken\":\"...\",\"refreshToken\":\"...\"},\"userKeyPassword\":\"...\",\"cachePassword\":\"...\"}"
fi
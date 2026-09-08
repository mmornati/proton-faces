#!/bin/sh
# proton-bridge entrypoint.
#
# When PROTON_DRIVE_CREDENTIALS_STORE=pass, bootstrap the encrypted credential
# store before starting the bridge:
#   1. generate a container-local GPG key (no passphrase — the container must
#      boot unattended; the key never leaves the bridge-gnupg volume),
#   2. `pass init` the store in PASSWORD_STORE_DIR,
#   3. one-time migrate a plaintext /data/auth-session.json into the store and
#      remove the plaintext file.
# With any other store backend this script is a no-op and execs the bridge.
set -eu

if [ "${PROTON_DRIVE_CREDENTIALS_STORE:-unsafe_file}" = "pass" ]; then
  GNUPGHOME="${GNUPGHOME:-/secrets/gnupg}"
  PASSWORD_STORE_DIR="${PASSWORD_STORE_DIR:-/secrets/password-store}"
  export GNUPGHOME PASSWORD_STORE_DIR

  GPG_ID="proton-bridge@localhost"
  SESSION_ENTRY="ch.proton.drive/drive-sdk-cli/auth-session"
  PLAINTEXT_SESSION="${DATA_DIR:-/data}/auth-session.json"

  mkdir -p "$GNUPGHOME" "$PASSWORD_STORE_DIR"
  chmod 700 "$GNUPGHOME"

  if ! gpg --batch --list-keys "$GPG_ID" >/dev/null 2>&1; then
    echo "[entrypoint] generating GPG key for the pass credential store" >&2
    gpg --batch --gen-key <<EOF
%no-protection
Key-Type: eddsa
Key-Curve: ed25519
Key-Usage: sign
Subkey-Type: ecdh
Subkey-Curve: cv25519
Subkey-Usage: encrypt
Name-Real: proton-bridge
Name-Email: $GPG_ID
Expire-Date: 0
%commit
EOF
  fi

  if [ ! -f "$PASSWORD_STORE_DIR/.gpg-id" ]; then
    echo "[entrypoint] initializing pass store in $PASSWORD_STORE_DIR" >&2
    pass init "$GPG_ID" >/dev/null
  fi

  if ! pass show "$SESSION_ENTRY" >/dev/null 2>&1 && [ -f "$PLAINTEXT_SESSION" ]; then
    echo "[entrypoint] migrating $PLAINTEXT_SESSION into the pass store" >&2
    pass insert -f -m "$SESSION_ENTRY" < "$PLAINTEXT_SESSION"
    if pass show "$SESSION_ENTRY" >/dev/null 2>&1; then
      echo "[entrypoint] migration complete" >&2
      # Best-effort cleanup only: the plaintext path is usually a compose bind
      # mount, which cannot be unlinked (EBUSY). The pass entry is the
      # authoritative copy, so leaving the file behind is harmless; don't let
      # a failed rm abort the boot under `set -eu` (the misleading "removed"
      # echo below used to mask that failure via its own exit status).
      rm -f "$PLAINTEXT_SESSION" || true
      if [ -e "$PLAINTEXT_SESSION" ]; then
        echo "[entrypoint] note: kept plaintext $PLAINTEXT_SESSION (bind mount cannot be removed)" >&2
      fi
    else
      echo "[entrypoint] ERROR: migration verification failed; keeping $PLAINTEXT_SESSION" >&2
      exit 1
    fi
  fi
fi

exec /usr/local/bin/proton-bridge "$@"

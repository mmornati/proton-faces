# Session file

The `proton-bridge` container authenticates against Proton Drive with the same session file the Proton Drive CLI uses. If you already have the CLI set up, this is a one-liner. If not, you'll need to install the CLI, sign in once, and export the session (about 5 minutes).

## Why a session file?

Proton doesn't publish an OAuth provider for third-party apps, so the bridge speaks the same protocol the CLI does — same auth flow, same session format, same tokens. The session is encrypted at rest and mounted **only into the bridge container** (issue #32): the indexer and the internet-facing `app` container have no path to it, so a vulnerability in one of them cannot leak your session.

## First time: install the CLI and sign in

The bridge reuses a session created by the official **Proton Drive CLI** (`proton-drive`).

**1. Download the CLI** for your platform from the [official Proton Drive CLI page](https://proton.me/download/drive/cli). The page lists SHA-512 checksums; verify the archive before running it. On NAS or embedded x86-64 CPUs (most home servers), use the `linux/x64-baseline` build if the default `linux/x64` one crashes at startup with `Illegal instruction`, and `linux/arm64` on ARM hardware.

**2. Sign in once** — browser-based, no password on the command line:

```bash
proton-drive auth login
```

A browser tab opens to sign in with your Proton account. Keep the terminal open until it prints `Authentication successful` — the session is stored in your OS secret store as soon as login completes. On a machine without a browser (e.g. a headless server), the CLI prints a URL instead; open it on any phone or desktop, and the session is delivered to the machine that ran the command.

**3. Export it for the bridge** with the [bundled helper](#export-with-the-bundled-helper-recommended):

```bash
scripts/export-session.sh
```

This produces `credentials/auth-session.json` with `chmod 600` and prints the path. That's it — continue with the [Installation guide](installation.md#1-get-a-proton-session-file).

If you already completed these steps on another machine, you don't need to log in again — export from the store you used (below) and copy the file over.

## Where the CLI keeps the session

After `auth login`, the Proton Drive CLI stores the session in your OS secret store (or `pass`, if you configured `PROTON_DRIVE_CREDENTIALS_STORE=pass`).

| Platform | Storage |
|---|---|
| Linux (default) | libsecret — GNOME Keyring / KWallet |
| macOS (default) | Keychain entry `ch.proton.drive/drive-sdk-cli` |
| Windows (default) | Credential Manager (`cmdkey /list:ch.proton.drive`) |
| Any + `pass` | GPG-encrypted `pass show ch.proton.drive/drive-sdk-cli/auth-session` |

`scripts/export-session.sh` (bundled) handles every case for you — it runs a fresh, throwaway CLI login in `unsafe_file` mode so it never touches your real keychain. You just need the CLI installed. The one-liners below only matter if you want to export a session that's **already** in a specific store.

## Export with the bundled helper (recommended)

```bash
scripts/export-session.sh
```

The script:

1. Runs `proton-drive auth login` once, with the credential store forced to a throwaway
   plaintext temp dir (`PROTON_DRIVE_CREDENTIALS_STORE=unsafe_file` + a `mktemp -d` cache dir),
   so your real keychain or `pass` store is never touched.
2. Copies the freshly created session into `./credentials/auth-session.json` (overridable via
   `AUTH_SESSION_OUT=`) and applies `chmod 600`.

If a browser cannot open on the machine (headless server), the script prints the sign-in URL —
open it on any phone or desktop.

## Export with `pass` (if the session is already there)

```bash
pass show ch.proton.drive/drive-sdk-cli/auth-session > credentials/auth-session.json
chmod 600 credentials/auth-session.json
```

## Export with the macOS Keychain (if the session is already there)

```bash
security find-generic-password -s "ch.proton.drive/drive-sdk-cli" -w \
    > credentials/auth-session.json
chmod 600 credentials/auth-session.json
```

If you have multiple accounts, `-a <account>` selects which one.

## Where it goes

Once exported, `credentials/auth-session.json` is mounted into the **bridge container only** at `/data/auth-session.json`:

```yaml
# compose.yml (already configured — see issue #32)
services:
  proton-bridge:
    volumes:
      - ${AUTH_SESSION_MOUNT:-./credentials/auth-session.json}:/data/auth-session.json
```

The bridge reads it on first start, decrypts your tokens, and starts streaming the timeline. The mount is **writable** because the SDK rewrites the file when it refreshes tokens — a `:ro` mount would break refresh. `auth-session.json` is excluded from the bridge's `/cache/clear` glob, so that endpoint never deletes it.

The session file is NOT part of the shared `data` volume: the `app` and `indexer` containers have no path to it. Verify after deploying with:

```bash
docker compose exec app test ! -e /data/auth-session.json   # should exit 0
```

!!! danger "Treat this file like a password"
    Anyone with this file can read your encrypted Proton Drive. Never commit it (the repo's `.gitignore` already excludes `credentials/`), never share it, never paste it into a chat.

## Encrypted store (`pass`)

If you'd rather not keep the session as a plaintext file on the host, the SDK supports an encrypted backend backed by `pass` + GPG, and the bridge image ships both. To enable it:

```bash
# .env
PROTON_DRIVE_CREDENTIALS_STORE=pass
```

```bash
docker compose pull proton-bridge   # pass support needs a current image
docker compose up -d                # recreate — `restart` does NOT apply .env changes
```

On first start the bridge entrypoint:

1. Generates a container-local GPG key (no passphrase — the container must boot unattended) in the `bridge-gnupg` volume.
2. Runs `pass init` in the `bridge-pass-store` volume.
3. If `/data/auth-session.json` still exists, inserts it into the store, verifies the read-back, and removes the plaintext file.

Both volumes are mounted **only into the bridge container** — same isolation as the session file (issue #32). Verify with:

```bash
docker compose exec proton-bridge pass show ch.proton.drive/drive-sdk-cli/auth-session
```

To update the session while in `pass` mode, insert the freshly exported file over the existing entry and restart:

```bash
docker compose exec -T proton-bridge pass insert -f -m ch.proton.drive/drive-sdk-cli/auth-session \
    < credentials/auth-session.json
docker compose restart proton-bridge
```

!!! note "What this does and doesn't protect"
    The encrypted store protects the session **at rest on the host**: volume backups, stray reads, an accidental `git add credentials/`. It is **not** a boundary against an attacker already inside the bridge container — key and store are both reachable there by construction. The issue #32 isolation (app/indexer have no path to keyring or store) is unchanged.

To roll back to the plaintext store, extract the session and switch the variable:

```bash
docker compose exec -T proton-bridge pass show ch.proton.drive/drive-sdk-cli/auth-session \
    > credentials/auth-session.json
chmod 600 credentials/auth-session.json
# .env: PROTON_DRIVE_CREDENTIALS_STORE=unsafe_file
docker compose up -d
```

## What if I revoke the session?

If you sign the CLI out, the bridge's next sync fails with a 401-style error. Export a new session and `docker compose restart proton-bridge`.

## Demo mode doesn't need this

If you're only using [demo mode](demo-mode.md), you can skip this entirely — the bridge container doesn't run.

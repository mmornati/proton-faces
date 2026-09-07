# Session file

The `proton-bridge` container authenticates against Proton Drive with the same session file the Proton Drive CLI uses. If you already have the CLI set up, this is a one-liner. If not, you'll need to log into the CLI once and capture the session.

## Why a session file?

Proton doesn't publish an OAuth provider for third-party apps, so the bridge speaks the same protocol the CLI does — same auth flow, same session format, same tokens. The session is encrypted at rest and mounted **only into the bridge container** (issue #32): the indexer and the internet-facing `app` container have no path to it, so a vulnerability in one of them cannot leak your session.

## Where the file lives

The Proton Drive CLI normally keeps the session in your platform's secret store:

| Platform | Storage |
|---|---|
| Linux (with `pass`) | `pass show ch.proton.drive/drive-sdk-cli/auth-session` |
| macOS (Keychain) | Keychain entry `ch.proton.drive/drive-sdk-cli` |
| Windows (Credential Manager) | `cmdkey /list:ch.proton.drive` |

`scripts/export-session.sh` (bundled) handles all three. You just need the CLI installed and logged in once.

## Export with `pass` (Linux)

```bash
pass show ch.proton.drive/drive-sdk-cli/auth-session > credentials/auth-session.json
chmod 600 credentials/auth-session.json
```

## Export with the macOS Keychain

```bash
security find-generic-password -s "ch.proton.drive/drive-sdk-cli" -w \
    > credentials/auth-session.json
chmod 600 credentials/auth-session.json
```

If you have multiple accounts, `-a <account>` selects which one.

## Export with the bundled helper

```bash
scripts/export-session.sh
```

The script writes to `./credentials/auth-session.json` by default (overridable via `AUTH_SESSION_OUT=`). chmod 600 is applied automatically.

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

## What if I revoke the session?

If you sign the CLI out, the bridge's next sync fails with a 401-style error. Export a new session and `docker compose restart proton-bridge`.

## Demo mode doesn't need this

If you're only using [demo mode](demo-mode.md), you can skip this entirely — the bridge container doesn't run.

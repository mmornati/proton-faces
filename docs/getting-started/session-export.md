# Session file

The `proton-bridge` container authenticates against Proton Drive with the same session file the Proton Drive CLI uses. If you already have the CLI set up, this is a one-liner. If not, you'll need to log into the CLI once and capture the session.

## Why a session file?

Proton doesn't publish an OAuth provider for third-party apps, so the bridge speaks the same protocol the CLI does — same auth flow, same session format, same tokens. The session is encrypted at rest and mounted **read-only** into the bridge container.

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

The script tries `pass` first, then macOS Keychain, then Credential Manager. If none match, it prints the steps to set up the Proton Drive CLI.

## Where it goes

Once exported, `credentials/auth-session.json` is mounted read-only into the bridge container:

```yaml
# compose.yml (already configured)
services:
  proton-bridge:
    volumes:
      - ./credentials/auth-session.json:/app/auth-session.json:ro
```

The bridge reads it on first start, decrypts your tokens, and starts streaming the timeline. It never writes to the file.

!!! danger "Treat this file like a password"
    Anyone with this file can read your encrypted Proton Drive. Never commit it (the repo's `.gitignore` already excludes `credentials/`), never share it, never paste it into a chat.

## What if I revoke the session?

If you sign the CLI out, the bridge's next sync fails with a 401-style error. Export a new session and `docker compose restart proton-bridge`.

## Demo mode doesn't need this

If you're only using [demo mode](demo-mode.md), you can skip this entirely — the bridge container doesn't run.

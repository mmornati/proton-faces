#!/usr/bin/env bash
# Backup the search index (SQLite only — thumbnails are re-derivable).
# Usage: scripts/backup.sh [backup-dir]
#
# Uses `VACUUM INTO` through Python's sqlite3 module: a plain `cp` of a
# WAL-mode database while the indexer is writing produces a torn copy
# (the -wal file holds committed pages the main file does not). The
# result is compacted, consistent, and readable only by the owner — it
# contains every user's bcrypt hash and encrypted TOTP secret.
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
BACKUP_DIR="${1:-./backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SRC="$DATA_DIR/index.sqlite3"
DEST="$BACKUP_DIR/index-${STAMP}.sqlite3"

mkdir -p "$BACKUP_DIR"
if [ ! -f "$SRC" ]; then
  echo "no index yet at $SRC"
  exit 0
fi
umask 077
python3 - "$SRC" "$DEST" <<'PYSNAP'
import sqlite3, sys
src, dest = sys.argv[1], sys.argv[2]
conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30)
try:
    conn.execute("VACUUM INTO ?", (dest,))
finally:
    conn.close()
PYSNAP
chmod 600 "$DEST"
echo "backed up index -> $DEST"

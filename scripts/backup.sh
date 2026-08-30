#!/usr/bin/env bash
# Backup the search index (SQLite only — thumbnails are re-derivable).
# Usage: scripts/backup.sh [backup-dir]
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
BACKUP_DIR="${1:-./backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$BACKUP_DIR"
if [ -f "$DATA_DIR/index.sqlite3" ]; then
  cp "$DATA_DIR/index.sqlite3" "$BACKUP_DIR/index-${STAMP}.sqlite3"
  echo "backed up index -> $BACKUP_DIR/index-${STAMP}.sqlite3"
else
  echo "no index yet at $DATA_DIR/index.sqlite3"
fi
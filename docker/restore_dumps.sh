#!/bin/bash
# Restore pg_dump custom-format files (<db>.dump) into the t2sql-postgres container, one database each.
# Existing databases with the same name are dropped first, so the script can be re-run.
# usage: docker/restore_dumps.sh data/pg_dumps
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash on Windows would otherwise rewrite /tmp/... arguments

DUMP_DIR="${1:?usage: restore_dumps.sh <dir with .dump files>}"
C=t2sql-postgres

docker exec "$C" mkdir -p /tmp/dumps
docker cp "$DUMP_DIR/." "$C:/tmp/dumps/"
for f in "$DUMP_DIR"/*.dump; do
  db="$(basename "$f" .dump)"
  echo "restoring $db"
  docker exec "$C" dropdb -U postgres --if-exists --force "$db"
  docker exec "$C" createdb -U postgres -E UTF8 --locale=en_US.utf8 "$db"
  docker exec "$C" pg_restore --exit-on-error --no-owner --no-acl -U postgres -d "$db" "/tmp/dumps/$db.dump"
done
docker exec "$C" rm -rf /tmp/dumps

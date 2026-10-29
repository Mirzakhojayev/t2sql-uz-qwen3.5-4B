#!/bin/bash
# Runs once, on first start of an empty data volume.
# Creates the read-only role that the eval harness uses to execute generated SQL.
# `postgres` (admin) is used only for loading databases.
set -euo pipefail

psql -v ON_ERROR_STOP=1 -v ro_pw="$RO_PASSWORD" --username "$POSTGRES_USER" --dbname postgres <<'EOSQL'
CREATE ROLE sql_ro LOGIN PASSWORD :'ro_pw';

-- pg_read_all_data (PG14+): SELECT on every table/view/sequence, no write rights at all
GRANT pg_read_all_data TO sql_ro;

-- defaults for every session of this role; the harness also sets them per connection
ALTER ROLE sql_ro SET default_transaction_read_only = on;
ALTER ROLE sql_ro SET statement_timeout = '10s';
ALTER ROLE sql_ro SET idle_in_transaction_session_timeout = '30s';

-- no temp tables for the read-only role
REVOKE TEMPORARY ON DATABASE postgres FROM PUBLIC;
-- new databases inherit from template1, so future DBs get the same
REVOKE TEMPORARY ON DATABASE template1 FROM PUBLIC;
EOSQL

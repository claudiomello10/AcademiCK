#!/usr/bin/env bash
# Run the service test suites against the running dev compose stack.
#
# Usage: scripts/run-tests.sh [--llm] [--e2e] [pytest args...]
#   --llm   also run tests that call the configured real LLMs (costs money)
#   --e2e   also run the full-stack end-to-end suite (needs the whole stack up)
#
# Infra (Postgres/Redis/Qdrant) is only exposed on the docker network, so
# connection URLs are derived from the container IPs plus .env credentials.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

RUN_LLM=0
RUN_E2E=0
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --llm) RUN_LLM=1 ;;
    --e2e) RUN_E2E=1 ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

if [ ! -f "$ROOT/.env" ]; then
  echo "error: $ROOT/.env not found — tests use the dev stack's credentials" >&2
  exit 1
fi
# Parse .env like docker compose does (literal values, no shell expansion);
# sourcing would break on spaces and strip the JSON quotes in AVAILABLE_MODELS.
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue ;; esac
  key="${line%%=*}"
  value="${line#*=}"
  case "$value" in
    \"*\") value="${value#\"}"; value="${value%\"}" ;;
    \'*\') value="${value#\'}"; value="${value%\'}" ;;
  esac
  export "$key=$value"
done < "$ROOT/.env"

container_ip() {
  docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$1" 2>/dev/null
}

PG_IP="$(container_ip academick-postgres)"
REDIS_IP="$(container_ip academick-redis)"
QDRANT_IP="$(container_ip academick-qdrant)"
EMBEDDING_IP="$(container_ip academick-embedding)"
INTENT_IP="$(container_ip academick-intent)"
if [ -z "$PG_IP" ] || [ -z "$REDIS_IP" ] || [ -z "$QDRANT_IP" ] || [ -z "$EMBEDDING_IP" ] || [ -z "$INTENT_IP" ]; then
  echo "error: dev stack not running (postgres/redis/qdrant/embedding/intent containers not found)." >&2
  echo "start it with: docker compose up -d postgres redis qdrant embedding-service intent-service" >&2
  exit 1
fi

export DATABASE_URL="postgresql://${POSTGRES_USER:-academick}:${POSTGRES_PASSWORD}@${PG_IP}:5432/${POSTGRES_DB:-academick}"
export REDIS_URL="redis://:${REDIS_PASSWORD}@${REDIS_IP}:6379/0"
export QDRANT_HOST="$QDRANT_IP"
export QDRANT_PORT=6333
export EMBEDDING_SERVICE_URL="http://${EMBEDDING_IP}:8002"
export INTENT_SERVICE_URL="http://${INTENT_IP}:8001"
# Snapshot metadata is written by the gateway (in-process here); the real
# mount (data/qdrant_snapshots) is root-owned, so tests get their own dir.
export SNAPSHOT_DIR="${TMPDIR:-/tmp}/academick-test-snapshots"

MARKER_ARGS=()
if [ "$RUN_LLM" -eq 1 ]; then
  MARKER_ARGS=(-m "llm or not llm")
fi

failed=0

for dir in "$ROOT"/services/*/; do
  svc="$(basename "$dir")"
  if [ ! -d "$dir/tests" ]; then
    echo "-- $svc: no tests, skipping"
    continue
  fi
  venv="$dir/.venv-test"
  if [ ! -x "$venv/bin/python" ]; then
    echo "== $svc: creating test venv =="
    python3 -m venv "$venv" || exit 1
  fi
  "$venv/bin/pip" install -q -r "$dir/requirements.txt" -r "$dir/requirements-dev.txt" || exit 1
  echo "== $svc =="
  (cd "$dir" && "$venv/bin/python" -m pytest "${MARKER_ARGS[@]}" "${EXTRA_ARGS[@]}") || failed=1
done

if [ "$RUN_E2E" -eq 1 ]; then
  venv="$ROOT/tests/e2e/.venv-test"
  if [ ! -x "$venv/bin/python" ]; then
    echo "== e2e: creating test venv =="
    python3 -m venv "$venv" || exit 1
  fi
  "$venv/bin/pip" install -q -r "$ROOT/tests/e2e/requirements.txt" || exit 1
  echo "== e2e =="
  (cd "$ROOT/tests/e2e" && "$venv/bin/python" -m pytest "${EXTRA_ARGS[@]}") || failed=1
fi

exit "$failed"

#!/bin/bash
# Build and (re)start Graft. Runs on the deployment host, mirroring
# graphmind's deploy_api.sh:
#   /opt/graft/deployment/deploy.sh
set -euo pipefail

REPO="${GRAFT_REPO:-/opt/graft}"
DATA_DIR="${GRAFT_DATA_DIR:-/opt/graft-data}"

cd "$REPO"

# The data is shipped separately by sync-data.sh; refuse to start with a
# half-populated mount rather than failing deep inside data loading.
for required in edges dense metadata s2orc minilm_onnx; do
    [ -d "${DATA_DIR}/${required}" ] || {
        echo "missing ${DATA_DIR}/${required} -- run deployment/sync-data.sh first" >&2
        exit 1
    }
done
[ -f .env ] || { echo "missing ${REPO}/.env (needs OPENAI_API_KEY)" >&2; exit 1; }

# The GraphMind stack's network, which its Caddy is already on
docker network inspect deployment_default >/dev/null 2>&1 || {
    echo "deployment_default network missing -- is the GraphMind stack up?" >&2
    exit 1
}

git pull --quiet
docker compose -f deployment/docker-compose.yml up -d --build --quiet-pull
docker image prune -f >/dev/null

echo "Waiting for the corpus to load..."
for _ in $(seq 1 60); do
    status=$(docker inspect --format '{{.State.Health.Status}}' \
        "$(docker compose -f deployment/docker-compose.yml ps -q graft)" 2>/dev/null || echo starting)
    [ "$status" = "healthy" ] && { echo "healthy"; exit 0; }
    sleep 5
done

echo "did not become healthy; recent logs:" >&2
docker compose -f deployment/docker-compose.yml logs --tail 40 graft >&2
exit 1

#!/usr/bin/env bash
# Start all local services (dev mode) — full stack.
# Brings up the Docker stack (TimescaleDB, Qdrant, Neo4j, Redis, Prometheus,
# Grafana) then launches the autonomous orchestrator and the API server.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

echo "==> Starting Docker stack (timescaledb, qdrant, neo4j, redis, prometheus, grafana)"
docker compose up -d

echo "==> Starting autonomous orchestrator"
python main.py orchestrate &
ORCH_PID=$!

echo "==> Starting API server on :8000 (UI proxies /api here)"
python main.py serve &
API_PID=$!

trap 'kill $ORCH_PID $API_PID 2>/dev/null || true' EXIT

echo "==> UI: cd ui && npm install && npm run dev  (port 3001)"
echo "==> Monitoring:"
echo "     Prometheus  -> http://localhost:9090"
echo "     Grafana     -> http://localhost:3000  (admin / \$GRAFANA_PASSWORD or admin)"
echo "     Neo4j       -> http://localhost:7474"
wait $API_PID

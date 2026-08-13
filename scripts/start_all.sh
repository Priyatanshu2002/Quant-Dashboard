#!/usr/bin/env bash
# Start all local services (dev mode).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

echo "==> Starting API server on :8000 (UI proxies /api here)"
python main.py serve &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

echo "==> UI: cd ui && npm install && npm run dev  (port 3001)"
echo "==> Monitoring: Prometheus :9090, Grafana :3000 (docker compose up -d)"
wait $API_PID

#!/usr/bin/env bash
# Project Agonistes — bootstrap (dev mode, no Docker required).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Creating virtualenv"
python -m venv .venv
# shellcheck disable=SC1091
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

echo "==> Installing core dependencies"
pip install -U pip
pip install -e ".[dev]"

echo "==> Environment file"
if [ ! -f .env ]; then cp .env.example .env; fi
echo "    .env ready — add OPENROUTER_API_KEY when you reach Phase 4."

echo "==> Optional extras (heavy):"
echo "    pip install -e '.[ml]'   # TFT training (torch + pytorch-forecasting)"
echo "    pip install -e '.[llm]'  # LangGraph debate with real LLM calls"
echo "    pip install -e '.[rl]'   # PPO agent (gymnasium + stable-baselines3)"
echo "    pip install -e '.[db]'   # TimescaleDB / Qdrant / Neo4j clients"

echo "==> Smoke test"
python scripts/smoke_test.py

echo "Done. Try: python main.py screen && python main.py backtest --symbol SPY"

#!/usr/bin/env bash
# Run a strategy backtest with the full report.
# Usage: ./scripts/run_backtest.sh [SYMBOL] [STRATEGY]
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

SYMBOL="${1:-SPY}"
STRATEGY="${2:-ma_cross}"
python main.py backtest --symbol "$SYMBOL" --strategy "$STRATEGY"

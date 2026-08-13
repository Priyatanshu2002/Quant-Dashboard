# Project Agonistes

Multi-asset paper-trading + backtesting platform: universal screener → ~75–120 feature vectors →
Temporal Fusion Transformer signals → adversarial Bull/Bear LLM debate (LangGraph + OpenRouter) →
PPO execution agent → Black-Litterman portfolio manager, all validated with a full cost-model
backtesting suite. **Paper trading only until Go-Live Criteria are met** (see plan §12).

> Source of truth: `implementation_plan_v2.md`. This repo follows its directory layout and
> module specifications. One deliberate addition: a shared `core/` package (config, storage
> adapter, logging, event bus) so every layer talks to the same store.

## Layout

```
core/                  shared config, storage adapter (SQLite dev / TimescaleDB), events
data_ingestion/        price, fundamental, sentiment, macro, on-chain feeds
screener/              asset universe, composite scorer, top-N selector
feature_engineering/   ~75 features/asset/timeframe → feature store
transformer_model/     TFT (pytorch-forecasting) + configs per timeframe
langgraph_app/         LangGraph pipeline: analyst → bull/bear → gating → PM → RL → reflection
rl_agent/              Gymnasium env + PPO (Stable-Baselines3)
portfolio_manager/     Black-Litterman, ledger, risk budget, PnL
backtesting/           engine, cost model, walk-forward, purged K-fold, Monte Carlo, metrics
ui/                    React dashboard (screener, financials, backtest, portfolio, debate)
databases/             TimescaleDB schema + Qdrant/Neo4j configs
scripts/               bootstrap, start_all, run_backtest, smoke test
```

## Quickstart (dev mode — no Docker required)

```bash
# Project root: C:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice
cd "C:/Users/Priyatanshu Ghosh/Documents/Python Practice/CFA Practice"
python -m venv .venv && source .venv/Scripts/activate      # git-bash on Windows
pip install -e ".[dev]"
cp .env.example .env                                       # add OPENROUTER_API_KEY when needed
python scripts/smoke_test.py                               # full Phase-1 pipeline on real data
```

`core/db.py` runs on SQLite by default (`data/agonistes_dev.db`). To switch to TimescaleDB:
`docker compose up -d` then set `DATABASE_URL=postgresql://agonistes:...@localhost:5432/agonistes`.

## LLM provider (Bull/Bear debate)

The debate layer is provider-agnostic — **Nous Portal** or **OpenRouter**:

| | Nous Portal (recommended if you have Nous balance) | OpenRouter |
|---|---|---|
| Key | `NOUS_API_KEY` from [portal.nousresearch.com → API Keys](https://portal.nousresearch.com) | `OPENROUTER_API_KEY` from openrouter.ai/settings/keys |
| Base URL | `https://inference-api.nousresearch.com/v1` | `https://openrouter.ai/api/v1` |
| Model | `deepseek/deepseek-v4-flash-0731` | `deepseek/deepseek-v4-flash-0731` |

Put the key in `.env` — the provider auto-detects. Without any key the debate
graph runs in deterministic heuristic mode (no cost).

## CLI

```bash
agonistes backfill --tickers AAPL MSFT BTC-USD --period 2y
agonistes features --tickers AAPL
agonistes screen
agonistes backtest --tickers AAPL SPY --strategy ma_cross
agonistes smoke
```

## Status vs plan

| Plan phase | Status |
|---|---|
| 1 Foundation | ✅ 100% — full-universe backfill (31 symbols, ~67k bars, 5y daily), technical features, feature store (38,235 vectors), screener with **4 live qualified candidates**, macro (VIX 15.3 / US10Y 4.68% / DXY 99.9), news sentiment live |
| 2 Fundamentals | ✅ (news sentiment = cutoff) — SEC EDGAR 10/10 US equities, DCF, yfinance info 17 tickers, earnings calendar 250 rows, macro via yfinance (keyless), **news sentiment live: 925 events / 21 symbols** (Yahoo News + Google RSS + Yahoo RSS + StockTwits); GDELT/Reddit best-effort (network-blocked here); NSE/BSE watcher stub (SEBI) |
| 3 Transformer | 🟡 code + configs complete, needs `pip install -e ".[ml]"` + training on the 38k-vector store |
| 4 LLM Debate | ✅ graph + 9 nodes run real LangGraph cycles on Nous Portal (`deepseek-v4-flash-0731`) |
| 5 RL Agent | 🟡 env + reward + PPO code, needs `.[rl]` + training |
| 6 Backtesting | ✅ engine + full cost model + all 10 metrics + walk-forward/regimes/Monte Carlo |
| 7 UI/Monitoring | ✅ React UI builds; Prometheus/Grafana configs ready |
| 8 Paper trading | ⬜ Go-Live gated |

**Tests:** 70 pytest cases green (`python -m pytest tests/ -q`).

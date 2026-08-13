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

> ⚠️ **Accuracy audit (2026-08-14):** The previous table marked several phases "✅ done" on the strength of a *proxy* (one live data channel) rather than the phase's full checklist. The corrected table below is honest about what is built-and-running vs. stubbed / gated / proxy-verified. See `docs/AUDIT.md` for the full gap analysis and remediation plan.

| Plan phase | Status (corrected) | Reality check |
|---|---|---|
| 1 Foundation | 🟡 **partial** — not the "100%" previously claimed | Docker stack (TimescaleDB/Qdrant/Neo4j) **never set up** (Docker not installed); backfill is **31 symbols, not the full universe**; screener + technical features + feature store + macro (3 keyless numbers) run, but there is **no live macro regime layer** and only a 3-series keyless macro path. |
| 2 Fundamentals | 🟡 **partial** — the "✅ (news sentiment = cutoff)" claim was a proxy | SEC EDGAR parser (10/10 US) + DCF + yfinance info (17 tickers) + earnings calendar run. **Not done from the phase checklist:** earnings-call **transcript LLM sentiment** (no transcript module exists), NSE/BSE watcher is a **stub** (SEBI-blocked), FRED/BLS **key-gated**, GDELT/Reddit **network-blocked** here. No data lineage / provenance on any stored value. |
| 3 Transformer | 🟡 code + configs complete, **untrained** | Needs `pip install -e ".[ml]"` + training on the 38k-vector store. Correct as marked. |
| 4 LLM Debate | 🟡 **partial** — "9 nodes run real cycles" is true but overstates readiness | LangGraph graph + 9 nodes execute real cycles on Nous. **However** Node B attaches TFT only *when a trained checkpoint exists* — currently it defaults to `NEUTRAL/0.0`, so debates run on an **empty ML signal**. No RAG, no citations, no persistent research/knowledge retrieval. |
| 5 RL Agent | 🟡 env + reward + PPO code, **untrained** | Needs `.[rl]` + training. Correct as marked. |
| 6 Backtesting | 🟡 **partial** — engine complete, but has only run hand-written strategies | Engine + cost model + 10 metrics + walk-forward/regimes/Monte Carlo exist. **Has not validated the system's actual signals** (untrained TFT/RL, debate outputs) — only canned strategies (`ma_cross`, `shuffled`, …). |
| 7 UI/Monitoring | 🟡 **partial** — React UI builds, monitoring is config-only | React UI builds; **Grafana dashboards are pending, Prometheus/Grafana are not actually running**; UI is a dashboard, not a research/analysis workbench (no search, no report export). |
| 8 Paper trading | ⬜ Go-Live gated | Correct as marked. |

**Tests:** 70 pytest cases green (`python -m pytest tests/ -q`).

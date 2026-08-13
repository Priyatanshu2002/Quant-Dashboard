# Handoff Prompt — paste this verbatim into a new Hermes chat/session

You are continuing work on **Project Agonistes**, a multi-asset paper-trading + backtesting
platform. A previous session built and verified it. Your job: understand the current state,
then continue the roadmap IN ORDER, delivering results **visually** (serve the UI, take
screenshots, show charts) — never just terminal dumps.

## Project facts (verified, do not re-derive)

- **Location:** `C:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice`
- **Python:** use `.venv\Scripts\python` from the project root (Windows + git-bash; POSIX syntax in terminal)
- **Storage:** SQLite dev mode at `data/agonistes_dev.db` (TimescaleDB schema is ready; Docker is NOT installed on this machine — do not attempt docker)
- **LLM:** Nous Portal already configured — `NOUS_API_KEY` is set in `.env` (never print or echo the key). Base: `https://inference-api.nousresearch.com/v1`. **Use ONLY model `deepseek/deepseek-v4-flash-0731` — never switch to other models.**
- **Already installed:** pandas, numpy, ta, yfinance, lxml, scipy, pydantic, langgraph, openai, instructor, pytest
- **NOT installed (heavy extras, install only when needed):** `pip install -e ".[ml]"` (torch/pytorch-forecasting), `pip install -e ".[rl]"` (gymnasium/stable-baselines3)

## What exists and is verified (do NOT rebuild)

- **Data:** 31 symbols (US/Indian equities, ETFs, bonds, FX) × 5y daily ≈ 67k OHLCV bars + crypto (BTC/ETH/SOL/BNB/XRP/DOGE — **user deprioritized crypto; leave it alone**)
- **Feature store:** 39,695 labeled SWING feature vectors / 31 symbols (`feature_vectors` table)
- **Fundamentals:** 10 US equities via SEC EDGAR XBRL parser (live-verified), 17 yfinance info snapshots, 250 earnings-calendar dates
- **Macro (keyless):** `data_ingestion/macro_feeds/yfinance_macro.py` — VIX 15.3, US10Y 4.68%, DXY 99.9, gold 5d live (2Y spread = None: no keyless source reachable; leave graceful)
- **Sentiment (NEWS SENTIMENT ANALYSIS = the completed cutoff):** `news_aggregator.py` multi-source (Yahoo News, Google RSS, Yahoo RSS, StockTwits) — ~1,000+ events / 21 symbols. GDELT/Reddit code exists but are network-blocked here (best-effort only)
- **Screener:** live, 4 qualified candidates above threshold 60 (NVDA 66.4, XOM 64.6, MSFT 61.8, SBIN.NS 60.4)
- **LangGraph debate:** 9 nodes, runs REAL LLM cycles on Nous (`python -m langgraph_app.src.graph_definition --symbol AAPL`)
- **Backtesting:** engine + full cost model + all 10 metrics + walk-forward + Monte Carlo + regime tests (all tested)
- **UI:** React+Vite dashboard in `ui/` (5 pages + DCFGauge + charts) — **builds, but was never served end-to-end. SERVING IT AND SHOWING SCREENSHOTS IS A TOP PRIORITY.**
- **Tests:** 70 pytest cases green — keep them green after every change (`python -m pytest tests/ -q`)

## User constraints (hard rules)

1. **Show results visually** — serve `python main.py serve` + `cd ui && npm run dev`, open pages in the browser, take screenshots, attach them in chat (MEDIA:). For data analysis use charts (matplotlib/plotly in scripts under `scripts/`). Terminal text alone is NOT acceptable delivery.
2. **Ignore crypto** — do not extend, backfill, or verify crypto work; leave existing crypto rows untouched.
3. **Nous Portal only, model `deepseek/deepseek-v4-flash-0731` only.** Never touch other models.
4. Keep `.env` secrets private — never print the API key.
5. Docker is unavailable — never attempt `docker compose`; keep using SQLite dev mode.
6. Do not break the 70-test suite; add tests for anything new.

## Next steps IN ORDER (start with #1 unless user redirects)

1. **Serve the platform visually (first!):** start `python main.py serve` (port 8000) and `cd ui && npm run dev` (port 3001), open the dashboard in the browser, screenshot every page (Screener, Financials w/ DCF gauge, Backtests, Portfolio, Debate) and present them with MEDIA: paths. Fix any wiring bugs found (UI ↔ API).
2. **TFT training (Phase 3):** `pip install -e ".[ml]"`, then train on the 39,695-vector store with walk-forward CV (`python -m transformer_model.train --config transformer_model/configs/tft_swing.yaml`). Show loss curves + validation metrics visually. Checkpoints → `data/checkpoints/`.
3. **Wire TFT into the debate:** Node B already picks up checkpoints automatically; run live debate cycles and show Bull/Bear theses + gating visually.
4. **RL agent (Phase 5):** `pip install -e ".[rl]"`, train PPO (`python -m rl_agent.train_rl`), show reward curves; Node H auto-uses the checkpoint.
5. **Scheduled paper-trading loop (Phase 8):** daily cron — backfill increment, sentiment sweep, features, screener, debate cycles, portfolio snapshot. Use Hermes cronjob or a plain scheduler script under `scripts/`.
6. **Go-Live gating:** only after Sharpe > 1.5 over 90 paper-trading days (per plan §12). Do NOT deploy real capital.

## Gotchas

- SQLite concurrent writers can lock — the storage layer now uses `_session()` (explicit close); keep writes sequential in scripts.
- Overlapping backfills used to duplicate bars — `query_ohlcv` dedupes now; do not "fix" this again.
- yfinance earnings calendar needs `lxml` (installed).
- `^2YY` doesn't exist on Yahoo — yield_curve_spread stays None; that's expected.
- GDELT/Reddit/Treasury/FRED hosts are blocked on this network — rely on yfinance/CoinGecko/StockTwits/Google News paths.
- The screener threshold (60.0) and weights are per plan §2 — don't tune them to "force" candidates; report honestly.

## Definition of done for this session

Every step you complete must end with a **visible artifact**: browser screenshot, chart PNG, or rendered report — attached in chat, with a one-paragraph explanation. Start by serving the dashboard and showing me the current state of the platform.

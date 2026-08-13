# Handoff Prompt — Next Hermes Session
## Project: Agonistes (Automated Quant Trading System)
### Workspace: `C:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice`

---

## 1. Context: What This Project Is

Agonistes is a fully automated, multi-asset quant trading system. Pipeline order:
1. **Data Ingestion** (`data_ingestion/`) — prices, fundamentals, SEC filings, 3-statement, macro, on-chain, sentiment, Neo4j graph, Qdrant vectors
2. **Feature Engineering** (`feature_engineering/`) — 75-column feature vectors
3. **Screener** (`screener/`) — scores universe, picks top-N
4. **LangGraph Debate** (`langgraph_app/`) — 9-node adversarial debate
5. **RL Agent** (`rl_agent/`) — PPO sizing/timing
6. **Portfolio Manager** (`portfolio_manager/`) — risk budgeting
7. **Backtesting** (`backtesting/`) — strategy validation
8. **Strategy Builder** (`strategy_builder/`) — 37-model benchmark
9. **Orchestrator** (`orchestrator.py`) — autonomous scheduler
10. **REST API + UI** (`core/api_server.py` + `ui/`) — serves React UI + live model viewer

Reference: `implementation_plan_v2.md` (project root) and `docs/system_visual_map.md`.

---

## 2. Current Status — What Is LIVE and Working

- **Full data pipeline verified end-to-end on real data** (SQLite dev DB `data/agonistes_dev.db`): prices (37 syms, 5y), feature vectors (39k, incl. labeled `future_return_5d`), fundamentals, company profiles, 3-statement history, macro, news/StockTwits/Yahoo-RSS sentiment, LLM-analyst verdicts.
- **All free/no-key data sources wired into the orchestrator as daily jobs** (committed in `1c2ba31`, pushed):
  - `ingest_prices`, `ingest_fundamentals`, `ingest_macro` (treasury/VIX/yfinance/crypto-global), `ingest_edgar` (SEC 3-statement), `ingest_profiles`, `ingest_news` (news/GDELT/Reddit + LLM), `build_features`, `screen_and_debate`. Weekly: `model_drift_check`, `ingest_graph`, `ingest_vectors`, `quick_benchmark`.
- **Fixed two latent bugs**: `job_ingest_news` imported nonexistent `run_sentiment_sweep`; `yfinance_earnings.refresh_info_snapshot` divide-by-`None`.
- **Neo4j writer + Qdrant vector pipeline** built (GAP 1/2 of earlier handoff) — graceful degradation verified; containers not running (no Docker).
- **Monitoring stack** (docker-compose: TimescaleDB/Qdrant/Neo4j/Redis/Prometheus/Grafana) + 2 Grafana dashboards + `prometheus.yml` — configured, not running (no Docker).
- **Git**: `main` pushed to `github.com/Priyatanshu2002/Quant-Dashboard`; 4 commits ahead of origin's initial commit (aec7991 → 973db74 → 665117f → 1c2ba31). gh CLI authenticated (device flow, repo scope).

---

## 3. IN-PROGRESS / UNCOMMITTED (This session's pending work)

**The CFA-standard 3-statement + DCF model is BUILT, SERVING LIVE, and COMMITTED+PUSHED** (commits `94a1d6e` + `9fe4553`, on `main` and `origin/main`). Working tree clean.

- `valuation/cfa_model.py`, `valuation/viewer.html`, `valuation/__init__.py` — committed.
- `core/api_server.py` `/api/model` + `/model` routes — committed.

**The API server is currently RUNNING on `http://127.0.0.1:8000`** (background process). The live viewer is at **http://127.0.0.1:8000/model** (defaults to AAPL).

Verified at HTTP layer: `/api/model?symbol=AAPL` returns a coherent model (WACC 8.63%, intrinsic $200.2 vs price $302.25, margin −33.8%); `/model` serves `viewer.html` byte-identical (8481 B, `text/html`). `ruff` + `py_compile` clean; 121 tests pass.

**REMAINING ACTION:**
1. **Browser visual verification — DONE.** The browser tool is fixed and the `/model` viewer is visually confirmed rendering correctly.

**Browser setup (how it was fixed):**
- Root cause was NOT `allow_private_urls` — it was that attaching to the user's running Chrome required an interactive "Allow remote debugging" click, and the daemon had 0 active browser connections (`browser-use --doctor`).
- Fix: a dedicated **headless Chrome** runs on CDP port `9222`, and Hermes is configured to route `browser_exec` through it via `browser.cdp_url: http://127.0.0.1:9222` (set with `hermes config set browser.cdp_url "http://127.0.0.1:9222"`). No popup needed.
- **To (re)start the headless Chrome if it's not running:**
  ```bash
  "/c/Program Files/Google/Chrome/Application/chrome.exe" \
    --headless=new --remote-debugging-port=9222 \
    --user-data-dir="$LOCALAPPDATA/hermes/cache/browser-use/chrome-cdp" \
    --no-first-run --no-default-browser-check --disable-gpu about:blank
  ```
  Verify: `curl -s http://127.0.0.1:9222/json/version` returns `webSocketDebuggerUrl`.
- Verified render (AAPL): WACC 8.63%, intrinsic $200 vs price $302 (margin −33.8%), scenarios Base $200/Bull $220/Bear $182, 5×5 sensitivity, 3-statement history, balance equation ✓. Also fixed a viewer bug: WACC weight row rendered `NaN%` — now `98% / 2%` (committed `3bb5ff5`).

---

## 4. BLOCKED ITEMS — Need the User's Input / Action

1. **API keys (empty in `.env`)** — to unlock: `FRED_API_KEY` (fred.stlouisfed.org), `BLS_API_KEY` (data.bls.gov), `DUNE_API_KEY` (dune.com). Without them, FRED/BLS macro and Dune on-chain stay inactive (all other macro/on-chain already works key-free). The user said they are gathering these.
2. **Docker not installed** — Neo4j (graph) and Qdrant (vectors) can't start; the weekly `ingest_graph` / `ingest_vectors` jobs need `docker compose up -d`. User is thinking through whether to install Docker here or run the stack elsewhere.
3. **Sentiment tuning (user is deciding):** (a) GDELT returns 429 on rapid polling — needs politeness delays; (b) Reddit returns 0 — needs User-Agent/OAuth; or accept news+StockTwits as sentiment coverage.
4. **TimescaleDB backend incomplete** — `core/db.py` `TimescaleStorage` has 8 `NotImplementedError`s (fundamentals, sentiment, statements, trades, profiles, LLM, earnings, macro only work on SQLite). Decide whether to stay on SQLite or finish the Timescale backend.
5. **Drift-check is a no-op** — `job_model_drift_check` calls `query_recent_gating`, which doesn't exist on the db (guarded by `hasattr`); outcome tracking isn't wired.
6. **Prometheus metrics not emitted** — `core/` doesn't emit the metrics the Grafana dashboards reference (`agonistes_http_requests_total`, etc.); the API/event bus needs a `/metrics` endpoint for the dashboards to be live.
7. **RL training environment uses synthetic data** — `rl_agent/environment.py` uses random returns + zero-filled observations, not real feature vectors. Needs wiring to the real feature store before meaningful PPO training.
8. **Before any training compute**: populate real data (already largely done) + confirm TFT/strategy_builder loaders consume the store (`transformer_model/dataset.py`, `strategy_builder/features.py` take DataFrames a caller must supply).

---

## 5. Key Files to Know

- `orchestrator.py` — job registry `ALL_JOBS`, daily/weekly/monthly loops. Add new scheduled jobs here.
- `core/db.py` — unified storage adapter (SQLite dev + Timescale). Do NOT change; Neo4j/Qdrant have their own wrappers.
- `core/api_server.py` — stdlib HTTP API (`/api/model`, `/api/financials`, `/api/sentiment`, etc.). Serves the `/model` viewer.
- `valuation/cfa_model.py` + `valuation/viewer.html` — the CFA model + live viewer (in-progress).
- `data_ingestion/graph_feeds/neo4j_writer.py`, `data_ingestion/vector_feeds/qdrant_writer.py` — Neo4j/Qdrant writers (idempotent, graceful).
- `data_ingestion/fundamental_feeds/*` — SEC EDGAR, 3-statement, yfinance financials, DCF.
- `scripts/sentiment_sweep.py`, `scripts/backfill_fundamentals.py` — reusable sweeps (UNIVERSE + CIK_MAP used by orchestrator jobs).
- `langgraph_app/src/graph_definition.py` — the LangGraph graph (`build_graph()`), NOT `graph.py`.

---

## 6. Commands to Run / Verify

```bash
# Run the scheduler (dry-run prints schedule)
python orchestrator.py --list
python orchestrator.py --dry-run

# Force one job immediately
python orchestrator.py --now ingest_macro
python orchestrator.py --now ingest_profiles
python orchestrator.py --now ingest_news

# Serve the API (currently running on :8000)
python main.py serve --port 8000

# CFA model (live)
#   browser:  http://127.0.0.1:8000/model
#   JSON:     http://127.0.0.1:8000/api/model?symbol=AAPL

# Tests + lint
python -m pytest -q
python -m ruff check --select E,F valuation/ orchestrator.py core/ data_ingestion/

# Populate 3-statement for another symbol via API
curl -X POST "http://127.0.0.1:8000/api/fundamentals/refresh?symbol=MSFT"
```

---

## 7. Recommended Next Steps (priority order)

1. ~~**Commit + push** the in-progress `valuation/` package and `core/api_server.py`~~ — DONE (`94a1d6e`, pushed; tree clean).
2. ~~**Restart the browser daemon** so `allow_private_urls: true` takes effect, then visually verify the `/model` viewer renders~~ — DONE. Browser fixed via headless Chrome on CDP `:9222` + `browser.cdp_url`; viewer verified (see §3). Ensure the headless Chrome is running (command in §3) before using `browser_exec`.
3. When the user supplies **FRED/BLS/Dune keys**, add them to `.env` and verify FRED/BLS/Dune pull real data.
4. Decide on **Docker**: bring up Neo4j/Qdrant and verify `ingest_graph`/`ingest_vectors` populate live.
5. **Sentiment tuning**: add GDELT politeness + decide on Reddit auth.
6. Then the pre-training items: wire the **RL environment to real data**, finish the **Timescale backend** or stay SQLite, wire **drift-check outcome tracking**, add the **Prometheus `/metrics` endpoint**.

---

## 8. Constraints / Rules

- **Do not change `core/db.py`** (stable storage interface); Neo4j/Qdrant get their own client wrappers.
- **All writers idempotent** (MERGE / deterministic upserts).
- **Graceful degradation** — if Neo4j/Qdrant/LLM/feeds are down, the debate and pipeline still run (never raise into the debate).
- **Lazy imports** for optional packages; read credentials from `.env` via `python-dotenv`, never hardcode.
- `.env` is secret-bearing and gitignored — never commit it; read via `core.config.get()`.

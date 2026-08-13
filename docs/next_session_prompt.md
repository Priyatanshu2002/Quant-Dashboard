# Handoff Prompt — Next Chat Session
## Project: Agonistes (Automated Quant Trading System)
### Task: Neo4j Writer + Qdrant Writer + Docker/Monitoring Gaps

---

## Context: What This Project Is

**Project Agonistes** is a fully automated, multi-asset quant trading system.
The workspace is at:
`c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice\`

The system architecture (in pipeline order):
1. **Data Ingestion** (`data_ingestion/`) — OHLCV, fundamentals, news, SEC filings
2. **Feature Engineering** (`feature_engineering/`) — builds 75-column feature vectors
3. **Screener** (`screener/`) — scores universe, picks top-N candidates
4. **LangGraph Debate** (`langgraph_app/`) — 9-node adversarial debate pipeline
5. **RL Agent** (`rl_agent/`) — sizes and times trades (PPO)
6. **Portfolio Manager** (`portfolio_manager/`) — risk budgeting
7. **Backtesting** (`backtesting/`) — strategy validation
8. **Strategy Builder** (`strategy_builder/`) — 37-model benchmark (14 neural + 16 classical + 7 vol)
9. **Orchestrator** (`orchestrator.py`) — autonomous daily/weekly/monthly scheduler
10. **REST API** (`core/api_server.py`) — serves the React UI

The primary reference document is `implementation_plan_v2.md` in the project root.

---

## What Is Already Done (Do NOT rebuild these)

- All 9 LangGraph nodes exist: `langgraph_app/src/nodes/node_a_ingestion.py` through `node_i_mirofish.py`
- `docker-compose.yml` already defines **TimescaleDB**, **Qdrant**, **Neo4j** containers (but they have no data)
- `core/db.py` is the unified storage adapter (SQLite dev mode + TimescaleDB production)
- `data_ingestion/fundamental_feeds/` has: `sec_edgar_watcher.py`, `yfinance_financials.py`, `three_statement_parser.py`, `nse_bse_watcher.py`, `dcf_calculator.py`, `dcf_scenarios.py`
- `data_ingestion/sentiment_feeds/` exists with news ingesters
- The orchestrator at `orchestrator.py` already calls `job_ingest_prices()`, `job_ingest_fundamentals()`, `job_ingest_news()`, `job_build_features()`, `job_screen_and_debate()` on schedule
- Docker healthchecks for all three databases are already in `docker-compose.yml`

---

## The 3 Gaps to Solve

### GAP 1 — Neo4j Knowledge Graph Writer

**Problem:** `docker-compose.yml` runs Neo4j (port 7687), but nothing ever writes to it. The graph is permanently empty.

**What node_b_analyst needs from Neo4j (from `node_b_analyst.py`):**
- Entity relationships: Company → Competitor, Company → Supplier, Company → Customer
- Key relationships keyed by symbol: `graphrag_key_relationships(symbol)`
- Management quality signals: CEO tenure, insider ownership, board composition
- Industry/sector classification and peer group

**What to build:**
Create `data_ingestion/graph_feeds/neo4j_writer.py` with:

1. **Schema setup** (`setup_neo4j_schema()`):
   - Node types: `Company`, `Person` (CEO/CFO), `Sector`, `Index`, `ETF`
   - Relationship types: `COMPETES_WITH`, `SUPPLIES_TO`, `PART_OF_SECTOR`, `PART_OF_INDEX`, `CEO_OF`, `CFO_OF`
   - Constraints and indexes on `symbol` and `name`

2. **Company ingestion** (`ingest_company_to_neo4j(symbol, db)`):
   - Pull fundamental snapshot from `db.query_latest_fundamentals(symbol)`
   - Pull company profile (sector, industry, employees, description) from yfinance `Ticker(symbol).info`
   - Write Company node with: `symbol`, `name`, `sector`, `industry`, `market_cap`, `country`

3. **Relationship ingestion** (`ingest_peer_relationships(symbol)`):
   - yfinance `Ticker(symbol).info` has `sector` and `industry` → link to peer companies in the same sector
   - S&P 500 / Nifty 50 membership → `PART_OF_INDEX` relationships
   - For India (`.NS` symbols): NSE sector classification

4. **Management ingestion** (`ingest_management(symbol)`):
   - yfinance `Ticker(symbol).info` fields: `companyOfficers` → CEO/CFO nodes
   - Insider transactions from `db` → enrich Person nodes

5. **Query function** (`query_key_relationships(symbol)`) → returns a plain dict/string that `node_b_analyst.py` can consume directly

6. **Orchestrator hook:** Add `job_ingest_graph()` to `orchestrator.py` — runs weekly (Sunday), after `model_drift_check`. Call it from `ALL_JOBS` registry.

**Neo4j connection:** Use `neo4j` Python driver (already likely in `.venv`). Connection via env var `NEO4J_URI=bolt://localhost:7687` and `NEO4J_PASSWORD` (already in `.env`).

---

### GAP 2 — Qdrant Vector Embedding Pipeline

**Problem:** Qdrant container runs (port 6333), but no collection exists and nothing writes to it. Semantic search over filings, earnings transcripts, and analyst theses returns nothing.

**What the LangGraph nodes need from Qdrant:**
- Semantic similarity search: "find past analyst theses similar to this bull/bear argument"
- Earnings transcript embedding: retrieve relevant historical context for a symbol
- Filing search: "find 10-K passages about supply chain risk for AAPL"

**What to build:**
Create `data_ingestion/vector_feeds/qdrant_writer.py` with:

1. **Collection setup** (`setup_qdrant_collections()`):
   - Collection `filings`: 10-K/10-Q text chunks, metadata: `{symbol, doc_type, period, chunk_index}`
   - Collection `earnings_transcripts`: quarterly call chunks, metadata: `{symbol, quarter, year, speaker}`
   - Collection `analyst_theses`: past debate outputs, metadata: `{symbol, date, side (bull/bear), decision}`
   - Collection `news`: news article embeddings, metadata: `{symbol, source, published_at, sentiment}`
   - Vector dimension: **1536** (OpenAI text-embedding-3-small) with cosine distance
   - Fallback: **384** (sentence-transformers `all-MiniLM-L6-v2`) if no OpenAI key

2. **Embedder** (`embed_text(texts: list[str]) -> list[list[float]]`):
   - Try OpenAI `text-embedding-3-small` first (via `OPENAI_API_KEY` from `.env`)
   - Fall back to `sentence-transformers` locally (`all-MiniLM-L6-v2`, 384-dim)
   - Batch in chunks of 100 to avoid rate limits

3. **Filing ingester** (`ingest_sec_filing(symbol, filing_url, doc_type)`):
   - Download filing text from `sec_edgar_watcher.py` (already exists)
   - Chunk into ~500-token segments with 50-token overlap
   - Embed each chunk → upsert to `filings` collection
   - Use `symbol + doc_type + chunk_index` as point ID (deterministic, idempotent)

4. **Earnings transcript ingester** (`ingest_earnings_transcript(symbol, transcript_text, quarter, year)`):
   - Parse transcript into speaker-turn chunks
   - Embed → upsert to `earnings_transcripts` collection

5. **Thesis archiver** (`archive_debate_thesis(symbol, bull_thesis, bear_thesis, decision, date)`):
   - Called by `node_i_mirofish.py` after every completed debate cycle
   - Embeds bull/bear thesis summaries → upsert to `analyst_theses`
   - Enables future debates to retrieve: "what did the bear say last time about AAPL?"

6. **Query function** (`semantic_search(collection, query_text, symbol_filter, top_k=5)`) → returns list of text snippets for LangGraph nodes to inject into prompts

7. **Orchestrator hook:** Add `job_ingest_vectors()` to `orchestrator.py` — runs weekly alongside `job_ingest_graph()`.

**Qdrant connection:** Use `qdrant-client` Python package. `QDRANT_URL=http://localhost:6333` from `.env`.

---

### GAP 3 — Docker / Monitoring Completeness

**Problem:** `docker-compose.yml` has TimescaleDB + Qdrant + Neo4j but is missing:

1. **Redis** — not in docker-compose at all, but the LangGraph state checkpoint and rate-limiter likely need it
2. **Prometheus scrape config is incomplete** — `monitoring/prometheus/prometheus.yml` only scrapes `agonistes-api:8000` and `timescaledb:9187`, missing Qdrant and Neo4j exporters
3. **Grafana has no dashboards** — `monitoring/grafana/` directory exists but is empty
4. **No `postgres_exporter`** sidecar for TimescaleDB metrics
5. **`start_all.sh`** only starts the API server — doesn't start the orchestrator

**What to build:**

1. **Add Redis to `docker-compose.yml`:**
   ```yaml
   redis:
     image: redis:7-alpine
     container_name: agonistes-redis
     restart: unless-stopped
     ports: ["6379:6379"]
     healthcheck: ...
   ```

2. **Add `postgres_exporter` sidecar** for TimescaleDB Prometheus metrics

3. **Add Qdrant + Neo4j scrape targets** to `monitoring/prometheus/prometheus.yml`

4. **Create 2 Grafana dashboard JSONs** in `monitoring/grafana/dashboards/`:
   - `system_health.json`: container CPU/mem, DB query latency, API response times
   - `trading_activity.json`: daily trades, screener pass rate, gating decisions, circuit breakers, portfolio P&L

5. **Update `monitoring/grafana/` with:**
   - `grafana/provisioning/datasources/prometheus.yml` — auto-provision Prometheus datasource
   - `grafana/provisioning/dashboards/default.yml` — auto-load the dashboard JSONs

6. **Add Grafana service to `docker-compose.yml`:**
   ```yaml
   grafana:
     image: grafana/grafana:latest
     ports: ["3000:3000"]
     volumes:
       - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
       - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
   ```

7. **Update `scripts/start_all.sh`** to:
   - `docker compose up -d` (starts all containers)
   - `python main.py orchestrate &` (starts the autonomous loop)
   - `python main.py serve &` (starts the API)
   - Print the dashboard URLs

8. **Add `QDRANT_URL`, `NEO4J_URI`, `REDIS_URL` to `.env.example`** if not already there

---

## Key Files to Read First

Before writing any code, read these files to understand interfaces and avoid duplication:

1. `core/db.py` — the unified storage adapter (understand how `get_storage()` works)
2. `langgraph_app/src/nodes/node_b_analyst.py` — where Neo4j query results are consumed
3. `langgraph_app/src/nodes/node_i_mirofish.py` — where debate results are persisted (add Qdrant archiving here)
4. `data_ingestion/fundamental_feeds/sec_edgar_watcher.py` — the existing SEC filing fetcher
5. `orchestrator.py` — add your new jobs to `ALL_JOBS` dict here
6. `docker-compose.yml` — extend this, don't replace it
7. `.env` — read which env vars already exist (NEO4J_PASSWORD, etc.)
8. `langgraph_app/src/state.py` — the `DebateState` TypedDict (understand what data flows through)

---

## Constraints and Rules

- **Do not change `core/db.py`** — it's the stable storage interface. Neo4j and Qdrant get their own client wrappers.
- **All writers must be idempotent** — re-running them on the same data should not create duplicates (use upsert semantics everywhere)
- **Graceful degradation** — if Neo4j or Qdrant is down, the LangGraph debate must still run (just without graph/semantic enrichment). Never raise exceptions that crash the debate pipeline.
- **Use the existing `.env` pattern** — read credentials from environment variables via `python-dotenv`, never hardcode
- **Python package imports must be lazy** (inside functions) — so the system still starts if optional packages are missing

---

## Expected Deliverables

1. `data_ingestion/graph_feeds/__init__.py`
2. `data_ingestion/graph_feeds/neo4j_writer.py`
3. `data_ingestion/vector_feeds/__init__.py`
4. `data_ingestion/vector_feeds/qdrant_writer.py`
5. Updated `docker-compose.yml` (Redis + postgres_exporter + Grafana added)
6. Updated `monitoring/prometheus/prometheus.yml` (all services scraped)
7. `monitoring/grafana/provisioning/datasources/prometheus.yml`
8. `monitoring/grafana/provisioning/dashboards/default.yml`
9. `monitoring/grafana/dashboards/system_health.json`
10. `monitoring/grafana/dashboards/trading_activity.json`
11. Updated `orchestrator.py` — `job_ingest_graph()` and `job_ingest_vectors()` added to `ALL_JOBS`
12. Updated `scripts/start_all.sh` — full stack startup
13. Updated `.env.example` — new connection vars documented

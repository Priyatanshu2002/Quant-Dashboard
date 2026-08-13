# Data-Parser Maximization Plan

**Goal:** Maximize the data every ingestion parser captures from its source, map how each field flows into downstream calculations (features, screener, valuation, backtests, ML, LLM/debate), and land it with tests.

**Architecture:** Each source has a dedicated fetcher/parser under `data_ingestion/` that normalizes raw data into the storage layer (`core/db.py`). Downstream consumers are `feature_engineering/` (feature vectors), `screener/`, `valuation/` (CFA/DCF/quality/relative), `backtesting/`, `transformer_model/`+`rl_agent/` (ML), `llm_analyst.py`+debate. Maximization = broaden each parser's field capture, add raw-facts caching where re-parsing is iterative, and keep storage writes idempotent.

**Tech Stack:** Python, requests, pandas, yfinance, SEC EDGAR XBRL API, FRED/BLS/Treasury/CoinGecko/Yahoo/StockTwits/GDELT/Reddit HTTP APIs. SQLite (dev) storage.

**Current state:** The SEC EDGAR parser (the highest-value source) has already been maximized — ~130 US-GAAP concepts, full income/balance/cash-flow, raw-facts cache (`data/sec_facts/`), and a catch-all `extra` capture; the expanded annual backfill is running. yfinance now fetches annual + quarterly statements, a maximised profile, and analyst consensus. Remaining parsers are at varying capture levels; several (FRED/BLS/Dune) are key-gated.

---
## Part A — Parser inventory, capture capacity, relevance & downstream calculations

### A1. Price feeds — `data_ingestion/price_feeds/`
| File | Source | Currently captures | **Max capture capacity** | Relevance | Downstream calculations |
|---|---|---|---|---|---|
| `equity_ws.py` | yfinance OHLCV | OHLCV (daily/1m), asset class | Split-adjusted OHLCV; **dividends & splits (t.dividends/t.splits/t.actions)**, FX-adjusted, volume dollar (OHLC×vol), intraday bars, exchange/currency meta | All price-driven features, backtests, returns, beta | Returns/vol/corr → technical features, `valuation/beta.py`, `backtesting/engine.py`, TFT/TFT features |
| `crypto_ws.py` | Binance klines | OHLCV | + taker buy/sell vol, quote volume, funding rates, open interest | Crypto features, macro crypto | Technical features, on-chain macro |
| `coinbase_ws.py` | Coinbase | tick trades | + product/taker data | Crypto | — |
| `forex_ws.py` | FX rates | spot | + cross rates, forward points | FX for global/IN valuation | Currency normalization in valuation |

**Action:** add splits/dividends/actions capture + dollar-volume + fix `asset_class_for()` (currently garbled ternary → ETF vs EQUITY_US mislabels).

### A2. Fundamental feeds — `data_ingestion/fundamental_feeds/`
| File | Source | Currently captures | **Max capture capacity** | Relevance | Downstream calculations |
|---|---|---|---|---|---|
| `three_statement_parser.py` | SEC EDGAR XBRL | ~130 line items across income/balance/cash-flow + `extra` catch-all + ratios + cache | **Everything SEC tags** (income: rev/COGS/gross/R&D/SG&A/EBIT/D&A/interest/pretax/tax/NI/EPS; balance: full assets/liab/equity incl. ST-invest, prepaid, intangibles, deferred tax, leases, APIC, AOCI, treasury; cash flow: OCF/SBC/capex/acq/divest/debt/dividend/buyback) | CFA statements, ratios, DCF, quality, valuation | `valuation/ratios.py`, `quality.py`, `fcff_model.py`, `cfa_model.py`, `three_statement` UI |
| `yfinance_financials.py` | yfinance | quarterly+annual statements, maximised profile | + segment/revenue breakdown, executives, peer info; every income/balance/CF label | Statements, profiles, ratios | Same as above + `fetch_company_profile` → `company_profiles` |
| `yfinance_earnings.py` | yfinance | earnings dates, EPS act/est, analyst consensus, revenue estimate | + revenue/eps estimate time-series, surprise history, guidance | Earnings features, estimates | `feature_engineering/fundamental_features.py` (earnings_accel), valuation forward metrics |
| `nse_bse_watcher.py` | NSE/BSE | announcements (best-effort) | + parsed announcement type/dates, full Indian 3-statement from NSE/BSE filings | IN equities | event-driven fundamental updates |
| `sec_edgar_watcher.py` | SEC EDGAR | 8-K/10-K filing events + CIK map | + filing type taxonomy, accession, form-date, event classification | event-driven | re-score/re-alert on material filings |
| `dcf_calculator.py`, `dcf_scenarios.py` | (calculation, not fetch) | FCFF DCF, scenarios, sensitivity | n/a (consumer) | Valuation | — |

**Action (done/partial):** SEC maximized; yfinance annual+profile done. Remaining: wire NSE/BSE → statements; capture filings taxonomy; add segment/executive capture.

### A3. Macro feeds — `data_ingestion/macro_feeds/`
| File | Source | Currently captures | **Max capture capacity** | Relevance | Downstream calculations |
|---|---|---|---|---|---|
| `fred_fetcher.py` | FRED | VIX, 2Y/10Y yield, fed funds, DXY, gold (6 series) — **key-gated** | **Hundreds of FRED series**: CPI, PPI, PCE, core inflation, unemployment, NFP, ISM PMI, real yields (TIPS), credit spreads (IG/HY), breakevens, M2, retail sales, housing | Macro features, discount rates, regime | `feature_engineering/macro_features.py`, `valuation/discount_rates.py` (rf, ERP), `backtesting` regime tests |
| `bls_fetcher.py` | BLS | CPI + unemployment — **key-gated** | + PPI, wages, job openings, participation | Macro | macro features |
| `treasury_fetcher.py` | US Treasury (key-free) | 2Y, 10Y + VIX(stooq) | **Full yield curve (1M–30Y)**, real yields (TIPS), breakeven inflation | Risk-free rate for WACC, macro features | `valuation/discount_rates.py`, macro features |
| `yfinance_macro.py` | yfinance (key-free) | VIX, 10Y, 2Y, DXY, gold 5d, BTC | + more indices (CRB, HYG, TLT, QQQ), daily history (not just latest) | Macro features | macro features, regime |

**Action:** extend `treasury_fetcher` to full curve + real yields + breakevens (key-free, highest ROI since no key needed); extend FRED SERIES map so it lights up when the key is set; make yfinance macro capture daily history.

### A4. On-chain — `data_ingestion/onchain_feeds/`
| File | Source | Currently captures | **Max** | Relevance | Downstream |
|---|---|---|---|---|---|
| `dune_fetcher.py` | Dune (key-gated, query IDs TODO) | none (stub) | any custom SQL (exchange flows, stablecoin supply, miner flows) | crypto | crypto macro |
| `exchange_flow_fetcher.py` | CoinGecko (key-free) | BTC dominance, 24h mcap chg, top-10 exchange vol | + fear&greed index, stablecoin mcap, per-coin market data | crypto macro | crypto macro features |

### A5. Sentiment feeds — `data_ingestion/sentiment_feeds/`
| File | Source | Currently captures | **Max** | Relevance | Downstream |
|---|---|---|---|---|---|
| `news_aggregator.py` | Yahoo/Google/RSS/StockTwits | headlines + lexicon score in [-1,1] | + full article text, timestamps, source weight, ticker mentions | sentiment | `sentiment_features.py`, `llm_analyst.py` |
| `gdelt_fetcher.py` | GDELT | article titles → score | + themes, locations, tone, event codes | sentiment/macro | sentiment |
| `reddit_fetcher.py` | Reddit | post titles+selftext → score | + comments, upvote ratios, subreddit | sentiment | sentiment |
| `llm_analyst.py` | LLM (Nous) | NEWS + FUNDAMENTAL verdicts | + structured statement-analysis earnings-direction forecast, RAG over 10-K/10-Q, citations | AI analyst, debate | bull/bear/debate, quality |

**Action:** news_aggregator capture timestamps + store full text; reddit capture comments/score; llm_analyst → structured multi-period statement analysis (the Chicago-Booth pattern) + RAG once vector stack is up.

### A6. Graph/vector writers — `graph_feeds/neo4j_writer.py`, `vector_feeds/qdrant_writer.py`
Both exist but never run (no Docker). Relevance: entity graph + filing embeddings/RAG. Gated on infra.

---
## Part B — Relevance & downstream calculation map
Every captured field is consumed as follows:
1. **Prices →** `feature_engineering/technical_features.py` (returns, RSI, MACD, vol, ~30 → plan target ~75), `valuation/beta.py` (regression beta), `backtesting/engine.py` (returns, drawdown), `transformer_model` features.
2. **Statements (SEC/yfinance) →** `valuation/ratios.py` (activity/liquidity/solvency/profitability/CCC/DuPont), `valuation/quality.py` (Piotroski/Altman Z/Beneish/flags), `valuation/fcff_model.py` (revenue→margin→tax→reinvest→FCFF→TV→equity bridge), `valuation/cfa_model.py` (WACC via `discount_rates.py` + `beta.py`), `feature_engineering/fundamental_features.py`.
3. **Macro →** `discount_rates.py` (rf = 10Y, ERP, country premia), `macro_features.py`, `backtesting` regime tests, WACC.
4. **Sentiment/LLM →** `sentiment_features.py`, `llm_analyst.py` verdicts → screener + debate.
5. **Earnings/estimates →** earnings features, forward P/E, PEG, relative valuation.
6. **Company profile →** screener sector/peer grouping, relative-valuation peer comparison, country ERP lookup.

---
## Part C — Implementation plan (TDD, bite-sized, commit after each task)

### C1. Treasury full curve + real yields + breakevens (key-free, highest ROI)
- **Files:** `data_ingestion/macro_feeds/treasury_fetcher.py`, tests `tests/test_macro.py`
- **Task 1:** extend `COLUMNS` to full curve (1M,2M,3M,6M,1Y,2Y,3Y,5Y,7Y,10Y,20Y,30Y) → `us_1m…us_30y_yield`; write a failing test asserting the fetch returns ≥10 maturities when network available (else skip).
- **Task 2:** add real-yield (TIPS) fetch → `us_10y_real_yield`; **Task 3:** add `breakeven = 10Y nominal − 10Y real`; test. **Task 4:** persist + commit.
- **Verify:** `.venv/Scripts/python scripts/backfill_macro.py` writes a snapshot with `us_10y_yield`, `us_30y_yield`, `us_10y_real_yield`, `breakeven_inflation`; `pytest tests/test_macro.py -v` green.

### C2. FRED + BLS series expansion (lights up when key set)
- **Files:** `fred_fetcher.py` (`SERIES`), `bls_fetcher.py` (`SERIES`), tests
- **Task 1:** add ~30 key FRED series (CPIAUCSL, PPIFIS, PCEPI, UNRATE, PAYEMS, ISMPMI, T10Y2Y, T10YIE, BAMLH0A0HYM2 (HY), BAMLC0A0CM (IG), DGS1/3/5/7/20/30, FEDFUNDS, M2SL, RRSFS, HOUST…). **Task 2:** add transforms + tests (percent→decimal, index values). **Task 3:** BLS add PPI/wages/job-openings. **Task 4:** wire into `job_ingest_macro`. Commit each.

### C3. Equity price feed: corporate actions + dollar volume + asset-class fix
- **Files:** `equity_ws.py`, tests `tests/test_price_feeds.py`
- **Task 1:** fix `asset_class_for()` to a clean mapping (EQUITY_US/ETF/INDEX/EQUITY_IN) with tests (SPY→ETF, ^GSPC→INDEX, ^NSEI→INDEX, AAPL→EQUITY_US, RELIANCE.NS→EQUITY_IN).
- **Task 2:** fetch `t.dividends`, `t.splits`, `t.actions` in `backfill_equities` and persist (new `write_corporate_actions` in `core/db.py`); store `dollar_volume = close×volume`. **Task 3:** add `adjusted` handling (verify against unadjusted for split detection). Commit each.

### C4. NSE/BSE statements + filings taxonomy
- **Files:** `nse_bse_watcher.py`, `sec_edgar_watcher.py`
- **Task 1:** classify NSE/BSE announcements (results, buyback, dividend, board) into event types → persist; test parser on fixture.
- **Task 2:** `sec_edgar_watcher` — map 8-K items/10-K/10-Q to material-event classes; test. Commit.

### C5. Sentiment depth
- **Files:** `news_aggregator.py`, `reddit_fetcher.py`, `gdelt_fetcher.py`
- **Task 1:** store `created_at` timestamp + full text/url on sentiment events; **Task 2:** reddit capture score/num_comments; **Task 3:** GDELT capture themes/tone. Tests for the scoring/storage shape. Commit.

### C6. yfinance segment + executives + profile depth
- **Files:** `yfinance_financials.py`
- **Task 1:** capture `t.segments` (revenue/operating income by segment) + `t.news`; **Task 2:** capture executives/peers from info; persist to `company_profiles`. Test the profile shape. Commit.

### C7. LLM analyst → structured statement analysis + RAG
- **Files:** `llm_analyst.py`, `valuation/` reuse, tests
- **Task 1:** add `analyze_statements(statements)` that feeds normalized multi-period statements + ratios → earnings-direction forecast (up/down/flat) + confidence + narrative (the Chicago-Booth pattern); **Task 2:** persist to `llm_analyses`; test with a fixture. **Task 3 (infra-gated):** RAG over 10-K/10-Q via `qdrant_writer` once embeddings stack is live. Commit.

### C8. Wire new fields into downstream calculations
- **Files:** `feature_engineering/` (technical ~30→~75 target), `valuation/discount_rates.py` (use full curve/real yields/breakevens), `macro_features.py`
- **Task 1:** technical features — add missing indicators (PSAR, Supertrend, Ichimoku, Keltner, MFI, ADL, Aroon, range-vol, skew/kurt) with vectorized tests; **Task 2:** discount_rates uses `us_10y_real_yield`/`breakeven` where available; **Task 3:** macro_features consumes full curve. Commit each.

### C9. Full-suite + live verification
- **Task 1:** `pytest -q` (expect ≥158 + new) green; `ruff check --select E,F data_ingestion/ valuation/` clean.
- **Task 2:** run `job_ingest_prices`, `job_ingest_macro`, `job_ingest_edgar`; verify DB rows gained new fields.
- **Task 3:** rebuild UI (`cd ui && npm run build`), restart server, spot-check Financials/Valuation.

---
## Risks & open questions
- **Key-gated feeds (FRED/BLS/Dune):** can't reach max without API keys; plan marks them as "expand SERIES so they light up when key set" and relies on key-free Treasury/yfinance for now. Ask user if they want to add keys.
- **SEC backfill cost:** expanded parser re-fetches full companyfacts; mitigated by `data/sec_facts/` cache (cheap re-parses).
- **Quarterly vs annual:** SEC 10-Q tagging is sparser than 10-K; quarterly view stays on yfinance to avoid regression (already the design).
- **RAG/vector + Neo4j:** gated on Docker/embeddings — not in scope until infra decision.
- **Rate limits:** SEC ~2 req/s, CoinGecko ~10-30 req/min, Reddit ~60/min — the backfill sleeps are the control.

## Suggested order (if approved)
C1 → C3 → C6 → C5 → C2 → C4 → C7(1–2) → C8 → C9. Highest-value-first (key-free sources that feed valuation/features), infra-gated work (C7-3) deferred.

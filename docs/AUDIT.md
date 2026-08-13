# Project Agonistes — Architecture Audit (up to Phase 2)

> **Date:** 2026-08-14
> **Scope:** Everything built or claimed through Phase 2 (Foundation + Fundamentals), audited against what modern Bloomberg-alternative research terminals (ZinsCapital, OpenBB, AlphaSense, Koyfin, Macrobond) actually do.
> **Status of this doc:** Living. Update as remediation lands. Source of truth for the gap list; the data-parser maximization plan (`2026-08-14_002500-data-parser-maximization.md`) is the execution backlog.

---

## 1. Framing — the core misalignment

The plan (`implementation_plan_v2.md` §1) is a **symbol-first signal-generation pipeline**:

    screener → ~75–120 features → TFT → Bull/Bear debate → RL → PM → backtest

The modern tools that are winning macro/single-name investors are the **opposite ordering**: they are research instruments built around (a) regime/macro context, (b) data lineage and revision history, and (c) search over qualitative content with citations. The useful question is not "what does Bloomberg have that we lack?" but **"have we built a defensible research foundation before stacking a model/execution layer on top of it?"** — and the answer is no.

| Modern tool | Their centre of gravity | What we are missing relative to them |
|---|---|---|
| **ZinsCapital** | Regime-first; liquidity & credit as first-class inputs; visible/repeatable methodology | We are symbol-first; macro is a minor composite weight; no credit-spread/liquidity intake; no regime pre-filter |
| **OpenBB** | Provider abstraction + bring-your-own-data; AI copilot; instant reports | We hardwire yfinance as the near-universal fallback with no provider seam; no report/export workflow |
| **AlphaSense** | AI-native **search** over earnings calls/filings/broker research with **citations** | We have no full-text/vector search, no transcript ingestion, no citations, no internal-knowledge retrieval |
| **Macrobond** | Institutional time-series with **revision history + governed lineage** | We store "current" values with no source/rev/as-of attribution; no diffable history |
| **Koyfin** | Custom dashboards, portfolio analytics, **client-ready reports** | Our UI is a dashboard, not a research/publishing workbench |

---

## 2. Confirmed issues (each verified against the repo)

### I1 — Symbol-first, not regime-first
The screener scores the universe top-down (`top_n_selector`); macro is only a 10–50% weight inside `AssetSignal.composite_score`. Regime is checked **late**, at LLM gating (`node_e_gating` VIX check), after a symbol is already selected. ZinsCapital's whole thesis is regime → liquidity/credit → cross-asset confirmation → then drill down. Our funnel is upside down.

### I2 — The regime inputs are the least-built layer
- FRED and BLS fetchers are **key-gated** (`fred_fetcher.py`, `bls_fetcher.py`).
- Treasury fetcher returns only 2Y/10Y — **no full curve, no TIPS real yields, no breakevens** (this is the key-free C1 work in the parser plan).
- **No credit spreads (IG/HY) and no liquidity (M2/reserves)** in the live path at all.
- `yfinance_macro` is used as the keyless fallback — a 3-series snapshot, not daily history.

### I3 — No data lineage / provenance / revision tracking
`core/db.py` exposes clean `write_*`/`query_*` methods, but every stored value is "current": **no source attribution, no parse revision, no change history.** For CFA-grade valuation (DCF/ratios/quality) this is a credibility gap — data revisions move inputs, and you cannot defend a number you cannot provenance. This is Macrobond's entire product.

### I4 — No search / retrieval layer over the corpus
There is **no full-text index and no vector index** in the codebase (SQLite FTS5 is available and un-used; Qdrant writer is "pending", gated on Docker which isn't installed). AlphaSense's edge — intent search over earnings calls/filings/news with citations — is entirely absent. We compute scores but cannot answer "where does this company say X?"

### I5 — Earnings-call transcript sentiment is unimplemented
Phase 2's checklist includes "earnings call transcript LLM sentiment scoring (OpenRouter)". **No transcript ingestion module exists** (searched the repo; nothing). Phase 2 was declared "done" using news sentiment as the cutoff — a proxy for the phase's actual goal.

### I6 — "Done" was declared on proxies
The old README marked Phases 1, 2, 4, 6, 7 as ✅. Verification shows:
- **Phase 1:** Docker stack never set up (Docker not installed); backfill is 31 symbols, not the full universe; no regime layer.
- **Phase 2:** carried by SEC parser + news sentiment; transcripts/NSE/BSE/FRED/BLS/GDELT/Reddit incomplete or gated.
- **Phase 4:** real LangGraph cycles run, **but Node B attaches TFT only when a trained checkpoint exists** — otherwise `NEUTRAL/0.0`. Debates run on an empty ML signal. No RAG/citations.
- **Phase 6:** engine complete but **has only run hand-written strategies** (`ma_cross`, `shuffled`, …), never the system's actual ML/debate outputs.
- **Phase 7:** React UI builds; **Grafana/Prometheus are config-only, not running**; UI is a dashboard, not a research workbench.

### I7 — Fragile single-source coupling (no provider abstraction)
Almost everything falls back to **yfinance** (macro keyless, profile, financials, earnings). yfinance is reverse-engineered with no SLA. OpenBB's architectural centerpiece is a provider-agnostic layer + BYO data. We have a *storage* adapter (SQLite/Timescale) but **no *data-provider* abstraction** — one scrape-API hiccup takes down screen + valuation + macro together.

### I8 — The "agentic OS" / unified-system vision depends on infra that isn't running
The intended storage/search/knowledge plane (TimescaleDB, Qdrant, Neo4j, Redis) exists only as docker-compose config. **Docker is not installed**, so none of it runs. Anything depending on time-series hypertables, vectors, or the knowledge graph cannot execute locally.

### I9 — No persistent research/knowledge layer
The daily reflection agent feeds trade *lessons* back into the debate, but there is no accumulating, searchable research object per company (theses, notes, prior LLM analyses, RAG over our own verdicts). AlphaSense and ZinsCapital treat accumulated internal analysis as a first-class, retrievable asset; we treat it as transient state.

---

## 3. What we are doing right (keep)
- Correctly prioritising the **SEC EDGAR parser** (~130 GAAP concepts + raw-facts cache + catch-all `extra`) and deferring heavy training until data+integration is solid. That matches the tools' data-quality-first priority.
- The parser plan's Part B (field → downstream calculation map) is exactly the lineage **thinking** we need — it just isn't persisted to storage.
- **Single-origin unified UI** — right call, not scattered bolt-on pages.
- Clean `core/` storage adapter and an honest, expanding test suite.

---

## 4. Prioritised remediation (cheapest → highest leverage)

| # | Work item | Why | Fits plan |
|---|---|---|---|
| **R1** | **Persist data lineage**: add `source`, `as_of`, `rev` to storage + a `data_revisions` diff table; record on every SEC/yfinance write | Cheap, key-free, underpins valuation credibility | Extends parser plan Part B into storage |
| **R2** | **Macro regime classifier** (`regime.py`): full curve + real yields + breakevens + credit spreads + liquidity → regime label that *pre-filters* the universe before scoring | Re-orders funnel to match ZinsCapital; highest strategic ROI | New; feeds screener + gating |
| **R3** | **Treasury full curve + TIPS + breakevens** (key-free) | Unlocks R2 and discount_rates | Parser plan **C1** |
| **R4** | **Search layer**: SQLite FTS5 now over filings+news+own analyses; Qdrant later | AlphaSense-style intent search with citations; the biggest qualitative gap | Parser plan C5/C7 + infra |
| **R5** | **Data-provider seam** so yfinance is one swappable provider, not the single point of failure | OpenBB's core; de-risks the whole loop | Refactor |
| **R6** | **Transcript ingestion + LLM sentiment** | Completes the real Phase-2 checklist item | Parser plan **C7** (statement analysis) + new |
| **R7** | **Honest phase gate**: don't mark a phase done until its full checklist is real (or rename scope explicitly) | Stops the proxy-done pattern | Process |

**Explicitly deferred (infra-gated):** TimescaleDB/Qdrant/Neo4j/Redis runtime (no Docker), graph writers, vector RAG, Grafana runtime, monitoring — until a decision on infra/keys (see parser plan Risks & open questions).

---

## 5. Recommended build order (ties into the data-parser plan)
Follow the parser plan's suggested order (C1 → C3 → C6 → C5 → C2 → C4 → C7 → C8 → C9) **and** fold in this audit's highest-value items:
1. **R1 lineage** (small, unblocks credibility) — do alongside parser work as writes land.
2. **R3 Treasury curve** (C1) then **R2 regime classifier**.
3. **R4 search (FTS5)** as sentiment/filings depth (C5/C7) grows.
4. **R6 transcripts** with C7's structured statement analysis.
5. **R5 provider seam** as a refactor once the data path stabilises.
6. **R7** continuously — never mark done on a proxy again.

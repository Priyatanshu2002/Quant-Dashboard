# Project Agonistes — Visual System Map

![System Architecture](agonistes_architecture_1786592383348.png)

---

## 1. The Big Picture — What the System Does Daily

```mermaid
flowchart TD
    A(["🌍 World Markets Open"]) --> B

    subgraph INGEST ["📥 DATA INGESTION  09:15 IST"]
        B["Price Feeds\nyfinance · Binance"] 
        C["Fundamental Feeds\nSEC EDGAR · NSE · yfinance"]
        D["Sentiment Feeds\nNews · Reddit · GDELT"]
        E["Macro Feeds\nVIX · Fed Rate · DXY"]
    end

    subgraph STORES ["🗄️ DATABASES"]
        TS[("TimescaleDB\nOHLCV + Features")]
        N4[("Neo4j\nKnowledge Graph\n⚠️ writers pending")]
        QD[("Qdrant\nVector Embeddings\n⚠️ writers pending")]
    end

    B & C & D & E --> TS
    C --> N4
    C & D --> QD

    TS --> FE

    subgraph FE ["⚙️ FEATURE ENGINEERING  10:00 IST"]
        FE1["75 features per asset\nRSI · MACD · BB · ATR\nDCF MoS · EPS growth\nSentiment · VIX regime\nMomentum · Seasonality"]
    end

    FE --> SC

    subgraph SC ["🔍 SCREENER  10:15 IST"]
        SC1["Score entire universe\n30+ US equities\n15 Indian stocks\nETFs · FX · Gold · Crypto"]
        SC2["Multi-factor composite score\nTechnical 30%\nFundamental 25%\nSentiment 20%\nMacro 15%\nMomentum 10%"]
        SC1 --> SC2
    end

    SC --> DEB

    subgraph DEB ["⚖️ LANGGRAPH DEBATE  10:30 IST"]
        D1["A: Ingest pack"]
        D2["B: Analyst\nTFT signal attach"]
        D3["C: Bull agent\nLLM long thesis"]
        D4["D: Bear agent\nLLM short thesis"]
        D5["E: Gating\nconfidence delta\nVIX regime check"]
        D6["F: Portfolio\nRL position size"]
        D7["G: Reflection\nself-critique"]
        D8["H: Execution\norder params"]
        D9["I: Circuit breaker\nfinal risk gate"]
        D1-->D2-->D3-->D4-->D5-->D6-->D7-->D8-->D9
    end

    DEB --> OUT

    subgraph OUT ["📤 OUTPUTS"]
        O1["📋 Trade Log\n(paper trades with rationale)"]
        O2["📊 React UI Dashboard"]
        O3["🖥️ Grafana Monitoring"]
        O4["🔔 Alerts\n(Telegram/email — pending)"]
    end
```

---

## 2. The LangGraph Debate — Node by Node

```mermaid
flowchart LR
    subgraph DEBATE ["LangGraph 9-Node Adversarial Pipeline"]
        A["A\nIngestion\n──────\nPulls all data\nfor this symbol\ninto AnalystPack"]
        B["B\nAnalyst\n──────\nAttaches TFT\ndirection signal\n+ conviction"]
        C["C\nBull Agent\n──────\nLLM constructs\nstrongest LONG\ncase with numbers"]
        D["D\nBear Agent\n──────\nLLM constructs\nstrongest SHORT\ncase adversarially"]
        E["E\nGating\n──────\nComputes confidence\ndelta. Checks VIX\nregime + TFT align"]
        F["F\nPortfolio\n──────\nRL agent sizes\nposition. Applies\nvol targeting"]
        G["G\nReflection\n──────\nSelf-critique:\nwhat could be\nwrong?"]
        H["H\nExecution\n──────\nBuilds order:\nnotional, direction\ntimeframe, SL/TP"]
        I["I\nMiroFish\n──────\nCircuit breaker:\nposition limits\nmax open trades"]

        A-->B-->C-->D-->E
        E-->|"APPROVED\ndelta > threshold"| F
        E-->|"PASS\ndelta < threshold"| SKIP(["⏭ Skip"])
        F-->G-->H-->I
        I-->|"CLEARED"| LOG(["📋 Trade Logged"])
        I-->|"BLOCKED\nlimit hit"| CB(["🚨 Circuit Breaker"])
    end
```

---

## 3. The Orchestrator Schedule — Who Kicks Off What

```mermaid
gantt
    title Orchestrator Weekly Schedule (IST)
    dateFormat  HH:mm
    axisFormat %H:%M

    section Monday–Friday Daily
    Ingest prices           :09:15, 15m
    Ingest fundamentals     :09:30, 5m
    Ingest news/sentiment   :09:35, 25m
    Build feature vectors   :10:00, 30m
    Screen + Debate         :10:30, 60m

    section Sunday (Weekly)
    Model drift check       :02:00, 30m
    Quick benchmark (4 models) :02:30, 90m

    section 1st of Month
    Full benchmark (37 models) :01:00, 240m
    Volatility regime review   :05:00, 60m
```

---

## 4. The Model Layer — What Each Model Does

```mermaid
mindmap
  root((37 Models))
    Neural DL (14)
      TFT — main signal model
      xLSTM sLSTM + mLSTM
      Mamba2 SSM
      PatchTST patches + transformer
      iTransformer feature-as-token
      LPatchTST LSTM denoiser + patch
      PsLSTM patch per channel
      VSN+LSTM feature selection
      VSN+xLSTM
      VSN+Mamba2
      DLinear trend decomp
      NLinear normalize + project
      AR1x autoregressive baseline
      LSTM vanilla baseline
    Classical ML (16)
      Linear
        Ridge L2
        Lasso L1
        ElasticNet L1+L2
        Bayesian Ridge auto-reg
      Tree/Boosting
        Random Forest
        Extra Trees
        sklearn GBM
        XGBoost
        LightGBM
        CatBoost
      Kernel
        SVR RBF
      Instance
        k-NN
      Classification
        Logistic Regression
        LDA
      Hybrid
        HMM + LightGBM regime
        Strategy XGBoost meta
    Volatility (7)
      vol_timing GARCH regime gate
      vol_carry falling vol signal
      vol_momentum vol surprise
      har_signal HAR-RV direction
      vol_regime_ml LightGBM on GARCH+HAR
      vol_timing_egarch leverage effect
      vol_timing_gjr threshold asymmetry
```

---

## 5. What Each Database Stores

| Database | Port | What's in it | Who writes | Who reads |
|---|---|---|---|---|
| **TimescaleDB** | 5432 | OHLCV, feature vectors, trade log, gating log, fundamentals, sentiment, circuit breakers | All ingesters + orchestrator | Everything — primary store |
| **Neo4j** | 7687 | Company→Sector→Peer relationships, CEO/CFO nodes, S&P 500 / Nifty membership | ⚠️ `neo4j_writer.py` (pending) | `node_b_analyst.py` — enriches debate context |
| **Qdrant** | 6333 | Embedded: 10-K filings, earnings transcripts, past debate theses, news | ⚠️ `qdrant_writer.py` (pending) | `node_c_bull.py`, `node_d_bear.py` — semantic retrieval |
| **Redis** | 6379 | LangGraph state checkpoints, API rate limiter, short-term cache | LangGraph runtime | LangGraph runtime |

---

## 6. What Files Map to What Functions

```
orchestrator.py              ← THE BRAIN — runs everything on schedule
│
├── main.py                  ← CLI: backfill / features / screen / backtest / orchestrate
│
├── data_ingestion/
│   ├── price_feeds/         ← OHLCV from yfinance + Binance
│   ├── fundamental_feeds/   ← DCF, SEC filings, earnings, NSE/BSE
│   ├── sentiment_feeds/     ← News, Reddit, GDELT
│   ├── macro_feeds/         ← VIX, Fed rate, DXY, yield curve
│   ├── graph_feeds/         ← ⚠️ Neo4j writer (pending)
│   └── vector_feeds/        ← ⚠️ Qdrant writer (pending)
│
├── feature_engineering/     ← 75 features → TimescaleDB
│
├── screener/                ← Multi-factor scoring, top-N selection
│
├── langgraph_app/
│   └── src/nodes/           ← 9 nodes: A B C D E F G H I
│
├── transformer_model/       ← TFT architecture + inference (⚠️ untrained)
│
├── rl_agent/                ← PPO position sizer (⚠️ untrained)
│
├── strategy_builder/
│   ├── models.py            ← 14 neural encoders
│   ├── classical.py         ← 16 classical ML models
│   ├── volatility_models.py ← 7 vol models
│   ├── trainer.py           ← Pooled-Sharpe walk-forward training
│   ├── backtest.py          ← Oxford protocol metrics
│   └── run.py               ← Benchmark runner (37 models)
│
├── backtesting/             ← Strategy backtesting engine + regime tests
│
├── portfolio_manager/       ← Risk budgeting, position limits
│
├── core/
│   ├── db.py                ← Unified storage adapter (SQLite dev / TimescaleDB prod)
│   ├── api_server.py        ← REST API for React UI (9 endpoints)
│   └── logging.py           ← Structured logging
│
├── ui/                      ← React dashboard
│
├── monitoring/
│   ├── prometheus/          ← Metrics scraping config
│   └── grafana/             ← ⚠️ Dashboards (pending)
│
└── docker-compose.yml       ← TimescaleDB + Neo4j + Qdrant + (⚠️ Redis + Grafana pending)
```

---

## 7. Completion Status at a Glance

```mermaid
pie title Project Completion
    "✅ Done & wired" : 78
    "⚠️ Code exists, untrained (TFT + PPO)" : 10
    "❌ Still to build (Neo4j + Qdrant writers, Grafana, Alerts)" : 12
```

| Layer | Status | Blocker |
|---|---|---|
| Data ingestion (price, fundamental, sentiment, macro) | ✅ | — |
| Feature engineering (75 features) | ✅ | — |
| Screener | ✅ | — |
| LangGraph (all 9 nodes) | ✅ | — |
| RL Agent (PPO code) | ✅ code / ❌ trained | Needs GPU training |
| TFT model (code) | ✅ code / ❌ trained | Needs GPU training |
| Strategy builder (37 models) | ✅ | — |
| Backtesting engine | ✅ | — |
| REST API (9 endpoints) | ✅ | — |
| Orchestrator (daily/weekly/monthly) | ✅ | — |
| Docker compose (TS + Neo4j + Qdrant) | ✅ | — |
| **Neo4j writer pipeline** | ❌ | Next sprint |
| **Qdrant embedding pipeline** | ❌ | Next sprint |
| **Redis in Docker** | ❌ | Next sprint |
| **Grafana dashboards** | ❌ | Next sprint |
| **Alert system (Telegram)** | ❌ | Future |
| **RunPod API automation** | ❌ | Future |

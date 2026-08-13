# Implementation Plan — Intraday Layer + Multi-Timeframe Backtesting

## Goal

Add a fully parallel intraday system that runs **alongside** the existing swing system.
Both systems share the same databases, models, and orchestrator — but operate on
different timeframes, features, and signal generation paths.

> [!IMPORTANT]
> The existing swing pipeline (daily bars, LangGraph debate, RL swing sizer) is **not
> changed**. The intraday layer is additive. Both run simultaneously.

---

## How the Two Systems Coexist

```
SWING SYSTEM (existing, unchanged)       INTRADAY SYSTEM (new, parallel)
────────────────────────────────         ──────────────────────────────────
Timeframe: Daily bars                    Timeframe: 1-min / 5-min bars
Runs: Once daily 09:15–10:30 IST         Runs: Every 5 min, market hours only
Signal: LangGraph 9-node debate          Signal: Direct model inference (~50ms)
Features: 75 daily features              Features: 25 intraday features
RL agent: PPO swing sizer                RL agent: PPO intraday sizer (separate)
Output: Swing paper trades               Output: Intraday paper trades
Backtest: Daily engine (existing)        Backtest: Intraday engine (new)
Strategy builder: 37 models on daily     Strategy builder: same 37 models on 5-min
```

The two systems share:
- TimescaleDB (separate tables: `ohlcv_1min`, `intraday_features`, `intraday_trades`)
- The trained model weights (TFT/xLSTM inference runs on both timeframes)
- The strategy builder benchmark infrastructure (same 37 models, different data)
- The orchestrator (4th loop: intraday, runs during market hours)
- The REST API + UI (new intraday endpoints)

---

## Proposed Changes

---

### Component 1 — Intraday Data Ingestion

#### [MODIFY] [equity_ws.py](file:///c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice\data_ingestion\price_feeds\equity_ws.py)

The file already has `interval` parameter and an event bus. Extend:
- `backfill_intraday(symbols, interval="5m", days=5)` — pull recent 5-min history
- `stream_intraday(symbols, interval="1m", poll_every=30)` — async loop that polls
  yfinance every 30 seconds during market hours, publishes new closed bars on
  `EVENT_PRICE_BAR` bus with `timeframe="1m"`
- Market hours guard: only poll 09:15–15:30 IST (NSE) or 09:30–16:00 ET (NYSE)

**Data source:** yfinance `download(interval="1m")` gives last 7 days free.
For longer intraday history (backtesting): yfinance `interval="5m"` gives 60 days.

#### [NEW] `databases/timescaledb/init/002_intraday_schema.sql`

```sql
-- 1-minute OHLCV (hypertable, chunk by 1 day)
CREATE TABLE ohlcv_1min (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    vwap        DOUBLE PRECISION,
    PRIMARY KEY (time, symbol)
);
SELECT create_hypertable('ohlcv_1min', 'time', chunk_time_interval => INTERVAL '1 day');

-- Intraday feature vectors
CREATE TABLE intraday_features (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    -- 25 intraday features (see feature list below)
    rsi_14_5m   DOUBLE PRECISION, macd_hist_5m DOUBLE PRECISION,
    vwap_pct    DOUBLE PRECISION, volume_ratio DOUBLE PRECISION,
    order_flow_imbalance DOUBLE PRECISION,
    -- ... (full list in feature engineering section)
    PRIMARY KEY (time, symbol)
);

-- Intraday paper trades
CREATE TABLE intraday_trades (
    trade_id    TEXT PRIMARY KEY,
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT,
    direction   TEXT,
    entry_price DOUBLE PRECISION,
    exit_price  DOUBLE PRECISION,
    holding_mins INTEGER,
    pnl_pct     DOUBLE PRECISION,
    signal_source TEXT,   -- which model generated the signal
    timeframe   TEXT DEFAULT '5m'
);
```

---

### Component 2 — Intraday Feature Engineering

#### [NEW] `feature_engineering/intraday_features.py`

25 intraday-appropriate features (replacing the 75 daily features):

**Price/momentum (8 features):**
- `ret_1bar`, `ret_3bar`, `ret_12bar` — 1/3/12 bar returns (5/15/60 min)
- `rsi_14_5m` — RSI(14) on 5-min closes
- `macd_hist_5m` — MACD histogram on 5-min
- `bb_pct_b_5m` — Bollinger %B on 5-min
- `price_vs_open_pct` — current price vs today's open
- `overnight_gap_pct` — today's open vs yesterday's close

**Volume / microstructure (7 features):**
- `volume_ratio` — current bar volume / 20-bar average volume
- `vwap_pct` — price vs intraday VWAP (key intraday anchor)
- `volume_z_score_1h` — volume z-score over last 60 bars
- `order_flow_imbalance` — (buy_vol - sell_vol) / total_vol (estimated from close vs midpoint)
- `trade_intensity` — bars with above-avg volume in last 12 bars
- `intraday_range_pct` — (high - low) / open for today so far
- `vwap_slope` — slope of VWAP over last 12 bars

**Regime / context (5 features):**
- `session_progress` — 0.0 (open) to 1.0 (close) — position in trading day
- `vix_intraday` — VIX at time of signal (refreshed every 5 min)
- `market_breadth` — % of index constituents above their VWAP
- `prev_day_ret` — yesterday's return (context)
- `time_to_event` — bars until next earnings/dividend/FOMC (from daily calendar)

**Volatility (5 features):**
- `realized_vol_30m` — 30-minute realized volatility
- `realized_vol_1h` — 1-hour realized volatility
- `vol_ratio` — 30m vol / 1h vol (vol acceleration)
- `atr_5m` — ATR(14) on 5-min bars
- `garch_intraday` — GARCH(1,1) fitted on 5-min returns (daily refit)

**Build function:** `build_intraday_features(symbol, bars_5m: pd.DataFrame) -> pd.DataFrame`
- Input: last 60+ bars of 5-min OHLCV
- Output: single-row feature vector for the current bar
- Designed to be called in real-time every 5 minutes

---

### Component 3 — Fast Signal Path (No LLM)

#### [NEW] `langgraph_app/src/fast_path.py`

```python
def intraday_signal(symbol: str, features: pd.Series,
                    model_name: str = None) -> IntradaySignal:
    """
    Direct model inference — no LLM, no debate, no Neo4j/Qdrant.
    Latency target: < 100ms per symbol.

    Uses whichever model is set in .env SIGNAL_MODEL (or best from benchmark).
    Returns: direction (LONG/SHORT/FLAT), conviction (0-1), entry_params.
    """
```

This is the intraday equivalent of the full LangGraph debate. It:
1. Loads the trained model (TFT or benchmark winner, same weights)
2. Runs inference on the 25 intraday features
3. Applies a simple threshold gate (conviction > 0.6 → signal, else FLAT)
4. Checks a fast risk gate: is intraday drawdown > 1.5% today? → halt
5. Returns `IntradaySignal(direction, conviction, suggested_hold_bars)`

**No LLM involved.** This is pure model inference.

Optional enrichment (if available and latency allows): pull GARCH intraday vol from Qdrant or cache — use to scale position size.

---

### Component 4 — Intraday Orchestrator Loop

#### [MODIFY] [orchestrator.py](file:///c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice\orchestrator.py)

Add a **4th loop**: `_intraday_loop()` — runs every 5 minutes during market hours.

```
INTRADAY LOOP (new):
  ├── Market hours check (09:20-15:25 IST or 09:35-15:55 ET)
  ├── For each symbol in screener top-20:
  │     ├── Fetch latest 5-min bars (yfinance, last 60 bars)
  │     ├── Compute 25 intraday features
  │     ├── Call fast_path.intraday_signal()
  │     ├── If signal != FLAT and conviction > threshold:
  │     │     ├── Log to intraday_trades table
  │     │     └── Publish to EVENT_BUS → UI
  │     └── Check exit conditions for open intraday positions
  └── Repeat every 5 minutes
```

New jobs registered in `ALL_JOBS`:
- `intraday_signal_sweep` — runs every 5 min during market hours
- `intraday_backfill` — runs once at 09:15 IST, fetches last 7 days of 1-min data

---

### Component 5 — Multi-Timeframe Backtesting Engine

#### [MODIFY] [backtesting/engine.py](file:///c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice\backtesting\engine.py)

The current engine says `"Execution model (daily bars)"` in its docstring and hardcodes daily fill logic. Generalize it:

- Add `timeframe: str = "1d"` parameter to `BacktestEngine.run()`
- Add `bar_duration_mins: int` derived from timeframe (`"1d"→390`, `"5m"→5`, `"1m"→1`)
- Execution model: when signal changes → fill at **next bar open** (same logic, different bar size)
- Commission/slippage: scale by `sqrt(bar_duration_mins / 390)` — shorter bars = more trades = more costs
- Add `holding_bars` to `Trade` dataclass (alongside existing `holding_hours`)
- Annualization: use `252 * (390 / bar_duration_mins)` bars per year for Sharpe calculation

#### [MODIFY] [backtesting/strategies.py](file:///c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice\backtesting\strategies.py)

Add intraday-appropriate strategies:
- `VWAPMomentum` — long when price crosses above VWAP with volume surge
- `OpeningRangeBreakout` — long/short when price breaks first 30-min range
- `IntraVolCarry` — based on vol_carry signal from `volatility_models.py`
- `MeanReversionBB` — fade extreme Bollinger moves on 5-min bars

#### [NEW] `backtesting/intraday_backtest.py`

Convenience wrapper:
```python
def backtest_intraday(
    symbol: str,
    strategy_name: str,
    interval: str = "5m",
    days_history: int = 60,
) -> BacktestResult:
    """
    End-to-end intraday backtest:
      1. Fetch 5-min bars (yfinance, up to 60 days)
      2. Compute intraday features
      3. Generate signals with the strategy
      4. Run BacktestEngine with timeframe="5m"
      5. Return full BacktestResult + report
    """
```

#### [MODIFY] [main.py](file:///c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice\main.py)

Add `backtest-intraday` subcommand:
```bash
python main.py backtest-intraday --symbol AAPL --strategy vwap_momentum --interval 5m
python main.py backtest-intraday --symbol RELIANCE.NS --strategy orb --interval 1m
```

---

### Component 6 — Strategy Builder on Intraday Data

#### [MODIFY] [strategy_builder/run.py](file:///c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice\strategy_builder\run.py)

Add `--timeframe` flag:
```bash
# Existing (daily)
python -m strategy_builder.run --all --seeds 3

# New (intraday)
python -m strategy_builder.run --all --seeds 3 --timeframe 5m
python -m strategy_builder.run --models lightgbm xlstm vol_timing --timeframe 5m --quick
```

When `--timeframe 5m`:
- Load 5-min OHLCV from `ohlcv_1min` table (aggregated to 5-min)
- Use `build_intraday_features()` instead of `build_features()`
- Walk-forward windows: train=30d, test=5d (instead of 36m/6m)
- Same 37 models, same Sharpe loss, same registry — just different data

This answers the question: **"Do the same models that work on daily also work on 5-min?"** — empirically, on your data.

#### [MODIFY] [strategy_builder/features.py](file:///c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice\strategy_builder\features.py)

Add `build_intraday_universe_frame(prices_5m: pd.DataFrame) -> pd.DataFrame` — same interface as `build_universe_frame` but uses intraday features and 5-min bars.

---

### Component 7 — REST API + UI Intraday Endpoints

#### [MODIFY] [core/api_server.py](file:///c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice\core\api_server.py)

Add:
- `GET /api/intraday/signals?symbol=` — current intraday signal + conviction
- `GET /api/intraday/trades?symbol=&date=` — intraday trade log for a day
- `GET /api/intraday/equity?symbol=&date=` — intraday equity curve (5-min P&L)
- `GET /api/backtest/intraday?symbol=&strategy=&interval=` — run intraday backtest on demand

---

## Open Questions

> [!IMPORTANT]
> **1. Which intraday data source?**
> yfinance gives free 1-min data for last 7 days and 5-min for last 60 days.
> For longer intraday backtest history (1+ year), a paid source is needed:
> - Polygon.io (~$29/month) — US equities, 1-min, 2+ years
> - Zerodha Kite API — Indian equities, 1-min, free with account
> - Alpha Vantage free tier — 5 calls/min, limited
>
> **Decision needed: US only (free via yfinance 60d), or pay for longer history?**

> [!IMPORTANT]
> **2. Intraday signal universe — how many symbols?**
> Running 5-min signals on 50+ symbols every 5 minutes = 10 inference calls/minute.
> Suggested: top-20 from the daily screener (already computed at 10:15 IST).
> Should intraday track a different, smaller universe?

> [!IMPORTANT]
> **3. Separate RL agent for intraday?**
> Training a separate PPO on 5-min bars requires ~60 days of 1-min data minimum.
> Until trained, the fast path uses a simple threshold gate (conviction > 0.6).
> Should we add an intraday PPO to the RunPod training queue alongside the swing PPO?

---

## Verification Plan

### Automated Tests
```bash
# Test intraday feature computation
python -m pytest tests/test_intraday_features.py -v

# Test fast signal path
python -m pytest tests/test_fast_path.py -v

# Test intraday backtesting engine
python -m pytest tests/test_intraday_backtest.py -v

# Smoke test: full intraday loop on 1 symbol
python orchestrator.py --now intraday_signal_sweep
```

### Manual Verification
1. Run `python main.py backtest-intraday --symbol AAPL --strategy vwap_momentum --interval 5m` → confirm metrics look reasonable (Sharpe > 0.5 in-sample)
2. Run `python -m strategy_builder.run --models lightgbm xlstm --timeframe 5m --quick` → confirm benchmark runs end-to-end on intraday data
3. Check `GET /api/intraday/signals?symbol=AAPL` returns a response during market hours

---

## File Summary

| File | Action | Notes |
|---|---|---|
| `data_ingestion/price_feeds/equity_ws.py` | MODIFY | Add `backfill_intraday()` + `stream_intraday()` |
| `databases/timescaledb/init/002_intraday_schema.sql` | NEW | 1-min OHLCV + intraday features + intraday trades tables |
| `feature_engineering/intraday_features.py` | NEW | 25 intraday features |
| `langgraph_app/src/fast_path.py` | NEW | Direct inference, no LLM, < 100ms |
| `orchestrator.py` | MODIFY | 4th loop: `_intraday_loop()` every 5 min |
| `backtesting/engine.py` | MODIFY | Add `timeframe` param, generalize bar logic |
| `backtesting/strategies.py` | MODIFY | Add VWAP Momentum, ORB, MeanRevBB |
| `backtesting/intraday_backtest.py` | NEW | Convenience wrapper for intraday backtest |
| `main.py` | MODIFY | Add `backtest-intraday` subcommand |
| `strategy_builder/run.py` | MODIFY | Add `--timeframe` flag |
| `strategy_builder/features.py` | MODIFY | Add `build_intraday_universe_frame()` |
| `core/api_server.py` | MODIFY | Add 4 intraday endpoints |
| `core/db.py` | MODIFY | Add `query_ohlcv_1min()`, `write_intraday_trade()` |

-- Project Agonistes — TimescaleDB bootstrap schema
-- Loaded automatically into the `agonistes` DB on first container start.
-- NOTE: the storage layer in core/db.py mirrors these tables on SQLite in
-- dev mode, so the same code runs with or without the Docker stack.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ─────────────────────────────────────────────────────────────────────
-- 1. market_data — OHLCV for all assets (10-year retention)
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_data (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    source      TEXT NOT NULL,           -- BINANCE, YAHOO, FRANKFURTER, ...
    interval    TEXT NOT NULL,           -- 1m, 5m, 1h, 1d, ...
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    raw         JSONB
);
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_md_symbol ON market_data (symbol, interval, time DESC);

-- ─────────────────────────────────────────────────────────────────────
-- 2. feature_vectors — ML feature store + target labels
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feature_vectors (
    time                TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL,
    asset_class         TEXT NOT NULL,
    timeframe           TEXT NOT NULL,           -- INTRADAY, SWING, LONGTERM

    -- Technical (~30)
    rsi_14              DOUBLE PRECISION,
    rsi_7               DOUBLE PRECISION,
    macd_histogram      DOUBLE PRECISION,
    macd_signal_val     DOUBLE PRECISION,
    stoch_k             DOUBLE PRECISION,
    bb_pct_b            DOUBLE PRECISION,
    bb_width            DOUBLE PRECISION,
    atr_pct             DOUBLE PRECISION,
    adx_14              DOUBLE PRECISION,
    cci_20              DOUBLE PRECISION,
    volume_z_score      DOUBLE PRECISION,
    vwap_pct            DOUBLE PRECISION,
    price_vs_ema200_pct DOUBLE PRECISION,
    realized_vol_20     DOUBLE PRECISION,
    return_1bar         DOUBLE PRECISION,
    return_5bar         DOUBLE PRECISION,
    return_20bar        DOUBLE PRECISION,
    return_60bar        DOUBLE PRECISION,

    -- Fundamental (~20)
    eps_surprise_pct    DOUBLE PRECISION,
    eps_yoy_growth      DOUBLE PRECISION,
    revenue_yoy_growth  DOUBLE PRECISION,
    dcf_margin_of_safety DOUBLE PRECISION,
    forward_pe          DOUBLE PRECISION,
    peg_ratio           DOUBLE PRECISION,
    fcf_yield           DOUBLE PRECISION,
    roic                DOUBLE PRECISION,
    ebitda_margin       DOUBLE PRECISION,
    debt_to_equity      DOUBLE PRECISION,
    insider_buy_sell_ratio DOUBLE PRECISION,
    inst_ownership_change DOUBLE PRECISION,
    earnings_call_sentiment DOUBLE PRECISION,
    margin_trend        DOUBLE PRECISION,
    revenue_accel       DOUBLE PRECISION,

    -- Sentiment (~10)
    sentiment_score     DOUBLE PRECISION,
    sentiment_momentum  DOUBLE PRECISION,
    sentiment_volume    INTEGER,
    reddit_sentiment    DOUBLE PRECISION,
    gdelt_sentiment     DOUBLE PRECISION,
    news_sentiment      DOUBLE PRECISION,
    sentiment_extreme   BOOLEAN,

    -- Macro (~10)
    vix                 DOUBLE PRECISION,
    vix_regime          INTEGER,
    yield_curve_spread  DOUBLE PRECISION,
    fed_funds_rate      DOUBLE PRECISION,
    btc_dominance       DOUBLE PRECISION,
    dxy                 DOUBLE PRECISION,

    -- Calendar (~5)
    days_to_earnings    INTEGER,
    earnings_week       BOOLEAN,
    days_to_expiry      INTEGER,
    day_of_week         INTEGER,
    month_end_effect    BOOLEAN,
    quarter_end_effect  BOOLEAN,

    -- Target labels (added retrospectively for supervised training)
    future_return_1d    DOUBLE PRECISION,
    future_return_5d    DOUBLE PRECISION,
    future_return_20d   DOUBLE PRECISION,
    future_sharpe_5d    DOUBLE PRECISION,

    extra_features      JSONB
);
SELECT create_hypertable('feature_vectors', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_fv_symbol ON feature_vectors (symbol, timeframe, time DESC);

-- ─────────────────────────────────────────────────────────────────────
-- 3. fundamental_snapshots — quarterly IS + BS + CF + DCF
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fundamental_snapshots (
    time                    TIMESTAMPTZ NOT NULL,    -- Report/filing date
    symbol                  TEXT NOT NULL,
    asset_class             TEXT NOT NULL,
    period_type             TEXT NOT NULL,           -- ANNUAL, QUARTERLY
    fiscal_year             INTEGER,
    fiscal_quarter          INTEGER,

    -- Income Statement
    revenue                 DOUBLE PRECISION,
    revenue_estimate        DOUBLE PRECISION,
    revenue_yoy_growth      DOUBLE PRECISION,
    gross_profit            DOUBLE PRECISION,
    ebitda                  DOUBLE PRECISION,
    net_income              DOUBLE PRECISION,
    eps_actual              DOUBLE PRECISION,
    eps_estimate            DOUBLE PRECISION,
    eps_yoy_growth          DOUBLE PRECISION,

    -- Balance Sheet
    total_assets            DOUBLE PRECISION,
    total_debt              DOUBLE PRECISION,
    cash_and_equivalents    DOUBLE PRECISION,
    net_debt                DOUBLE PRECISION,
    shareholders_equity     DOUBLE PRECISION,
    debt_to_equity          DOUBLE PRECISION,
    current_ratio           DOUBLE PRECISION,
    interest_coverage_ratio DOUBLE PRECISION,

    -- Cash Flow Statement
    operating_cash_flow     DOUBLE PRECISION,
    capex                   DOUBLE PRECISION,
    free_cash_flow          DOUBLE PRECISION,

    -- Derived Metrics
    roic                    DOUBLE PRECISION,
    gross_margin            DOUBLE PRECISION,
    ebitda_margin           DOUBLE PRECISION,
    fcf_yield               DOUBLE PRECISION,

    -- Valuation at report date
    market_cap              DOUBLE PRECISION,
    current_price           DOUBLE PRECISION,
    forward_pe              DOUBLE PRECISION,
    peg_ratio               DOUBLE PRECISION,
    ev_to_ebitda            DOUBLE PRECISION,

    -- DCF Output
    dcf_intrinsic_value     DOUBLE PRECISION,
    dcf_margin_of_safety    DOUBLE PRECISION,       -- (intrinsic - price) / price
    wacc_used               DOUBLE PRECISION,

    -- Corporate Actions
    insider_buy_value       DOUBLE PRECISION,
    insider_sell_value      DOUBLE PRECISION,
    institutional_ownership_change_pct DOUBLE PRECISION,

    -- Earnings Call Sentiment (LLM-scored)
    transcript_sentiment_score  DOUBLE PRECISION,   -- [-1, +1]
    transcript_summary          TEXT,

    filing_url              TEXT,
    raw_data                JSONB
);
SELECT create_hypertable('fundamental_snapshots', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_fund_snap_symbol ON fundamental_snapshots (symbol, time DESC);

-- ─────────────────────────────────────────────────────────────────────
-- 4. trade_log — every simulated/executed trade
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trade_log (
    time                TIMESTAMPTZ NOT NULL,
    trade_id            TEXT NOT NULL,
    cycle_id            TEXT,
    symbol              TEXT NOT NULL,
    asset_class         TEXT NOT NULL,
    direction           TEXT NOT NULL,              -- LONG / SHORT
    timeframe           TEXT NOT NULL,
    entry_price         DOUBLE PRECISION,
    exit_price          DOUBLE PRECISION,
    quantity            DOUBLE PRECISION,
    notional_usd        DOUBLE PRECISION,
    entry_time          TIMESTAMPTZ,
    exit_time           TIMESTAMPTZ,
    commission          DOUBLE PRECISION,
    slippage            DOUBLE PRECISION,
    funding_cost        DOUBLE PRECISION,
    pnl_usd             DOUBLE PRECISION,
    pnl_pct             DOUBLE PRECISION,
    exit_reason         TEXT,
    strategy            TEXT,
    raw                 JSONB
);
SELECT create_hypertable('trade_log', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_log (symbol, time DESC);

-- ─────────────────────────────────────────────────────────────────────
-- 5. gating_log — Bull/Bear debate decisions
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gating_log (
    time              TIMESTAMPTZ NOT NULL,
    cycle_id          TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    bull_confidence   DOUBLE PRECISION,
    bear_confidence   DOUBLE PRECISION,
    confidence_delta  DOUBLE PRECISION,
    adaptive_threshold DOUBLE PRECISION,
    dominant_side     TEXT,
    tft_direction     TEXT,
    tft_aligned       BOOLEAN,
    decision          TEXT NOT NULL,                -- TRADE / NO_TRADE
    vix               DOUBLE PRECISION,
    bull_summary      TEXT,
    bear_summary      TEXT,
    raw               JSONB
);
SELECT create_hypertable('gating_log', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_gate_cycle ON gating_log (cycle_id, time DESC);

-- ─────────────────────────────────────────────────────────────────────
-- 6. portfolio_snapshot — daily portfolio state
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_snapshot (
    time                TIMESTAMPTZ NOT NULL,
    nav_usd             DOUBLE PRECISION,
    cash_usd            DOUBLE PRECISION,
    invested_usd        DOUBLE PRECISION,
    daily_pnl_usd       DOUBLE PRECISION,
    unrealized_pnl_usd  DOUBLE PRECISION,
    realized_pnl_usd    DOUBLE PRECISION,
    var_95_usd          DOUBLE PRECISION,
    gross_exposure_usd  DOUBLE PRECISION,
    position_count      INTEGER,
    raw                 JSONB
);
SELECT create_hypertable('portfolio_snapshot', 'time', if_not_exists => TRUE);

-- ─────────────────────────────────────────────────────────────────────
-- 7. daily_performance — daily PnL + all 10 metrics
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS daily_performance (
    time                    TIMESTAMPTZ NOT NULL,
    total_return_pct        DOUBLE PRECISION,
    cagr                    DOUBLE PRECISION,
    sharpe_ratio            DOUBLE PRECISION,
    sortino_ratio           DOUBLE PRECISION,
    calmar_ratio            DOUBLE PRECISION,
    information_ratio       DOUBLE PRECISION,
    max_drawdown_pct        DOUBLE PRECISION,
    max_drawdown_duration_days INTEGER,
    daily_var_95            DOUBLE PRECISION,
    volatility_annualized   DOUBLE PRECISION,
    total_trades            INTEGER,
    win_rate                DOUBLE PRECISION,
    profit_factor           DOUBLE PRECISION,
    expectancy_per_trade_usd DOUBLE PRECISION,
    alpha_vs_sp500          DOUBLE PRECISION,
    cost_drag_pct           DOUBLE PRECISION,
    raw                     JSONB
);
SELECT create_hypertable('daily_performance', 'time', if_not_exists => TRUE);

-- ─────────────────────────────────────────────────────────────────────
-- 8. reflection_prompts — LLM lesson injection (90-day retention)
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reflection_prompts (
    time        TIMESTAMPTZ NOT NULL,
    cycle_id    TEXT,
    lesson_text TEXT NOT NULL,
    applied     BOOLEAN DEFAULT FALSE
);
SELECT create_hypertable('reflection_prompts', 'time', if_not_exists => TRUE);

-- ─────────────────────────────────────────────────────────────────────
-- 9. circuit_breaker_events — risk system events
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    time        TIMESTAMPTZ NOT NULL,
    event_type  TEXT NOT NULL,          -- MAX_DRAWDOWN, POSITION_LIMIT, ...
    symbol      TEXT,
    severity    TEXT NOT NULL,          -- INFO / WARNING / CRITICAL
    message     TEXT,
    raw         JSONB
);
SELECT create_hypertable('circuit_breaker_events', 'time', if_not_exists => TRUE);

-- ─────────────────────────────────────────────────────────────────────
-- 10. Earnings & expiry calendars (used by calendar_features)
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS earnings_calendar (
    symbol        TEXT NOT NULL,
    earnings_date DATE NOT NULL,
    estimate      DOUBLE PRECISION,
    actual        DOUBLE PRECISION,
    PRIMARY KEY (symbol, earnings_date)
);

CREATE TABLE IF NOT EXISTS expiry_calendar (
    symbol      TEXT NOT NULL,
    expiry_date DATE NOT NULL,
    instrument  TEXT,
    PRIMARY KEY (symbol, expiry_date)
);

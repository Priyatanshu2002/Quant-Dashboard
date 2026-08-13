"""Unified storage adapter.

Two backends behind one API:
  * SQLite dev mode (default) — zero-dependency, mirrors the TimescaleDB schema.
  * TimescaleDB/PostgreSQL — when DATABASE_URL points at the Docker stack.

Schema is defined in databases/timescaledb/init/001_schema.sql; the SQLite
mirror is created automatically here.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.config import DATABASE_URL, PROJECT_ROOT

UTC = timezone.utc

# ─────────────────────────────────────────────────────────────────────
# Feature vector columns (mirrors 001_schema.sql feature_vectors table)
# ─────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "rsi_14", "rsi_7", "macd_histogram", "macd_signal_val", "stoch_k",
    "bb_pct_b", "bb_width", "atr_pct", "adx_14", "cci_20", "volume_z_score",
    "vwap_pct", "price_vs_ema200_pct", "realized_vol_20",
    "return_1bar", "return_5bar", "return_20bar", "return_60bar",
    "eps_surprise_pct", "eps_yoy_growth", "revenue_yoy_growth",
    "dcf_margin_of_safety", "forward_pe", "peg_ratio", "fcf_yield", "roic",
    "ebitda_margin", "debt_to_equity", "insider_buy_sell_ratio",
    "inst_ownership_change", "earnings_call_sentiment", "margin_trend",
    "revenue_accel",
    "sentiment_score", "sentiment_momentum", "sentiment_volume",
    "reddit_sentiment", "gdelt_sentiment", "news_sentiment", "sentiment_extreme",
    "vix", "vix_regime", "yield_curve_spread", "fed_funds_rate",
    "btc_dominance", "dxy",
    "days_to_earnings", "earnings_week", "days_to_expiry", "day_of_week",
    "month_end_effect", "quarter_end_effect",
    "future_return_1d", "future_return_5d", "future_return_20d", "future_sharpe_5d",
]

_FUNDAMENTAL_COLUMNS = [
    "period_type", "fiscal_year", "fiscal_quarter",
    "revenue", "revenue_estimate", "revenue_yoy_growth", "gross_profit",
    "ebitda", "net_income", "eps_actual", "eps_estimate", "eps_yoy_growth",
    "total_assets", "total_debt", "cash_and_equivalents", "net_debt",
    "shareholders_equity", "debt_to_equity", "current_ratio",
    "interest_coverage_ratio", "operating_cash_flow", "capex", "free_cash_flow",
    "roic", "gross_margin", "ebitda_margin", "fcf_yield", "market_cap",
    "current_price", "forward_pe", "peg_ratio", "ev_to_ebitda",
    "dcf_intrinsic_value", "dcf_margin_of_safety", "wacc_used",
    "insider_buy_value", "insider_sell_value",
    "institutional_ownership_change_pct", "transcript_sentiment_score",
    "transcript_summary", "filing_url", "raw_data",
]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_naive_utc(t: Any) -> datetime:
    """Normalize any timestamp to a tz-naive UTC datetime (SQLite-friendly)."""
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


# ─────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────
class Storage:
    backend: str = "base"

    # ── market data ──
    def write_ohlcv(self, df: pd.DataFrame, symbol: str, asset_class: str,
                    source: str, interval: str = "1d") -> None: ...
    def query_ohlcv(self, symbol: str, start: Any = None, end: Any = None,
                    interval: str = "1d") -> pd.DataFrame: ...
    def symbols(self) -> list[str]: ...

    # ── corporate actions ──
    def write_corporate_actions(self, symbol: str,
                                actions: Iterable[dict]) -> None: ...
    def query_corporate_actions(self, symbol: str) -> list[dict]: ...

    # ── feature store ──
    def write_feature_vectors(self, df: pd.DataFrame, symbol: str,
                              asset_class: str, timeframe: str) -> None: ...
    def query_feature_vectors(self, symbol: str | None = None,
                              timeframe: str | None = None,
                              start: Any = None, end: Any = None) -> pd.DataFrame: ...

    # ── fundamentals ──
    def upsert_fundamental_snapshot(self, snap: dict) -> None: ...
    def query_latest_fundamentals(self, symbol: str) -> dict | None: ...
    def query_fundamental_history(self, symbol: str, quarters: int = 8) -> list[dict]: ...

    # ── sentiment ──
    def write_sentiment_event(self, symbol: str, source: str, score: float,
                              headline: str = "", url: str = "",
                              source_weight: float = 1.0,
                              ts: datetime | None = None) -> None: ...
    def query_sentiment_events(self, symbol: str, hours: int = 24) -> list[dict]: ...
    def query_sentiment_avg(self, symbol: str, hours: int = 72) -> float: ...
    def sentiment_series(self, symbol: str, hours: int = 72,
                         bucket_minutes: int = 60) -> list[dict]: ...

    # ── financial statements (3-statement history, plan §8) ──
    def write_financial_statements(self, symbol: str, statement: str,
                                   rows: Iterable[dict]) -> None: ...
    def query_financial_statements(self, symbol: str,
                                   statement: str | None = None,
                                   quarters: int = 8,
                                   period_type: str = "QUARTERLY") -> list[dict]: ...

    # ── company profiles ──
    def upsert_company_profile(self, symbol: str, meta: dict) -> None: ...
    def get_company_profile(self, symbol: str) -> dict | None: ...

    # ── LLM analyses (news / earnings-call verdicts, plan §8.1) ──
    def upsert_llm_analysis(self, symbol: str, kind: str, verdict: dict,
                            model: str) -> None: ...
    def query_latest_llm_analysis(self, symbol: str,
                                  kind: str | None = None) -> dict | None: ...

    # ── macro ──
    def write_macro_snapshot(self, macro: dict) -> None: ...
    def query_latest_macro(self) -> dict | None: ...

    # ── calendars ──
    def write_earnings_dates(self, symbol: str, dates: Iterable[Any]) -> None: ...
    def get_next_earnings_date(self, symbol: str) -> date | None: ...
    def write_earnings_results(self, symbol: str, rows: Iterable[dict]) -> None: ...
    def query_earnings_results(self, symbol: str, limit: int = 8) -> list[dict]: ...
    def write_expiry_dates(self, symbol: str, dates: Iterable[Any]) -> None: ...
    def get_next_fo_expiry(self, symbol: str) -> date | None: ...

    # ── trade / risk logs ──
    def write_trade(self, trade: dict) -> None: ...
    def write_gating_log(self, row: dict) -> None: ...
    def write_portfolio_snapshot(self, row: dict) -> None: ...
    def write_daily_performance(self, row: dict) -> None: ...
    def write_reflection(self, cycle_id: str | None, lesson: str) -> None: ...
    def write_circuit_breaker(self, event_type: str, severity: str,
                              message: str, symbol: str | None = None,
                              raw: dict | None = None) -> None: ...


# ─────────────────────────────────────────────────────────────────────
# SQLite backend (dev mode)
# ─────────────────────────────────────────────────────────────────────
_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS market_data (
    time TEXT NOT NULL, symbol TEXT NOT NULL, asset_class TEXT NOT NULL,
    source TEXT NOT NULL, interval TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    dollar_volume REAL, raw TEXT
);
CREATE INDEX IF NOT EXISTS idx_md_sym ON market_data (symbol, interval, time);

CREATE TABLE IF NOT EXISTS feature_vectors (
    time TEXT NOT NULL, symbol TEXT NOT NULL, asset_class TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    {feature_cols},
    extra_features TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fv_uq ON feature_vectors (symbol, timeframe, time);

CREATE TABLE IF NOT EXISTS fundamental_snapshots (
    time TEXT NOT NULL, symbol TEXT NOT NULL, asset_class TEXT NOT NULL,
    {fund_cols}
);
CREATE INDEX IF NOT EXISTS idx_fs_sym ON fundamental_snapshots (symbol, time);

CREATE TABLE IF NOT EXISTS sentiment_events (
    ts TEXT NOT NULL, symbol TEXT NOT NULL, source TEXT NOT NULL,
    score REAL, source_weight REAL, headline TEXT, url TEXT,
    full_text TEXT, created_at TEXT, upvotes REAL, num_comments REAL,
    themes TEXT, tone REAL, raw TEXT
);
CREATE INDEX IF NOT EXISTS idx_se_sym ON sentiment_events (symbol, ts);

CREATE TABLE IF NOT EXISTS macro_snapshots (
    ts TEXT NOT NULL, us_10y_yield REAL, us_2y_yield REAL, fed_funds_rate REAL,
    vix REAL, dxy REAL, gold_pct_change_5d REAL, btc_dominance REAL,
    crypto_total_mcap_chg_24h REAL, yield_curve_spread REAL,
    us_1m_yield REAL, us_2m_yield REAL, us_3m_yield REAL, us_6m_yield REAL,
    us_1y_yield REAL, us_3y_yield REAL, us_5y_yield REAL, us_7y_yield REAL,
    us_20y_yield REAL, us_30y_yield REAL,
    us_5y_real_yield REAL, us_7y_real_yield REAL, us_10y_real_yield REAL,
    us_20y_real_yield REAL, us_30y_real_yield REAL, breakeven_inflation REAL,
    cpi_all_urban REAL, unemployment_rate REAL, core_cpi REAL, ppi_final REAL,
    pce_all REAL, nonfarm_payrolls REAL, ism_pmi REAL, m2_supply REAL,
    t10y2y REAL, t10y_breakeven_ie REAL, hy_credit_spread REAL,
    ig_credit_spread REAL, wti_price REAL, brent_price REAL,
    t5y_breakeven_ie REAL, job_openings REAL, retail_sales REAL,
    housing_starts REAL
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    symbol TEXT NOT NULL, earnings_date TEXT NOT NULL,
    PRIMARY KEY (symbol, earnings_date)
);
CREATE TABLE IF NOT EXISTS earnings_results (
    symbol TEXT NOT NULL, earnings_date TEXT NOT NULL,
    eps_actual REAL, eps_estimate REAL, eps_surprise_pct REAL,
    PRIMARY KEY (symbol, earnings_date)
);
CREATE TABLE IF NOT EXISTS expiry_calendar (
    symbol TEXT NOT NULL, expiry_date TEXT NOT NULL,
    PRIMARY KEY (symbol, expiry_date)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol TEXT NOT NULL, action_date TEXT NOT NULL,
    action_type TEXT NOT NULL, amount REAL, raw TEXT,
    PRIMARY KEY (symbol, action_date, action_type)
);

CREATE TABLE IF NOT EXISTS trade_log (
    time TEXT NOT NULL, trade_id TEXT, cycle_id TEXT, symbol TEXT NOT NULL,
    asset_class TEXT, direction TEXT, timeframe TEXT, entry_price REAL,
    exit_price REAL, quantity REAL, notional_usd REAL, entry_time TEXT,
    exit_time TEXT, commission REAL, slippage REAL, funding_cost REAL,
    pnl_usd REAL, pnl_pct REAL, exit_reason TEXT, strategy TEXT, raw TEXT
);

CREATE TABLE IF NOT EXISTS gating_log (
    time TEXT NOT NULL, cycle_id TEXT, symbol TEXT, bull_confidence REAL,
    bear_confidence REAL, confidence_delta REAL, adaptive_threshold REAL,
    dominant_side TEXT, tft_direction TEXT, tft_aligned INTEGER,
    decision TEXT, vix REAL, bull_summary TEXT, bear_summary TEXT, raw TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_snapshot (
    time TEXT NOT NULL, nav_usd REAL, cash_usd REAL, invested_usd REAL,
    daily_pnl_usd REAL, unrealized_pnl_usd REAL, realized_pnl_usd REAL,
    var_95_usd REAL, gross_exposure_usd REAL, position_count INTEGER, raw TEXT
);

CREATE TABLE IF NOT EXISTS daily_performance (
    time TEXT NOT NULL, total_return_pct REAL, cagr REAL, sharpe_ratio REAL,
    sortino_ratio REAL, calmar_ratio REAL, information_ratio REAL,
    max_drawdown_pct REAL, max_drawdown_duration_days INTEGER, daily_var_95 REAL,
    volatility_annualized REAL, total_trades INTEGER, win_rate REAL,
    profit_factor REAL, expectancy_per_trade_usd REAL, alpha_vs_sp500 REAL,
    cost_drag_pct REAL, raw TEXT
);

CREATE TABLE IF NOT EXISTS reflection_prompts (
    time TEXT NOT NULL, cycle_id TEXT, lesson_text TEXT, applied INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    time TEXT NOT NULL, event_type TEXT, symbol TEXT, severity TEXT,
    message TEXT, raw TEXT
);

CREATE TABLE IF NOT EXISTS financial_statements (
    symbol TEXT NOT NULL, statement TEXT NOT NULL, period TEXT NOT NULL,
    period_type TEXT NOT NULL DEFAULT 'QUARTERLY',
    data TEXT NOT NULL, fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, statement, period, period_type)
);
CREATE INDEX IF NOT EXISTS idx_fs_stmt ON financial_statements (symbol, statement, period);

CREATE TABLE IF NOT EXISTS company_profiles (
    symbol TEXT NOT NULL PRIMARY KEY, company_name TEXT, sector TEXT,
    industry TEXT, country TEXT, currency TEXT, website TEXT,
    employees INTEGER, updated_at TEXT NOT NULL, meta TEXT
);

CREATE TABLE IF NOT EXISTS llm_analyses (
    symbol TEXT NOT NULL, kind TEXT NOT NULL, verdict TEXT NOT NULL,
    model TEXT, created_at TEXT NOT NULL,
    PRIMARY KEY (symbol, kind)
);
"""


class SQLiteStorage(Storage):
    backend = "sqlite"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _session(self):
        """Connection with explicit commit/rollback/close — `with conn:` alone
        only commits and leaks the file handle (locks the DB on Windows)."""
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._session() as conn:
            conn.executescript(_SQLITE_DDL.format(
                feature_cols=", ".join(f"{c} REAL" for c in FEATURE_COLUMNS),
                fund_cols=", ".join(f"{c} REAL" for c in _FUNDAMENTAL_COLUMNS
                                    if c not in ("transcript_summary", "filing_url", "raw_data"))
                            + ", transcript_summary TEXT, filing_url TEXT, raw_data TEXT",
            ))
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotently add columns introduced after a DB was first created
        (so an existing data/agonistes_dev.db gains new macro/price fields)."""
        # {table: [(column, sql_type), ...]}
        _ADD_COLUMNS = {
            "macro_snapshots": [
                ("us_1m_yield", "REAL"), ("us_2m_yield", "REAL"),
                ("us_3m_yield", "REAL"), ("us_6m_yield", "REAL"),
                ("us_1y_yield", "REAL"), ("us_3y_yield", "REAL"),
                ("us_5y_yield", "REAL"), ("us_7y_yield", "REAL"),
                ("us_20y_yield", "REAL"), ("us_30y_yield", "REAL"),
                ("us_5y_real_yield", "REAL"), ("us_7y_real_yield", "REAL"),
                ("us_10y_real_yield", "REAL"), ("us_20y_real_yield", "REAL"),
                ("us_30y_real_yield", "REAL"), ("breakeven_inflation", "REAL"),
                ("cpi_all_urban", "REAL"), ("unemployment_rate", "REAL"),
                ("core_cpi", "REAL"), ("ppi_final", "REAL"), ("pce_all", "REAL"),
                ("nonfarm_payrolls", "REAL"), ("ism_pmi", "REAL"),
                ("m2_supply", "REAL"), ("t10y2y", "REAL"),
                ("t10y_breakeven_ie", "REAL"), ("hy_credit_spread", "REAL"),
                ("ig_credit_spread", "REAL"), ("wti_price", "REAL"),
                ("brent_price", "REAL"),
                ("t5y_breakeven_ie", "REAL"), ("job_openings", "REAL"),
                ("retail_sales", "REAL"), ("housing_starts", "REAL"),
            ],
            "market_data": [("dollar_volume", "REAL")],
            "company_profiles": [("meta", "TEXT")],
            "sentiment_events": [
                ("full_text", "TEXT"), ("created_at", "TEXT"),
                ("upvotes", "REAL"), ("num_comments", "REAL"),
                ("themes", "TEXT"), ("tone", "REAL"), ("raw", "TEXT"),
            ],
        }
        for table, cols in _ADD_COLUMNS.items():
            try:
                existing = {r["name"] for r in
                            conn.execute(f"PRAGMA table_info({table})").fetchall()}
            except sqlite3.OperationalError:
                continue
            for c, ctype in cols:
                if c not in existing:
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {c} {ctype}")
                    except sqlite3.OperationalError:
                        pass  # already added concurrently

    # ── market data ──
    def write_ohlcv(self, df: pd.DataFrame, symbol: str, asset_class: str,
                    source: str, interval: str = "1d") -> None:
        if df.empty:
            return
        out = pd.DataFrame({
            "time": [_as_naive_utc(t).isoformat(sep=" ") for t in df.index],
            "symbol": symbol, "asset_class": asset_class,
            "source": source, "interval": interval,
        })
        for c in ("open", "high", "low", "close", "volume", "dollar_volume"):
            out[c] = df[c].values if c in df.columns else None
        with self._session() as conn:
            out.to_sql("market_data", conn, if_exists="append", index=False)

    def query_ohlcv(self, symbol: str, start: Any = None, end: Any = None,
                    interval: str = "1d") -> pd.DataFrame:
        sql = "SELECT time, open, high, low, close, volume, dollar_volume FROM market_data WHERE symbol=? AND interval=?"
        args: list[Any] = [symbol, interval]
        if start is not None:
            sql += " AND time >= ?"
            args.append(_as_naive_utc(start).isoformat(sep=" "))
        if end is not None:
            sql += " AND time <= ?"
            args.append(_as_naive_utc(end).isoformat(sep=" "))
        sql += " ORDER BY time"
        with self._session() as conn:
            df = pd.read_sql_query(sql, conn, params=args)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"])
        # Overlapping backfills can leave duplicate timestamps — keep first.
        df = df.drop_duplicates(subset="time", keep="first")
        return df.set_index("time")

    def symbols(self) -> list[str]:
        with self._session() as conn:
            rows = conn.execute("SELECT DISTINCT symbol FROM market_data").fetchall()
        return [r["symbol"] for r in rows]

    # ── corporate actions ──
    def write_corporate_actions(self, symbol: str,
                                actions: Iterable[dict]) -> None:
        symbol = symbol.upper()
        with self._session() as conn:
            for a in actions:
                conn.execute(
                    "INSERT OR REPLACE INTO corporate_actions "
                    "(symbol, action_date, action_type, amount, raw) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (symbol, str(a["action_date"]), str(a["action_type"]),
                     a.get("amount"), json.dumps(a.get("raw") if a.get("raw") else a,
                                                 default=str)))

    def query_corporate_actions(self, symbol: str) -> list[dict]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM corporate_actions WHERE symbol=? ORDER BY action_date",
                (symbol.upper(),)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d.get("raw"):
                try:
                    d["raw"] = json.loads(d["raw"])
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(d)
        return out

    # ── feature store ──
    def write_feature_vectors(self, df: pd.DataFrame, symbol: str,
                              asset_class: str, timeframe: str) -> None:
        if df.empty:
            return
        cols = ["time", "symbol", "asset_class", "timeframe"] + [
            c for c in FEATURE_COLUMNS if c in df.columns]
        out = pd.DataFrame({
            "time": [_as_naive_utc(t).isoformat(sep=" ") for t in df.index],
            "symbol": symbol, "asset_class": asset_class, "timeframe": timeframe,
        })
        for c in FEATURE_COLUMNS:
            if c in df.columns:
                out[c] = df[c].values
        out = out[cols]
        with self._session() as conn:
            # Rebuild semantics: replace this symbol+timeframe slice atomically so
            # re-running feature builds is idempotent (UNIQUE index preserved).
            conn.execute("DELETE FROM feature_vectors WHERE symbol=? AND timeframe=?",
                         (symbol, timeframe))
            out = out.drop_duplicates(subset="time")  # guard vs duplicate bars
            out.to_sql("feature_vectors", conn, if_exists="append", index=False)

    def query_feature_vectors(self, symbol: str | None = None,
                              timeframe: str | None = None,
                              start: Any = None, end: Any = None) -> pd.DataFrame:
        sql = "SELECT * FROM feature_vectors WHERE 1=1"
        args: list[Any] = []
        if symbol:
            sql += " AND symbol=?"
            args.append(symbol)
        if timeframe:
            sql += " AND timeframe=?"
            args.append(timeframe)
        if start is not None:
            sql += " AND time >= ?"
            args.append(_as_naive_utc(start).isoformat(sep=" "))
        if end is not None:
            sql += " AND time <= ?"
            args.append(_as_naive_utc(end).isoformat(sep=" "))
        sql += " ORDER BY time"
        with self._session() as conn:
            df = pd.read_sql_query(sql, conn, params=args)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time")

    # ── fundamentals ──
    def upsert_fundamental_snapshot(self, snap: dict) -> None:
        snap = dict(snap)
        time_val = snap.pop("time", _utc_now())
        symbol = snap.pop("symbol")
        asset_class = snap.pop("asset_class", "EQUITY_US")
        raw = snap.pop("raw_data", None)
        cols = [c for c in _FUNDAMENTAL_COLUMNS if c in snap]
        cols_sql = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        vals = [json.dumps(raw) if c == "raw_data" else snap[c] for c in cols]
        sql = (f"INSERT OR REPLACE INTO fundamental_snapshots "
               f"(time, symbol, asset_class, {cols_sql}) VALUES (?, ?, ?, {placeholders})")
        with self._session() as conn:
            conn.execute(sql, [_as_naive_utc(time_val).isoformat(sep=" "),
                               symbol, asset_class, *vals])

    def query_latest_fundamentals(self, symbol: str) -> dict | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT * FROM fundamental_snapshots WHERE symbol=? ORDER BY time DESC LIMIT 1",
                (symbol,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["time"] = pd.Timestamp(d["time"])
        return d

    def query_fundamental_history(self, symbol: str, quarters: int = 8) -> list[dict]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM fundamental_snapshots WHERE symbol=? ORDER BY time DESC LIMIT ?",
                (symbol, quarters)).fetchall()
        return [dict(r) for r in rows]

    # ── sentiment ──
    def write_sentiment_event(self, symbol: str, source: str, score: float,
                              headline: str = "", url: str = "",
                              source_weight: float = 1.0,
                              ts: datetime | None = None,
                              full_text: str = "", created_at: str = "",
                              upvotes: float | None = None,
                              num_comments: float | None = None,
                              themes: str | None = None,
                              tone: float | None = None,
                              raw: dict | None = None) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT INTO sentiment_events "
                "(ts, symbol, source, score, source_weight, headline, url, "
                " full_text, created_at, upvotes, num_comments, themes, tone, raw) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_as_naive_utc(ts or _utc_now()).isoformat(sep=" "), symbol, source,
                 float(score), float(source_weight), headline, url,
                 full_text, created_at, upvotes, num_comments, themes, tone,
                 json.dumps(raw, default=str) if raw else None))

    def query_sentiment_events(self, symbol: str, hours: int = 24) -> list[dict]:
        cutoff = _as_naive_utc(_utc_now() - timedelta(hours=hours)).isoformat(sep=" ")
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM sentiment_events WHERE symbol=? AND ts >= ? ORDER BY ts",
                (symbol, cutoff)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d.get("raw"):
                try:
                    d["raw"] = json.loads(d["raw"])
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(d)
        return out

    def query_sentiment_avg(self, symbol: str, hours: int = 72) -> float:
        evs = self.query_sentiment_events(symbol, hours=hours)
        return sum(e["score"] for e in evs) / len(evs) if evs else 0.0

    def sentiment_series(self, symbol: str, hours: int = 72,
                         bucket_minutes: int = 60) -> list[dict]:
        """Time-bucketed weighted sentiment average → [{ts, score, volume}]."""
        evs = self.query_sentiment_events(symbol, hours=hours)
        if not evs:
            return []
        buckets: dict[pd.Timestamp, list[tuple[float, float]]] = {}
        step = pd.Timedelta(minutes=bucket_minutes)
        for e in evs:
            ts = pd.Timestamp(e["ts"]).floor(step)
            buckets.setdefault(ts, []).append((e["score"], e.get("source_weight", 1.0)))
        out = []
        for ts in sorted(buckets):
            pairs = buckets[ts]
            wsum = sum(w for _, w in pairs)
            out.append({
                "ts": ts.isoformat(sep=" "),
                "score": round(sum(s * w for s, w in pairs) / wsum, 6) if wsum else 0.0,
                "volume": len(pairs),
            })
        return out

    # ── financial statements ──
    def write_financial_statements(self, symbol: str, statement: str,
                                   rows: Iterable[dict]) -> None:
        symbol = symbol.upper()
        with self._session() as conn:
            for row in rows:
                period = str(row["period"])
                period_type = str(row.get("period_type", "QUARTERLY")).upper()
                data = {k: v for k, v in row.items()
                        if k not in ("period", "period_type")}
                conn.execute(
                    "INSERT OR REPLACE INTO financial_statements "
                    "(symbol, statement, period, period_type, data, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (symbol, statement, period, period_type,
                     json.dumps(data, default=str),
                     _as_naive_utc(_utc_now()).isoformat(sep=" ")))

    def query_financial_statements(self, symbol: str,
                                   statement: str | None = None,
                                   quarters: int = 8,
                                   period_type: str = "QUARTERLY") -> list[dict]:
        pt = period_type.upper()
        with self._session() as conn:
            if statement:
                rows = conn.execute(
                    "SELECT period, period_type, data FROM financial_statements "
                    "WHERE symbol=? AND statement=? AND period_type=? "
                    "ORDER BY period DESC LIMIT ?",
                    (symbol.upper(), statement, pt, quarters)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT statement, period, period_type, data FROM financial_statements "
                    "WHERE symbol=? AND period_type=? ORDER BY period DESC LIMIT ?",
                    (symbol.upper(), pt, quarters * 3)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"])
            out.append(d)
        return out

    # ── company profiles ──
    def upsert_company_profile(self, symbol: str, meta: dict) -> None:
        symbol = symbol.upper()
        # Drop the symbolic keys we store in dedicated columns; keep the rest
        # (incl. segments/executives/news/descriptions) as a JSON blob.
        blob = {k: v for k, v in meta.items()
                if k not in ("company_name", "sector", "industry", "country",
                             "currency", "website", "employees")}
        with self._session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO company_profiles "
                "(symbol, company_name, sector, industry, country, currency, "
                " website, employees, updated_at, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol,
                 meta.get("company_name"), meta.get("sector"),
                 meta.get("industry"), meta.get("country"), meta.get("currency"),
                 meta.get("website"), meta.get("employees"),
                 _as_naive_utc(_utc_now()).isoformat(sep=" "),
                 json.dumps(blob, default=str) if blob else None))

    def get_company_profile(self, symbol: str) -> dict | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT * FROM company_profiles WHERE symbol=?",
                (symbol.upper(),)).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("meta"):
            try:
                d["meta"] = json.loads(d["meta"])
            except (TypeError, json.JSONDecodeError):
                d["meta"] = {}
        return d

    # ── LLM analyses ──
    def upsert_llm_analysis(self, symbol: str, kind: str, verdict: dict,
                            model: str) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_analyses "
                "(symbol, kind, verdict, model, created_at) VALUES (?, ?, ?, ?, ?)",
                (symbol.upper(), kind.upper(), json.dumps(verdict, default=str),
                 model, _as_naive_utc(_utc_now()).isoformat(sep=" ")))

    def query_latest_llm_analysis(self, symbol: str,
                                  kind: str | None = None) -> dict | None:
        if kind:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT * FROM llm_analyses WHERE symbol=? AND kind=?",
                    (symbol.upper(), kind.upper())).fetchone()
        else:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT * FROM llm_analyses WHERE symbol=? ORDER BY created_at DESC LIMIT 1",
                    (symbol.upper(),)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["verdict"] = json.loads(d["verdict"])
        return d

    # ── macro ──
    def write_macro_snapshot(self, macro: dict) -> None:
        macro = dict(macro)
        ts = macro.pop("ts", _utc_now())
        cols = ", ".join(macro.keys())
        ph = ", ".join("?" for _ in macro)
        with self._session() as conn:
            conn.execute(
                f"INSERT INTO macro_snapshots (ts, {cols}) VALUES (?, {ph})",
                [_as_naive_utc(ts).isoformat(sep=" "), *macro.values()])

    def query_latest_macro(self) -> dict | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM macro_snapshots ORDER BY ts DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    # ── calendars ──
    def write_earnings_dates(self, symbol: str, dates: Iterable[Any]) -> None:
        with self._session() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO earnings_calendar (symbol, earnings_date) VALUES (?, ?)",
                [(symbol, pd.Timestamp(d).date().isoformat()) for d in dates])

    def get_next_earnings_date(self, symbol: str) -> date | None:
        today = date.today().isoformat()
        with self._session() as conn:
            row = conn.execute(
                "SELECT earnings_date FROM earnings_calendar WHERE symbol=? AND earnings_date >= ? "
                "ORDER BY earnings_date LIMIT 1", (symbol, today)).fetchone()
        return date.fromisoformat(row["earnings_date"]) if row else None

    def write_earnings_results(self, symbol: str, rows: Iterable[dict]) -> None:
        with self._session() as conn:
            for r in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO earnings_results "
                    "(symbol, earnings_date, eps_actual, eps_estimate, eps_surprise_pct) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (symbol.upper(), str(r["earnings_date"]), r.get("eps_actual"),
                     r.get("eps_estimate"), r.get("eps_surprise_pct")))

    def query_earnings_results(self, symbol: str, limit: int = 8) -> list[dict]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM earnings_results WHERE symbol=? ORDER BY earnings_date DESC LIMIT ?",
                (symbol.upper(), limit)).fetchall()
        return [dict(r) for r in rows]

    def write_expiry_dates(self, symbol: str, dates: Iterable[Any]) -> None:
        with self._session() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO expiry_calendar (symbol, expiry_date) VALUES (?, ?)",
                [(symbol, pd.Timestamp(d).date().isoformat()) for d in dates])

    def get_next_fo_expiry(self, symbol: str) -> date | None:
        today = date.today().isoformat()
        with self._session() as conn:
            row = conn.execute(
                "SELECT expiry_date FROM expiry_calendar WHERE symbol=? AND expiry_date >= ? "
                "ORDER BY expiry_date LIMIT 1", (symbol, today)).fetchone()
        return date.fromisoformat(row["expiry_date"]) if row else None

    # ── logs ──
    def write_trade(self, trade: dict) -> None:
        trade = dict(trade)
        raw = trade.pop("raw", None)
        cols = [c for c in trade if c != "time"] + (["raw"] if raw is not None else [])
        sql = (f"INSERT INTO trade_log (time, {', '.join(cols)}) VALUES (?, "
               f"{', '.join('?' for _ in cols)})")
        with self._session() as conn:
            conn.execute(sql, [_as_naive_utc(trade.pop("time", _utc_now())).isoformat(sep=" "),
                               *[json.dumps(raw) if c == "raw" else trade.get(c) for c in cols]])

    def write_gating_log(self, row: dict) -> None:
        row = dict(row)
        ts = row.pop("time", _utc_now())
        raw = row.pop("raw", None)
        cols = list(row.keys()) + (["raw"] if raw is not None else [])
        sql = f"INSERT INTO gating_log (time, {', '.join(cols)}) VALUES (?, {', '.join('?' for _ in cols)})"
        with self._session() as conn:
            conn.execute(sql, [_as_naive_utc(ts).isoformat(sep=" "),
                               *[json.dumps(raw) if c == "raw" else row.get(c) for c in cols]])

    def write_portfolio_snapshot(self, row: dict) -> None:
        row = dict(row)
        ts = row.pop("time", _utc_now())
        raw = row.pop("raw", None)
        cols = list(row.keys()) + (["raw"] if raw is not None else [])
        sql = (f"INSERT INTO portfolio_snapshot (time, {', '.join(cols)}) "
               f"VALUES (?, {', '.join('?' for _ in cols)})")
        with self._session() as conn:
            conn.execute(sql, [_as_naive_utc(ts).isoformat(sep=" "),
                               *[json.dumps(raw) if c == "raw" else row.get(c) for c in cols]])

    def write_daily_performance(self, row: dict) -> None:
        row = dict(row)
        ts = row.pop("time", _utc_now())
        raw = row.pop("raw", None)
        cols = list(row.keys()) + (["raw"] if raw is not None else [])
        sql = (f"INSERT INTO daily_performance (time, {', '.join(cols)}) "
               f"VALUES (?, {', '.join('?' for _ in cols)})")
        with self._session() as conn:
            conn.execute(sql, [_as_naive_utc(ts).isoformat(sep=" "),
                               *[json.dumps(raw) if c == "raw" else row.get(c) for c in cols]])

    def write_reflection(self, cycle_id: str | None, lesson: str) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT INTO reflection_prompts (time, cycle_id, lesson_text) VALUES (?, ?, ?)",
                (_as_naive_utc(_utc_now()).isoformat(sep=" "), cycle_id, lesson))

    def write_circuit_breaker(self, event_type: str, severity: str, message: str,
                              symbol: str | None = None, raw: dict | None = None) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT INTO circuit_breaker_events (time, event_type, symbol, severity, message, raw) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_as_naive_utc(_utc_now()).isoformat(sep=" "), event_type, symbol,
                 severity, message, json.dumps(raw) if raw else None))


# ─────────────────────────────────────────────────────────────────────
# TimescaleDB backend (Docker stack)
# ─────────────────────────────────────────────────────────────────────
class TimescaleStorage(Storage):
    backend = "timescaledb"

    def __init__(self, url: str):
        try:
            import psycopg2  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "TimescaleDB backend needs psycopg2-binary: pip install -e '.[db]'"
            ) from e
        self.url = url

    def _conn(self):
        import psycopg2
        return psycopg2.connect(self.url)

    @contextmanager
    def _session(self):
        """Connection with explicit commit/rollback/close (mirrors SQLite)."""
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def query_ohlcv(self, symbol: str, start: Any = None, end: Any = None,
                    interval: str = "1d") -> pd.DataFrame:
        sql = "SELECT time, open, high, low, close, volume FROM market_data WHERE symbol=%s AND interval=%s"
        args: list[Any] = [symbol, interval]
        if start is not None:
            sql += " AND time >= %s"
            args.append(pd.Timestamp(start))
        if end is not None:
            sql += " AND time <= %s"
            args.append(pd.Timestamp(end))
        sql += " ORDER BY time"
        with self._session() as conn:
            df = pd.read_sql_query(sql, conn, params=args)
        if df.empty:
            return df
        return df.set_index("time")

    def write_ohlcv(self, df: pd.DataFrame, symbol: str, asset_class: str,
                    source: str, interval: str = "1d") -> None:
        if df.empty:
            return
        rows = [(_as_naive_utc(t), symbol, asset_class, source, interval,
                 *[float(df.at[t, c]) if c in df.columns and pd.notna(df.at[t, c]) else None
                   for c in ("open", "high", "low", "close", "volume")])
                for t in df.index]
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO market_data (time, symbol, asset_class, source, interval, "
                "open, high, low, close, volume) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT DO NOTHING", rows)
            conn.commit()

    def write_feature_vectors(self, df: pd.DataFrame, symbol: str,
                              asset_class: str, timeframe: str) -> None:
        if df.empty:
            return
        cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        rows = [(_as_naive_utc(t), symbol, asset_class, timeframe,
                 *[None if pd.isna(df.at[t, c]) else float(df.at[t, c]) for c in cols])
                for t in df.index]
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO feature_vectors (time, symbol, asset_class, timeframe, "
                f"{', '.join(cols)}) VALUES (%s,%s,%s,%s,{', '.join(['%s'] * len(cols))}) "
                f"ON CONFLICT DO NOTHING", rows)
            conn.commit()

    def query_feature_vectors(self, symbol: str | None = None,
                              timeframe: str | None = None,
                              start: Any = None, end: Any = None) -> pd.DataFrame:
        sql = "SELECT * FROM feature_vectors WHERE 1=1"
        args: list[Any] = []
        if symbol:
            sql += " AND symbol=%s"
            args.append(symbol)
        if timeframe:
            sql += " AND timeframe=%s"
            args.append(timeframe)
        if start is not None:
            sql += " AND time >= %s"
            args.append(pd.Timestamp(start))
        if end is not None:
            sql += " AND time <= %s"
            args.append(pd.Timestamp(end))
        sql += " ORDER BY time"
        with self._session() as conn:
            return pd.read_sql_query(sql, conn, params=args).set_index("time")

    def symbols(self) -> list[str]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM market_data")
            return [r[0] for r in cur.fetchall()]

    def upsert_fundamental_snapshot(self, snap: dict) -> None:
        raise NotImplementedError("Fundamentals via TimescaleDB: use the SQLite backend "
                                  "until the ingestion pipeline is wired to psycopg2.")

    def query_latest_fundamentals(self, symbol: str) -> dict | None:
        return None

    def query_fundamental_history(self, symbol: str, quarters: int = 8) -> list[dict]:
        return []

    def write_sentiment_event(self, *args, **kwargs) -> None:
        raise NotImplementedError("Sentiment via TimescaleDB not yet wired.")

    def query_sentiment_events(self, symbol: str, hours: int = 24) -> list[dict]:
        return []

    def query_sentiment_avg(self, symbol: str, hours: int = 72) -> float:
        return 0.0

    def sentiment_series(self, symbol: str, hours: int = 72,
                         bucket_minutes: int = 60) -> list[dict]:
        return []

    def write_financial_statements(self, symbol: str, statement: str,
                                   rows: Iterable[dict]) -> None:
        raise NotImplementedError("Statements via TimescaleDB not yet wired.")

    def query_financial_statements(self, symbol: str,
                                   statement: str | None = None,
                                   quarters: int = 8) -> list[dict]:
        return []

    def upsert_company_profile(self, symbol: str, meta: dict) -> None:
        raise NotImplementedError("Company profiles via TimescaleDB not yet wired.")

    def get_company_profile(self, symbol: str) -> dict | None:
        return None

    def upsert_llm_analysis(self, symbol: str, kind: str, verdict: dict,
                            model: str) -> None:
        raise NotImplementedError("LLM analyses via TimescaleDB not yet wired.")

    def query_latest_llm_analysis(self, symbol: str,
                                  kind: str | None = None) -> dict | None:
        return None

    def write_macro_snapshot(self, macro: dict) -> None:
        raise NotImplementedError("Macro via TimescaleDB not yet wired.")

    def query_latest_macro(self) -> dict | None:
        return None

    def write_earnings_dates(self, symbol: str, dates: Iterable[Any]) -> None: ...
    def get_next_earnings_date(self, symbol: str) -> date | None:
        return None

    def write_earnings_results(self, symbol: str, rows: Iterable[dict]) -> None:
        raise NotImplementedError("Earnings results via TimescaleDB not yet wired.")

    def query_earnings_results(self, symbol: str, limit: int = 8) -> list[dict]:
        return []
    def write_expiry_dates(self, symbol: str, dates: Iterable[Any]) -> None: ...
    def get_next_fo_expiry(self, symbol: str) -> date | None:
        return None

    def write_trade(self, trade: dict) -> None:
        raise NotImplementedError("Trade logging via TimescaleDB not yet wired.")

    def write_gating_log(self, row: dict) -> None: ...
    def write_portfolio_snapshot(self, row: dict) -> None: ...
    def write_daily_performance(self, row: dict) -> None: ...
    def write_reflection(self, cycle_id: str | None, lesson: str) -> None: ...
    def write_circuit_breaker(self, event_type: str, severity: str, message: str,
                              symbol: str | None = None, raw: dict | None = None) -> None: ...


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────
_default_storage: Storage | None = None


def get_storage(url: str | None = None) -> Storage:
    """Return the process-wide storage instance (or create it)."""
    global _default_storage
    url = url or DATABASE_URL
    if _default_storage is None:
        _default_storage = make_storage(url)
    return _default_storage


def make_storage(url: str | None = None) -> Storage:
    url = url or DATABASE_URL
    if url.startswith("sqlite"):
        path = url.replace("sqlite:///", "", 1)
        if not Path(path).is_absolute():
            path = str(PROJECT_ROOT / path)
        return SQLiteStorage(path)
    if url.startswith("postgres"):
        return TimescaleStorage(url)
    raise ValueError(f"Unsupported DATABASE_URL scheme: {url}")

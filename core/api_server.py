"""Minimal stdlib JSON API for the UI dashboard (no extra dependencies).

Endpoints:
  GET  /api/screener/top            top-N candidates with signal breakdown
  GET  /api/backtest/report         BacktestReport for a symbol+strategy
  GET  /api/backtest/equity         equity curve points
  GET  /api/portfolio/snapshot      latest portfolio snapshot
  GET  /api/financials?symbol=      §8.4 bundle: snapshot + DCF + 3 statements
                                    + ratios + earnings + LLM analyst
  GET  /api/sentiment?symbol=&hours=  news sentiment: aggregate, per-source,
                                    series, events, LLM verdict
  GET  /api/debate/recent           recent gating decisions
  POST /api/fundamentals/refresh?symbol=  refetch snapshot + statements + DCF
                                    + company profile + LLM fundamental verdict
  POST /api/sentiment/refresh?symbol=     refetch news events + LLM verdict
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from core.db import get_storage
from core.logging import get_logger

log = get_logger(__name__)

# ── Prometheus metrics registry (in-memory, per-process) ────────────────
_metrics_lock = threading.Lock()
_http_requests: dict[str, int] = defaultdict(int)        # route → count
_http_duration_histo: dict[tuple[str, str], int] = defaultdict(int)  # (route, le) → count
_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf"))
_events_total: dict[str, int] = defaultdict(int)         # feed → count
_screener_scored: int = 0
_screener_passed: int = 0
_llm_cost_usd: float = 0.0


def record_event(feed: str, n: int = 1) -> None:
    """Increment the events_total counter for a feed (called from the event bus)."""
    with _metrics_lock:
        _events_total[feed] += n


def record_screener(scored: int, passed: int) -> None:
    """Record a screener run for the scored/passed Prometheus gauges."""
    global _screener_scored, _screener_passed
    with _metrics_lock:
        _screener_scored += scored
        _screener_passed += passed


def record_llm_cost(usd: float) -> None:
    global _llm_cost_usd
    with _metrics_lock:
        _llm_cost_usd += usd


def _record_duration(route: str, t0: float) -> None:
    """Record an HTTP request duration into the histogram buckets.

    Acquires the metrics lock internally (call WITHOUT holding it)."""
    dur = time.monotonic() - t0
    with _metrics_lock:
        _http_requests[route] += 1
        for b in _HTTP_BUCKETS:
            if dur <= b:
                _http_duration_histo[(route, b)] += 1


def _fmt_metric(name: str, help_text: str, type_: str) -> str:
    return (f"# HELP {name} {help_text}\n# TYPE {name} {type_}\n")


def _route_metrics(db=None) -> str:
    """Render Prometheus text-format metrics from in-memory counters + live DB."""
    import sqlite3
    db = db or get_storage()
    lines: list[str] = []

    # HTTP request counters (by route)
    with _metrics_lock:
        reqs = dict(_http_requests)
        histo = dict(_http_duration_histo)
        events = dict(_events_total)
        llm = _llm_cost_usd
        scored = _screener_scored
        passed = _screener_passed
    lines.append(_fmt_metric("agonistes_http_requests_total",
                             "HTTP requests served, by route.", "counter"))
    for route, n in sorted(reqs.items()):
        lines.append(f'agonistes_http_requests_total{{route="{route}"}} {n}')

    lines.append(_fmt_metric("agonistes_http_request_duration_seconds",
                             "HTTP request duration histogram.", "histogram"))
    for (route, le), n in sorted(histo.items()):
        _le = le if le != float("inf") else "+Inf"
        lines.append(f'agonistes_http_request_duration_seconds_bucket{{route="{route}",le="{_le}"}} {n}')

    lines.append(_fmt_metric("agonistes_events_total",
                             "Events published on the bus, by feed.", "counter"))
    for feed, n in sorted(events.items()):
        lines.append(f'agonistes_events_total{{feed="{feed}"}} {n}')

    lines.append(_fmt_metric("agonistes_llm_cost_usd_total",
                             "Accumulated LLM spend in USD.", "counter"))
    lines.append(f"agonistes_llm_cost_usd_total {llm}")

    lines.append(_fmt_metric("agonistes_screener_scored_total",
                             "Assets scored by the screener.", "counter"))
    lines.append(f"agonistes_screener_scored_total {scored}")
    lines.append(_fmt_metric("agonistes_screener_passed_total",
                             "Assets passing the screener threshold.", "counter"))
    lines.append(f"agonistes_screener_passed_total {passed}")

    # Live gauges from the SQLite DB
    def _gauge_int(sql: str, metric: str, help_text: str, label: str | None = None):
        lines.append(_fmt_metric(metric, help_text, "gauge"))
        try:
            with db._conn() as conn:
                rows = conn.execute(sql).fetchall()
        except sqlite3.OperationalError:
            return
        for r in rows:
            if label:
                lines.append(f'{metric}{{{label}="{r[0]}"}} {r[1] if r[1] is not None else 0}')
            else:
                lines.append(f"{metric} {r[0] if r[0] is not None else 0}")

    _gauge_int("SELECT direction, COUNT(*) FROM trade_log GROUP BY direction",
               "agonistes_trades_by_direction", "Trades by direction.", "direction")
    _gauge_int("SELECT direction, COUNT(*) FROM trade_log GROUP BY direction",
               "agonistes_trades_total", "Trades by direction.", "direction")
    _gauge_int("SELECT decision, COUNT(*) FROM gating_log GROUP BY decision",
               "agonistes_gating_total", "Gating decisions by outcome.", "decision")
    _gauge_int("SELECT event_type, COUNT(*) FROM circuit_breaker_events GROUP BY event_type",
               "agonistes_circuit_breaker_total", "Circuit breaker events by type.", "event_type")
    _gauge_int("SELECT COUNT(*) FROM market_data WHERE symbol NOT LIKE 'ONCHAIN_%'",
               "agonistes_ohlcv_bars_total",
               "OHLCV bars stored.", None)
    _gauge_int("SELECT COUNT(*) FROM feature_vectors", "agonistes_feature_vectors_total",
               "Feature vectors stored.", None)

    # Portfolio gauges
    try:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT nav_usd, daily_pnl_usd FROM portfolio_snapshot "
                "ORDER BY time DESC LIMIT 1").fetchone()
        if row:
            lines.append("# TYPE agonistes_nav_usd gauge\nagonistes_nav_usd "
                         f"{row['nav_usd'] if row['nav_usd'] is not None else 0}")
            lines.append("# TYPE agonistes_daily_pnl_usd gauge\nagonistes_daily_pnl_usd "
                         f"{row['daily_pnl_usd'] if row['daily_pnl_usd'] is not None else 0}")
    except Exception:  # noqa: BLE001
        pass

    return "\n".join(lines) + "\n"


def _q(qs: dict, key: str, default: str) -> str:
    return qs.get(key, [default])[0]


# ── route handlers ──────────────────────────────────────────────────────
# Map SEC EDGAR XBRL keys → the canonical yfinance keys the UI row-defs expect,
# so the Annual (SEC) and Quarterly (yfinance) statement views line up.
_SEC_KEY_ALIAS = {
    "revenue": "total_revenue",
    "eps_actual": "eps_diluted",
    "depreciation_amortization": "depreciation",
    "common_stock": "common_stock",
    "net_ppe": "net_ppe",
}


def _norm_statement_keys(row: dict) -> dict:
    out = dict(row)
    for src, dst in _SEC_KEY_ALIAS.items():
        if src in out and src != dst and dst not in out:
            out[dst] = out[src]
            del out[src]
    return out


def _route_financials(db, qs: dict) -> dict:
    symbol = _q(qs, "symbol", "AAPL").upper()

    from data_ingestion.fundamental_feeds.dcf_scenarios import dcf_bundle
    from valuation.cfa_model import build_model

    snap = db.query_latest_fundamentals(symbol) or {}
    if snap:
        snap = {k: v for k, v in snap.items()
                if k not in ("raw_data", "filing_url", "transcript_summary")}
        snap["time"] = str(snap.get("time"))

    # quarterly + annual statements (SEC EDGAR deep history), oldest → newest
    statements: dict[str, list] = {}
    annual_statements: dict[str, list] = {}
    for name in ("income", "balance", "cashflow"):
        q = db.query_financial_statements(symbol, statement=name, quarters=24, period_type="QUARTERLY")
        q = [_norm_statement_keys({"period": r["period"], **r["data"]}) for r in q]
        q.sort(key=lambda r: r["period"])
        statements[name] = q
        a = db.query_financial_statements(symbol, statement=name, quarters=24, period_type="ANNUAL")
        a = [_norm_statement_keys({"period": r["period"], **r["data"]}) for r in a]
        a.sort(key=lambda r: r["period"])
        annual_statements[name] = a

    # per-quarter derived ratios (merged view for the trend charts)
    ratios = db.query_financial_statements(symbol, quarters=24)
    ratio_rows: dict[str, dict] = {}
    for r in ratios:
        ratio_rows.setdefault(r["period"], {})[r["statement"]] = r["data"]
    ratio_series = [{"period": p, "income": v.get("income", {}),
                     "balance": v.get("balance", {}), "cashflow": v.get("cashflow", {})}
                    for p, v in sorted(ratio_rows.items())]

    # price change for the header (last close vs prior close)
    price_change: float | None = None
    ohlcv = db.query_ohlcv(symbol)
    if len(ohlcv) >= 2:
        closes = ohlcv["close"].dropna()
        if len(closes) >= 2:
            price_change = round(float(closes.iloc[-1] / closes.iloc[-2] - 1) * 100, 2)

    # earnings calendar + reported results
    earnings = {
        "next_date": db.get_next_earnings_date(symbol),
        "results": db.query_earnings_results(symbol, limit=8),
    }

    payload = {
        "symbol": symbol,
        "profile": db.get_company_profile(symbol),
        "snapshot": snap or None,
        "dcf": dcf_bundle(snap) if snap else None,
        "cfa": build_model(db, symbol),  # CFA-standard 3-statement + DCF model
        "statements": statements,
        "annual_statements": annual_statements,  # SEC EDGAR multi-year history
        "ratios": ratio_series,
        "price_change_pct": price_change,
        "earnings": earnings,
        "llm_analyses": {
            "news": db.query_latest_llm_analysis(symbol, kind="NEWS"),
            "fundamental": db.query_latest_llm_analysis(symbol, kind="FUNDAMENTAL"),
        },
    }
    return payload


def _route_sentiment(db, qs: dict) -> dict:
    symbol = _q(qs, "symbol", "AAPL").upper()
    hours = int(_q(qs, "hours", "72"))

    events = db.query_sentiment_events(symbol, hours=hours)
    scores = [e["score"] for e in events]
    weights = [e.get("source_weight", 1.0) for e in events]
    wsum = sum(weights)
    aggregate = {
        "score": round(sum(s * w for s, w in zip(scores, weights)) / wsum, 4) if wsum else 0.0,
        "volume": len(events),
        "positive_pct": round(sum(1 for s in scores if s > 0.2) / len(scores), 4) if scores else 0.0,
        "negative_pct": round(sum(1 for s in scores if s < -0.2) / len(scores), 4) if scores else 0.0,
        "momentum": round((sum(s * w for s, w in zip(scores, weights)) / wsum
                           if wsum else 0.0)
                          - db.query_sentiment_avg(symbol, hours=max(hours, 72)), 4),
    }

    per_source: dict[str, dict] = {}
    for e in events:
        src = per_source.setdefault(e["source"], {"score": 0.0, "volume": 0, "wsum": 0.0})
        w = e.get("source_weight", 1.0)
        src["score"] += e["score"] * w
        src["wsum"] += w
        src["volume"] += 1
    for src in per_source.values():
        src["score"] = round(src["score"] / src["wsum"], 4) if src["wsum"] else 0.0
        src.pop("wsum")

    return {
        "symbol": symbol,
        "hours": hours,
        "aggregate": aggregate,
        "per_source": per_source,
        "series": db.sentiment_series(symbol, hours=hours, bucket_minutes=180),
        "events": events[-60:][::-1],  # newest first, cap 60
        "llm": db.query_latest_llm_analysis(symbol, kind="NEWS"),
    }


def _route_refresh_fundamentals(db, qs: dict) -> dict:
    symbol = _q(qs, "symbol", "AAPL").upper()
    from data_ingestion.fundamental_feeds.dcf_scenarios import apply_dcf_to_snapshot
    from data_ingestion.fundamental_feeds.yfinance_earnings import (
        fetch_earnings_dates,
        refresh_info_snapshot,
    )
    from data_ingestion.fundamental_feeds.yfinance_financials import (
        fetch_company_profile,
        fetch_quarterly_statements,
    )
    from data_ingestion.sentiment_feeds.llm_analyst import analyze_fundamentals

    snap = refresh_info_snapshot(symbol, storage=db) or {}
    apply_dcf_to_snapshot(snap)
    if snap.get("dcf_intrinsic_value"):
        db.upsert_fundamental_snapshot(snap)

    stmts = fetch_quarterly_statements(symbol, quarters=8, storage=db)

    try:
        dates = fetch_earnings_dates(symbol, limit=8, storage=db)
        if not dates.empty:
            results = []
            for _, row in dates.iterrows():
                if pd.notna(row.get("eps_actual")) and pd.notna(row.get("eps_estimate")):
                    results.append({
                        "earnings_date": str(row["earnings_date"].date()),
                        "eps_actual": float(row["eps_actual"]),
                        "eps_estimate": float(row["eps_estimate"]),
                        "eps_surprise_pct": float(row["eps_actual"]) / float(row["eps_estimate"]) - 1
                        if row["eps_estimate"] else None,
                    })
            db.write_earnings_results(symbol, results)
    except Exception as e:  # noqa: BLE001
        log.debug("earnings dates failed for %s: %s", symbol, e)

    profile = fetch_company_profile(symbol, storage=db)
    verdict = analyze_fundamentals(symbol, storage=db, db=db, force=True)

    return {"symbol": symbol, "profile": profile,
            "statements": {k: len(v) for k, v in stmts.items()},
            "dcf_intrinsic_value": snap.get("dcf_intrinsic_value"),
            "dcf_margin_of_safety": snap.get("dcf_margin_of_safety"),
            "llm_fundamental": verdict}


def _route_refresh_sentiment(db, qs: dict) -> dict:
    symbol = _q(qs, "symbol", "AAPL").upper()
    from data_ingestion.sentiment_feeds.llm_analyst import analyze_news_sentiment
    from data_ingestion.sentiment_feeds.news_aggregator import fetch_news_events

    events = fetch_news_events(symbol, days=1, storage=db)
    verdict = analyze_news_sentiment(symbol, storage=db, db=db, force=True)
    return {"symbol": symbol, "events_fetched": len(events),
            "llm": verdict}


def _route_screener_market(db) -> dict:
    """Full market screener — every scoreable instrument with live price data,
    sorted by composite, plus per-signal components (TradingView-style table).

    Scored result is cached (TTL) so the screener page loads instantly after the
    first call instead of re-scoring ~112 instruments on every request.
    """
    import time
    now = time.time()
    cached = _MARKET_CACHE.get("payload")
    if cached and now - _MARKET_CACHE.get("ts", 0) < _MARKET_TTL:
        return cached

    from screener.pipeline import score_universe

    signals = score_universe(db)
    rows = []
    for s in signals:
        try:
            ohlcv = db.query_ohlcv(s.symbol)
            closes = ohlcv["close"].dropna()
            price = float(closes.iloc[-1]) if len(closes) else None
            change = float(closes.iloc[-1] / closes.iloc[-2] - 1) if len(closes) >= 2 else None
            volume = float(ohlcv["volume"].dropna().iloc[-1]) if len(ohlcv) else None
        except Exception:  # noqa: BLE001
            price = change = volume = None
        try:
            snap = db.query_latest_fundamentals(s.symbol) or {}
        except Exception:  # noqa: BLE001
            snap = {}
        try:
            prof = db.get_company_profile(s.symbol) or {}
        except Exception:  # noqa: BLE001
            prof = {}
        b = s.breakdown()
        rows.append({
            "symbol": s.symbol,
            "name": prof.get("company_name"),
            "asset_class": s.asset_class,
            "sector": prof.get("sector"),
            "price": round(price, 2) if price else None,
            "change_pct": round(change * 100, 2) if change is not None else None,
            "volume": volume,
            "market_cap": snap.get("market_cap"),
            "composite": round(b["composite"], 1),
            "technical": round(b["technical"], 1),
            "fundamental": round(b["fundamental"], 1),
            "sentiment": round(b["sentiment"], 1),
            "macro": round(b["macro"], 1),
            "momentum": round(b["momentum"], 1),
        })
    rows.sort(key=lambda r: -(r["composite"] or 0))
    payload = {"count": len(rows), "rows": rows}
    _MARKET_CACHE["payload"] = payload
    _MARKET_CACHE["ts"] = now
    return payload


_MARKET_CACHE: dict = {}
_MARKET_TTL = 900  # seconds (15 min)


def _warm_market_cache() -> None:
    """Pre-score the full universe in a background thread so the first UI load
    is instant instead of a ~40s cold scoring pass."""
    import threading
    import time

    def _warm() -> None:
        time.sleep(3)
        try:
            from core.db import get_storage
            _route_screener_market(get_storage())
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_warm, daemon=True).start()


def _route_screener_top(db, qs: dict) -> dict | list:
    from screener.pipeline import run_screener, screener_table
    top_n = int(_q(qs, "top", "10"))
    return screener_table(run_screener(top_n=top_n, db=db))


def _route_screener_universe(db) -> dict:
    """Cheap symbol+name+sector list for the whole universe (no scoring) —
    backs the ticker search dropdown. Cached briefly."""
    import time
    now = time.time()
    cached = _UNIVERSE_CACHE.get("payload")
    if cached and now - _UNIVERSE_CACHE.get("ts", 0) < _UNIVERSE_TTL:
        return cached

    from screener.asset_universe import get_universe
    u = get_universe()
    rows = []
    for e in u.entries:
        rows.append({
            "symbol": e.symbol,
            "name": (e.name or ""),
            "asset_class": e.asset_class,
            "sector": (e.sector or ""),
        })
    payload = {"count": len(rows), "rows": rows}
    _UNIVERSE_CACHE["payload"] = payload
    _UNIVERSE_CACHE["ts"] = now
    return payload


_UNIVERSE_CACHE: dict = {}
_UNIVERSE_TTL = 60  # 1 min — new tickers added to the config appear quickly, no restart


def _route_monitoring(db) -> dict:
    """System health: storage backend, table coverage, DB size, containerized
    services, and feed status (plan §11 Monitoring)."""
    from pathlib import Path

    # Feed status = whether a meaningful amount of data exists for each source.
    FEEDS = [
        ("prices", "Market prices", "market_data"),
        ("features", "Feature vectors", "feature_vectors"),
        ("fundamentals", "Fundamental snapshots", "fundamental_snapshots"),
        ("statements", "3-statement history", "financial_statements"),
        ("profiles", "Company profiles", "company_profiles"),
        ("sentiment", "News / sentiment", "sentiment_events"),
        ("llm_analyses", "LLM analyst verdicts", "llm_analyses"),
        ("earnings", "Earnings calendar", "earnings_results"),
        ("macro", "Macro indicators", "macro_snapshots"),
        ("onchain", "On-chain (Dune/CoinGecko)", "market_data"),
        ("debate", "Gating / debate log", "gating_log"),
        ("portfolio", "Portfolio snapshots", "portfolio_snapshot"),
    ]

    counts: dict[str, int] = {}
    if db.backend == "sqlite":
        import sqlite3
        try:
            with db._conn() as conn:
                for key, _, table in FEEDS:
                    try:
                        if key == "onchain":
                            # Count on-chain snapshots specifically (not all bars).
                            row = conn.execute(
                                "SELECT COUNT(*) FROM market_data "
                                "WHERE symbol LIKE 'ONCHAIN_%'").fetchone()
                        else:
                            row = conn.execute(
                                f"SELECT COUNT(*) FROM {table}").fetchone()
                        counts[table] = int(row[0]) if row else 0
                    except sqlite3.OperationalError:
                        counts[table] = 0
        except Exception:  # noqa: BLE001
            pass

    feeds = []
    for key, label, table in FEEDS:
        n = counts.get(table, 0)
        status = "ok" if n > 0 else ("empty" if table in counts else "missing")
        feeds.append({"key": key, "label": label, "table": table,
                      "count": n, "status": status})

    # DB file size
    db_size_mb = None
    db_path = None
    try:
        import core.db as dbmod
        p = getattr(dbmod, "DB_PATH", None) or getattr(db, "path", None)
        if p and Path(str(p)).exists():
            db_path = str(p)
            db_size_mb = round(Path(str(p)).stat().st_size / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        pass

    # Containerized services (not running — no Docker on this machine)
    services = [
        {"name": "TimescaleDB", "port": 5432, "running": False},
        {"name": "Qdrant", "port": 6333, "running": False},
        {"name": "Neo4j", "port": 7687, "running": False},
        {"name": "Redis", "port": 6379, "running": False},
        {"name": "Prometheus", "port": 9090, "running": False},
        {"name": "Grafana", "port": 3000, "running": False},
    ]
    import socket
    for s in services:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.3)
        try:
            s["running"] = sock.connect_ex(("127.0.0.1", s["port"])) == 0
        except Exception:  # noqa: BLE001
            s["running"] = False
        finally:
            sock.close()

    return {
        "backend": db.backend,
        "db_path": db_path,
        "db_size_mb": db_size_mb,
        "feeds": feeds,
        "services": services,
        "llm_spend": None,
        "note": "Neo4j/Qdrant/Timescale require Docker (`docker compose up -d`).",
    }


def _route_onchain(db) -> dict:
    """Dune on-chain snapshots: latest rows per configured query + freshness."""
    if db.backend != "sqlite":
        return {"queries": [], "note": "onchain snapshots require SQLite dev mode"}
    import sqlite3
    names = [
        "dex_volume_by_pair_7d",
        "dex_volume_daily_7d",
        "dex_volume_by_blockchain_7d",
        "dex_volume_by_protocol_7d",
    ]
    queries = []
    try:
        with db._conn() as conn:
            for name in names:
                row = conn.execute(
                    "SELECT time, raw FROM market_data "
                    "WHERE symbol=? ORDER BY time DESC LIMIT 1",
                    (f"ONCHAIN_{name.upper()}",)).fetchone()
                if not row:
                    queries.append({"name": name, "count": 0, "rows": [],
                                    "stored_at": None})
                    continue
                import json
                try:
                    rows = json.loads(row["raw"])
                except (TypeError, ValueError):
                    rows = []
                queries.append({"name": name, "count": len(rows),
                                "rows": rows, "stored_at": row["time"]})
    except sqlite3.OperationalError:
        return {"queries": [], "note": "market_data table not available"}
    return {"source": "dune", "queries": queries}


def _route_benchmark() -> dict:
    """Model benchmark leaderboard.

    Primary source is the canonical full walk-forward results
    (data/benchmark/leaderboard_full.json, 12 models). Falls back to merging
    whatever per-run benchmark files exist so the page never 500s.
    """
    import json as _json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    full = root / "data" / "benchmark" / "leaderboard_full.json"

    def _load(p: Path):
        try:
            return _json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    if full.exists():
        return _load(full) or {"summary": [], "note": "benchmark data unreadable"}

    # Fallback: merge results.json summary + leaderboard.json summary.
    summary = []
    sources = ["results.json", "leaderboard.json"]
    for name in sources:
        d = _load(root / "data" / "benchmark" / name) or {}
        s = d.get("summary")
        if isinstance(s, list):
            summary.extend(s)
    return {
        "mode": "merged_fallback",
        "generated": None,
        "description": "Fallback merge of results.json + leaderboard.json (partial).",
        "summary": summary,
        "note": "leaderboard_full.json not present; showing merged fallback.",
    }


class AgonistesHandler(BaseHTTPRequestHandler):
    def _send(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200,
                   ctype: str = "text/plain; version=0.0.4; charset=utf-8") -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self, method: str, path: str, qs: dict) -> dict | list:
        db = get_storage()

        if path == "/api/screener/top":
            return _route_screener_top(db, qs)

        if path == "/api/screener/market":
            return _route_screener_market(db)

        if path == "/api/screener/universe":
            return _route_screener_universe(db)

        if path == "/api/backtest/report":
            from backtesting.data_loader import get_ohlcv
            from backtesting.engine import BacktestEngine
            from backtesting.regime_tester import full_period_regime_breakdown
            from backtesting.strategies import make_strategy
            symbol = _q(qs, "symbol", "SPY")
            strategy_name = _q(qs, "strategy", "ma_cross")
            ohlcv = get_ohlcv(symbol, db=db)
            if ohlcv.empty:
                return {"error": f"no data for {symbol}"}
            strat = make_strategy(strategy_name)
            strat.fit(ohlcv)
            result = BacktestEngine().run(symbol, "EQUITY_US", ohlcv,
                                          strat.generate_signals(),
                                          strategy_name=strategy_name)
            report = result.report.to_dict()
            report["performance_by_regime"] = full_period_regime_breakdown(
                result.equity_curve)
            return report

        if path == "/api/backtest/equity":
            from backtesting.data_loader import get_ohlcv
            from backtesting.engine import BacktestEngine
            from backtesting.strategies import make_strategy
            symbol = _q(qs, "symbol", "SPY")
            ohlcv = get_ohlcv(symbol, db=db)
            strat = make_strategy("ma_cross")
            strat.fit(ohlcv)
            result = BacktestEngine().run(symbol, "EQUITY_US", ohlcv,
                                          strat.generate_signals())
            return [{"t": str(t.date()), "equity": round(float(e), 2)}
                    for t, e in result.equity_curve.items()]

        if path == "/api/portfolio/snapshot":
            if db.backend == "sqlite":
                import sqlite3
                with db._conn() as conn:
                    row = conn.execute(
                        "SELECT * FROM portfolio_snapshot ORDER BY time DESC LIMIT 1").fetchone()
                return dict(row) if row else {}
            return {}

        if path == "/api/financials":
            return _route_financials(db, qs)

        if path == "/api/model":
            from valuation.cfa_model import build_model
            symbol = _q(qs, "symbol", "AAPL")
            model = build_model(db, symbol)
            if model is None:
                return {"error": f"insufficient 3-statement data for {symbol.upper()}"}
            return model

        if path == "/api/sentiment":
            return _route_sentiment(db, qs)

        if path == "/api/debate/recent":
            if db.backend == "sqlite":
                import sqlite3
                limit = int(_q(qs, "limit", "10"))
                with db._conn() as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT * FROM gating_log ORDER BY time DESC LIMIT ?",
                        (limit,)).fetchall()
                return [dict(r) for r in rows]
            return []

        if path == "/api/monitoring":
            return _route_monitoring(db)

        if path == "/api/benchmark":
            return _route_benchmark()

        if path == "/api/onchain":
            return _route_onchain(db)

        if method == "POST":
            if path == "/api/fundamentals/refresh":
                return _route_refresh_fundamentals(db, qs)
            if path == "/api/sentiment/refresh":
                return _route_refresh_sentiment(db, qs)

        return {"error": f"unknown path {path}"}

    def _redirect(self, location: str, status: int = 301) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    @staticmethod
    def _mime(path: str) -> str:
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        return {
            "html": "text/html; charset=utf-8", "js": "application/javascript",
            "mjs": "application/javascript", "css": "text/css",
            "json": "application/json", "svg": "image/svg+xml",
            "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "ico": "image/x-icon", "woff2": "font/woff2", "woff": "font/woff",
        }.get(ext, "application/octet-stream")

    def _serve_spa(self, path: str) -> bool:
        """Serve the built React app (ui/dist) at '/'; return True if handled.

        The whole agentic-OS dashboard lives under one origin (:8000) — no
        separate dev-server address. Any non-API path that isn't a real file
        falls back to index.html (SPA routing).
        """
        from pathlib import Path
        dist = Path(__file__).resolve().parent.parent / "ui" / "dist"
        if not dist.is_dir():
            return False
        rel = path.lstrip("/") or "index.html"
        # guard against path traversal
        target = (dist / rel).resolve()
        if not str(target).startswith(str(dist.resolve())):
            return False
        if target.is_file():
            body = target.read_bytes()
            ctype = self._mime(str(target))
        else:
            # SPA fallback for client-side routes (e.g. /financials, /valuation)
            idx = dist / "index.html"
            if not idx.is_file():
                return False
            body = idx.read_bytes()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        route = parsed.path
        t0 = time.monotonic()
        try:
            # Prometheus /metrics — served outside /api/ so Prometheus can scrape
            # the app directly on :8000 (matching prometheus.yml target :8000).
            if route == "/metrics":
                _record_duration(route, t0)
                self._send_text(_route_metrics())
                return
            # Unify the old standalone CFA viewer into the app's Valuation page.
            if parsed.path in ("/model", "/model.html"):
                self._redirect("/valuation")
                return
            if not parsed.path.startswith("/api/"):
                if self._serve_spa(parsed.path):
                    _record_duration(route, t0)
                    return
            self._send(self._route("GET", parsed.path, qs))
            _record_duration(route, t0)
        except Exception as e:
            log.exception("GET %s failed", parsed.path)
            self._send({"error": str(e)}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        route = parsed.path
        t0 = time.monotonic()
        try:
            self._send(self._route("POST", parsed.path, qs))
            _record_duration(route, t0)
        except Exception as e:
            log.exception("POST %s failed", parsed.path)
            self._send({"error": str(e)}, status=500)

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), AgonistesHandler)
    log.info("Agonistes API on http://%s:%d (UI: npm run dev on :3001)", host, port)
    _warm_market_cache()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    serve()

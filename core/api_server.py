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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from core.db import get_storage
from core.logging import get_logger

log = get_logger(__name__)


def _q(qs: dict, key: str, default: str) -> str:
    return qs.get(key, [default])[0]


# ── route handlers ──────────────────────────────────────────────────────
def _route_financials(db, qs: dict) -> dict:
    symbol = _q(qs, "symbol", "AAPL").upper()

    from data_ingestion.fundamental_feeds.dcf_scenarios import dcf_bundle

    snap = db.query_latest_fundamentals(symbol) or {}
    if snap:
        snap = {k: v for k, v in snap.items()
                if k not in ("raw_data", "filing_url", "transcript_summary")}
        snap["time"] = str(snap.get("time"))

    # 8-quarter statements, oldest → newest
    statements: dict[str, list] = {}
    for name in ("income", "balance", "cashflow"):
        rows = db.query_financial_statements(symbol, statement=name, quarters=8)
        rows = [{"period": r["period"], **r["data"]} for r in rows]
        rows.sort(key=lambda r: r["period"])
        statements[name] = rows

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
        "statements": statements,
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
        fetch_earnings_dates, refresh_info_snapshot,
    )
    from data_ingestion.fundamental_feeds.yfinance_financials import (
        fetch_company_profile, fetch_quarterly_statements,
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


class AgonistesHandler(BaseHTTPRequestHandler):
    def _send(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self, method: str, path: str, qs: dict) -> dict | list:
        db = get_storage()

        if path == "/api/screener/top":
            from screener.pipeline import run_screener, screener_table
            return screener_table(run_screener(top_n=10, db=db))

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

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        # Unify the old standalone CFA viewer into the app's Valuation page.
        if parsed.path in ("/model", "/model.html"):
            self._redirect("/valuation")
            return
        if not parsed.path.startswith("/api/"):
            if self._serve_spa(parsed.path):
                return
        try:
            self._send(self._route("GET", parsed.path, qs))
        except Exception as e:  # noqa: BLE001
            log.exception("GET %s failed", parsed.path)
            self._send({"error": str(e)}, status=500)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        try:
            self._send(self._route("POST", parsed.path, qs))
        except Exception as e:  # noqa: BLE001
            log.exception("POST %s failed", parsed.path)
            self._send({"error": str(e)}, status=500)

    def log_message(self, fmt, *args):  # noqa: A003
        log.debug(fmt, *args)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), AgonistesHandler)
    log.info("Agonistes API on http://%s:%d (UI: npm run dev on :3001)", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    serve()

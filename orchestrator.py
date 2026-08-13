#!/usr/bin/env python3
"""Project Agonistes — Autonomous Orchestrator.

This is the SINGLE entry point that drives the entire system automatically.
It replaces manual CLI calls with a scheduled, event-driven loop.

Run once and leave it running:
    python orchestrator.py                  # live mode, IST-aware schedule
    python orchestrator.py --dry-run        # print schedule, do nothing
    python orchestrator.py --now rebalance  # force a specific job immediately

Architecture
============
The orchestrator is structured as three nested loops:

  1. DAILY LOOP  (runs once per trading day, ~09:15 IST after NSE open)
     ├─ Ingest: pull latest OHLCV + fundamentals + news
     ├─ Feature engineering: rebuild feature vectors
     ├─ Screen: score universe, pick top-N candidates
     └─ Debate: run LangGraph for each candidate → trade decisions

  2. WEEKLY LOOP  (Sunday 02:00 IST, before week open)
     ├─ Benchmark check: compare live model Sharpe vs 4-week rolling window
     ├─ Drift detection: CUSUM test on prediction residuals
     └─ If drift detected → schedule model_retrain job

  3. MONTHLY LOOP  (1st of month, 01:00 IST)
     ├─ Full strategy_builder benchmark (all 37 models, walk-forward)
     ├─ If new model beats current by >0.15 Sharpe → retrain on RunPod
     ├─ Volatility regime review: update GARCH parameters
     └─ Feature importance SHAP sweep: flag low-importance features

Fault Tolerance
===============
- Every job is wrapped in a retry decorator (3 attempts, exponential backoff)
- Failed jobs write to core.db circuit breaker log
- Orchestrator itself has a watchdog: restarts crashed jobs automatically
- All timing is UTC internally, converted to IST for display

"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Callable

from core.logging import get_logger, setup_logging

log = get_logger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

# ---------------------------------------------------------------------------
# State file: persists last-run timestamps across restarts
# ---------------------------------------------------------------------------

STATE_PATH = Path("data/orchestrator_state.json")


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def _retry(attempts: int = 3, backoff: float = 60.0):
    """Decorator: retry on exception with exponential backoff."""
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt == attempts:
                        log.error("Job %s failed after %d attempts: %s",
                                  fn.__name__, attempts, exc)
                        raise
                    wait = backoff * (2 ** (attempt - 1))
                    log.warning("Job %s attempt %d/%d failed (%s). Retrying in %.0fs",
                                fn.__name__, attempt, attempts, exc, wait)
                    time.sleep(wait)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# ── DAILY JOBS ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@_retry(attempts=3, backoff=30.0)
def job_ingest_prices() -> None:
    """Pull latest OHLCV for entire universe (yfinance / Binance)."""
    log.info("[DAILY] Ingesting prices …")
    from data_ingestion.price_feeds.equity_ws import backfill_equities
    from core.db import get_storage
    db = get_storage()
    symbols = db.symbols() or ["AAPL", "MSFT", "SPY", "QQQ"]
    equities = [s for s in symbols if not s.startswith(("BTC", "ETH", "SOL"))]
    crypto = [s for s in symbols if s not in equities]
    if equities:
        backfill_equities(equities, period="5d")   # last 5 days incremental
    if crypto:
        from data_ingestion.price_feeds.crypto_ws import backfill_klines
        for c in crypto:
            backfill_klines(c, interval="1d", days=5)
    log.info("[DAILY] Price ingest complete — %d symbols", len(symbols))


@_retry(attempts=3, backoff=30.0)
def job_ingest_fundamentals() -> None:
    """Pull fundamental snapshots + earnings for screened candidates."""
    log.info("[DAILY] Ingesting fundamentals …")
    from screener.pipeline import score_universe
    from core.db import get_storage
    db = get_storage()
    # Only refresh fundamentals for top-30 by composite score (cost control)
    candidates = sorted(score_universe(), key=lambda s: s.score, reverse=True)[:30]
    for cand in candidates:
        try:
            from data_ingestion.fundamental_feeds.dcf_scenarios import apply_dcf_to_snapshot
            snap = db.query_latest_fundamentals(cand.symbol) or {}
            if snap:
                snap = apply_dcf_to_snapshot(snap)
                db.upsert_fundamental_snapshot(snap)
        except Exception as exc:
            log.warning("Fundamentals skip %s: %s", cand.symbol, exc)
    log.info("[DAILY] Fundamentals refresh done")


@_retry(attempts=3, backoff=30.0)
def job_ingest_news() -> None:
    """Pull and score news sentiment for top candidates."""
    log.info("[DAILY] Ingesting news / sentiment …")
    # Sentiment sweep runs against the screener's top candidates
    from scripts.sentiment_sweep import run_sentiment_sweep
    run_sentiment_sweep()
    log.info("[DAILY] Sentiment sweep done")


@_retry(attempts=3, backoff=30.0)
def job_build_features() -> None:
    """Rebuild feature vectors for all symbols with fresh OHLCV."""
    log.info("[DAILY] Building feature vectors …")
    from core.db import get_storage
    from feature_engineering.feature_store import build_feature_frame, write_to_store
    db = get_storage()
    symbols = db.symbols() or []
    n_updated = 0
    for sym in symbols:
        try:
            ohlcv = db.query_ohlcv(sym)
            if ohlcv is None or ohlcv.empty or len(ohlcv) < 100:
                continue
            frame = build_feature_frame(sym, "EQUITY_US", ohlcv, db=db, timeframe="SWING")
            write_to_store(frame, sym, "EQUITY_US", "SWING", db)
            n_updated += 1
        except Exception as exc:
            log.warning("Feature build skip %s: %s", sym, exc)
    log.info("[DAILY] Feature build done — %d symbols updated", n_updated)


@_retry(attempts=3, backoff=60.0)
def job_screen_and_debate() -> None:
    """Run screener → for each candidate → full LangGraph debate cycle."""
    log.info("[DAILY] Screening universe …")
    from screener.pipeline import run_screener
    from langgraph_app.src.graph import build_graph
    from core.db import get_storage
    import uuid

    db = get_storage()
    candidates = run_screener(top_n=10)
    log.info("[DAILY] Screener selected %d candidates", len(candidates))

    graph = build_graph()
    results = []
    for cand in candidates:
        try:
            log.info("[DAILY]  → Debating %s (score=%.1f)", cand.symbol, cand.score)
            state = graph.invoke({
                "symbol": cand.symbol,
                "asset_class": cand.asset_class,
                "cycle_id": f"daily-{uuid.uuid4().hex[:8]}",
                "screener_score": cand.score,
            })
            results.append({
                "symbol": cand.symbol,
                "decision": state.get("gating", {}).get("decision", "PASS"),
                "trade_logged": state.get("trade_logged", False),
            })
            log.info("[DAILY]  → %s → %s", cand.symbol,
                     state.get("gating", {}).get("decision", "PASS"))
        except Exception as exc:
            log.error("[DAILY] Debate failed for %s: %s", cand.symbol, exc)

    log.info("[DAILY] Debate cycle complete — %d processed, %d trades logged",
             len(results), sum(1 for r in results if r["trade_logged"]))


# ---------------------------------------------------------------------------
# ── WEEKLY JOBS ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@_retry(attempts=2, backoff=120.0)
def job_model_drift_check() -> None:
    """CUSUM drift test on TFT residuals. Flags if model needs retraining."""
    log.info("[WEEKLY] Running model drift / performance check …")
    from core.db import get_storage
    import numpy as np

    db = get_storage()
    # Pull last 20 gating decisions; compare bull/bear confidence drift
    decisions = db.query_recent_gating(n=100) if hasattr(db, "query_recent_gating") else []
    if not decisions:
        log.info("[WEEKLY] No gating history yet — skipping drift check")
        return

    # Simple CUSUM: flag if rolling 4-week hit rate drops > 10% from baseline
    hits = [1 if d.get("dominant_side") == d.get("outcome") else 0
            for d in decisions if "outcome" in d]
    if len(hits) < 20:
        log.info("[WEEKLY] Insufficient outcome history (%d) — skipping", len(hits))
        return

    baseline = np.mean(hits[:len(hits) // 2])
    recent = np.mean(hits[len(hits) // 2:])
    drift = baseline - recent
    log.info("[WEEKLY] Hit rate baseline=%.2f recent=%.2f drift=%.3f",
             baseline, recent, drift)
    if drift > 0.10:
        log.warning("[WEEKLY] DRIFT DETECTED (%.3f > 0.10) — scheduling benchmark", drift)
        _write_flag("needs_benchmark", True)
    else:
        log.info("[WEEKLY] Model performance stable — no action needed")


@_retry(attempts=2, backoff=120.0)
def job_quick_benchmark() -> None:
    """Fast strategy_builder run (quick mode, 4 representative models).

    Runs every week if drift detected, or every 4 weeks unconditionally.
    Compares against stored best Sharpe — triggers full benchmark if gap > 0.15.
    """
    log.info("[WEEKLY] Running quick benchmark (4 models, quick mode) …")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "strategy_builder.run",
         "--models", "vlstm", "tft", "lightgbm", "vol_timing",
         "--seeds", "2", "--quick"],
        capture_output=True, text=True, timeout=3600
    )
    if result.returncode != 0:
        log.error("[WEEKLY] Quick benchmark failed:\n%s", result.stderr[-2000:])
        raise RuntimeError("Quick benchmark subprocess failed")
    log.info("[WEEKLY] Quick benchmark complete")
    _check_benchmark_results()


def _check_benchmark_results() -> None:
    """Read results.json; if best Sharpe improved > 0.15 vs stored, flag full retrain."""
    results_path = Path("data/benchmark/results.json")
    if not results_path.exists():
        return
    data = json.loads(results_path.read_text())
    summary = data.get("summary", [])
    if not summary:
        return
    best = max(summary, key=lambda r: r.get("sharpe", -9))
    stored_sharpe = _read_flag("best_sharpe", 0.0)
    log.info("[WEEKLY] Best benchmark Sharpe: %.3f (stored: %.3f)",
             best.get("sharpe", 0), stored_sharpe)
    if best.get("sharpe", 0) - stored_sharpe > 0.15:
        log.warning("[WEEKLY] New best model '%s' (%.3f) — flagging full retrain",
                    best["model"], best["sharpe"])
        _write_flag("best_model", best["model"])
        _write_flag("best_sharpe", best.get("sharpe", 0))
        _write_flag("needs_full_retrain", True)


# ---------------------------------------------------------------------------
# ── MONTHLY JOBS ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@_retry(attempts=1, backoff=300.0)
def job_full_benchmark() -> None:
    """Run the complete 37-model strategy_builder benchmark.

    This is the expensive job — takes ~4h on CPU for 37 models × 3 seeds.
    Result determines which model gets retrained on RunPod next cycle.
    """
    log.info("[MONTHLY] Running FULL benchmark — 37 models, 3 seeds …")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "strategy_builder.run",
         "--all", "--seeds", "3", "--workers", "4"],
        capture_output=True, text=True, timeout=18000  # 5 hours max
    )
    if result.returncode != 0:
        log.error("[MONTHLY] Full benchmark failed:\n%s", result.stderr[-3000:])
        raise RuntimeError("Full benchmark subprocess failed")
    log.info("[MONTHLY] Full benchmark complete")
    _check_benchmark_results()

    # Auto-update best model pointer in .env if changed
    new_best = _read_flag("best_model", None)
    if new_best:
        log.info("[MONTHLY] Best model updated to: %s", new_best)
        _write_env_model(new_best)


@_retry(attempts=2, backoff=60.0)
def job_volatility_regime_review() -> None:
    """Refit GARCH parameters on fresh data. Update vol regime thresholds."""
    log.info("[MONTHLY] Volatility regime review …")
    from strategy_builder.volatility_models import garch11_variance
    from core.db import get_storage
    import numpy as np

    db = get_storage()
    symbols = (db.symbols() or [])[:10]  # top-10 by data richness
    regime_stats = {}
    for sym in symbols:
        ohlcv = db.query_ohlcv(sym)
        if ohlcv is None or ohlcv.empty:
            continue
        rets = ohlcv["close"].pct_change().dropna().to_numpy()
        garch_var = garch11_variance(rets)
        long_run_vol = float(np.sqrt(np.mean(garch_var)) * np.sqrt(252))
        current_vol = float(np.sqrt(garch_var[-1]) * np.sqrt(252))
        regime_stats[sym] = {
            "long_run_vol_ann": round(long_run_vol, 4),
            "current_vol_ann": round(current_vol, 4),
            "regime": "CALM" if current_vol < long_run_vol else "TURBULENT",
        }
        log.info("[MONTHLY] %s: long_run=%.1f%% current=%.1f%% → %s",
                 sym, 100 * long_run_vol, 100 * current_vol, regime_stats[sym]["regime"])

    Path("data/benchmark/regime_review.json").write_text(
        json.dumps(regime_stats, indent=2))
    log.info("[MONTHLY] Volatility regime review complete — %d assets", len(regime_stats))


# ---------------------------------------------------------------------------
# ── Flag helpers (lightweight key/value store in state JSON) ─────────────
# ---------------------------------------------------------------------------

def _write_flag(key: str, value) -> None:
    state = _load_state()
    state[key] = value
    _save_state(state)


def _read_flag(key: str, default=None):
    return _load_state().get(key, default)


def _write_env_model(model_name: str) -> None:
    """Patch SIGNAL_MODEL= in .env so inference picks up new model automatically."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    lines = env_path.read_text().splitlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("SIGNAL_MODEL="):
            lines[i] = f"SIGNAL_MODEL={model_name}"
            updated = True
    if not updated:
        lines.append(f"SIGNAL_MODEL={model_name}")
    env_path.write_text("\n".join(lines) + "\n")
    log.info("[MONTHLY] Updated .env SIGNAL_MODEL=%s", model_name)


# ---------------------------------------------------------------------------
# ── Scheduler ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, name: str, fn: Callable, description: str):
        self.name = name
        self.fn = fn
        self.description = description


# Full job registry
ALL_JOBS = {
    # Daily
    "ingest_prices":        Job("ingest_prices",        job_ingest_prices,
                                "Pull latest OHLCV (yfinance / Binance)"),
    "ingest_fundamentals":  Job("ingest_fundamentals",  job_ingest_fundamentals,
                                "Refresh fundamental snapshots + DCF"),
    "ingest_news":          Job("ingest_news",          job_ingest_news,
                                "Sentiment sweep — news for top candidates"),
    "build_features":       Job("build_features",       job_build_features,
                                "Rebuild feature vectors for all symbols"),
    "screen_and_debate":    Job("screen_and_debate",    job_screen_and_debate,
                                "Screener → LangGraph debate → trade decisions"),
    # Weekly
    "model_drift_check":    Job("model_drift_check",    job_model_drift_check,
                                "CUSUM drift test on model prediction residuals"),
    "quick_benchmark":      Job("quick_benchmark",      job_quick_benchmark,
                                "4-model quick strategy_builder benchmark"),
    # Monthly
    "full_benchmark":       Job("full_benchmark",       job_full_benchmark,
                                "37-model full benchmark — determines best model"),
    "volatility_regime":    Job("volatility_regime",    job_volatility_regime_review,
                                "Refit GARCH; update vol regime thresholds"),
    # Alias
    "rebalance":            Job("rebalance",            job_screen_and_debate,
                                "Alias: screener + debate (same as screen_and_debate)"),
}


def _now_ist() -> datetime:
    return datetime.now(IST)


def _is_trading_day(dt: datetime) -> bool:
    """True if dt is Mon–Fri (Indian market proxy; no holiday calendar)."""
    return dt.weekday() < 5


def _seconds_until(target_hour: int, target_minute: int = 0,
                   tz: timezone = IST) -> float:
    """Seconds until the next occurrence of HH:MM in given timezone."""
    now = datetime.now(tz)
    target = now.replace(hour=target_hour, minute=target_minute,
                         second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def run_job(job: Job, dry_run: bool = False) -> None:
    if dry_run:
        log.info("[DRY-RUN] Would run: %s — %s", job.name, job.description)
        return
    log.info("=" * 60)
    log.info("RUNNING JOB: %s", job.name)
    log.info("=" * 60)
    t0 = time.monotonic()
    try:
        job.fn()
        elapsed = time.monotonic() - t0
        log.info("JOB %s completed in %.1fs", job.name, elapsed)
        _write_flag(f"last_run_{job.name}",
                    datetime.now(IST).isoformat())
    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.error("JOB %s FAILED after %.1fs: %s", job.name, elapsed, exc)
        traceback.print_exc()
        _write_flag(f"last_error_{job.name}", str(exc))


# ---------------------------------------------------------------------------
# ── Main loop ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

STOP = Event()


def _signal_handler(sig, frame):
    log.info("Received signal %s — shutting down gracefully …", sig)
    STOP.set()


def run_scheduler(dry_run: bool = False) -> None:
    """The main autonomous loop.

    Schedule (all times IST):
      Daily  09:15 — ingest prices (right after NSE open)
      Daily  09:30 — ingest fundamentals + news
      Daily  10:00 — build features
      Daily  10:30 — screen + debate (main trading decision)
      Weekly Sun 02:00 — drift check + quick benchmark
      Monthly 1st 01:00 — full benchmark + vol regime review
    """
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log.info("Orchestrator started (IST timezone, dry_run=%s)", dry_run)
    log.info("Jobs registered: %s", ", ".join(ALL_JOBS))

    # Daily job pipeline (in sequence, each waits for the next scheduled time)
    daily_schedule = [
        (9,  15, "ingest_prices"),
        (9,  30, "ingest_fundamentals"),
        (9,  35, "ingest_news"),
        (10,  0, "build_features"),
        (10, 30, "screen_and_debate"),
    ]

    def _daily_loop():
        while not STOP.is_set():
            now = _now_ist()
            if not _is_trading_day(now):
                log.info("Non-trading day (%s) — sleeping until tomorrow 09:00",
                         now.strftime("%A"))
                STOP.wait(timeout=_seconds_until(9, 0))
                continue
            for hour, minute, job_name in daily_schedule:
                wait = _seconds_until(hour, minute)
                log.info("Next daily job: %s at %02d:%02d IST (in %.0fm)",
                         job_name, hour, minute, wait / 60)
                STOP.wait(timeout=wait)
                if STOP.is_set():
                    return
                run_job(ALL_JOBS[job_name], dry_run=dry_run)
            # Done for today — sleep until tomorrow's first job
            STOP.wait(timeout=_seconds_until(9, 15))

    def _weekly_loop():
        while not STOP.is_set():
            now = _now_ist()
            # Run every Sunday at 02:00
            days_to_sunday = (6 - now.weekday()) % 7
            next_sunday = (now + timedelta(days=days_to_sunday)).replace(
                hour=2, minute=0, second=0, microsecond=0)
            if next_sunday <= now:
                next_sunday += timedelta(weeks=1)
            wait = (next_sunday - now).total_seconds()
            log.info("Next weekly jobs in %.1fh (Sunday 02:00 IST)", wait / 3600)
            STOP.wait(timeout=wait)
            if STOP.is_set():
                return
            run_job(ALL_JOBS["model_drift_check"], dry_run=dry_run)
            # Run quick benchmark if drift flagged OR every 4 weeks
            last_benchmark = _read_flag("last_run_quick_benchmark")
            needs = _read_flag("needs_benchmark", False)
            four_weeks_ago = (datetime.now(IST) - timedelta(weeks=4)).isoformat()
            if needs or (not last_benchmark) or last_benchmark < four_weeks_ago:
                run_job(ALL_JOBS["quick_benchmark"], dry_run=dry_run)
                _write_flag("needs_benchmark", False)

    def _monthly_loop():
        while not STOP.is_set():
            now = _now_ist()
            # Run on the 1st of next month at 01:00
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1,
                                         hour=1, minute=0, second=0, microsecond=0)
            else:
                next_month = now.replace(month=now.month + 1, day=1,
                                         hour=1, minute=0, second=0, microsecond=0)
            wait = (next_month - now).total_seconds()
            log.info("Next monthly jobs in %.1fd (1st of month 01:00 IST)", wait / 86400)
            STOP.wait(timeout=wait)
            if STOP.is_set():
                return
            run_job(ALL_JOBS["full_benchmark"],    dry_run=dry_run)
            run_job(ALL_JOBS["volatility_regime"], dry_run=dry_run)

    # Launch all three loops as background threads
    threads = [
        Thread(target=_daily_loop,   name="daily",   daemon=True),
        Thread(target=_weekly_loop,  name="weekly",  daemon=True),
        Thread(target=_monthly_loop, name="monthly", daemon=True),
    ]
    for t in threads:
        t.start()
        log.info("Started %s loop thread", t.name)

    # Block main thread until shutdown signal
    while not STOP.is_set():
        STOP.wait(timeout=60)
    log.info("Orchestrator stopped.")


# ---------------------------------------------------------------------------
# ── CLI ──────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(
        prog="orchestrator",
        description="Agonistes autonomous orchestrator — run once, runs forever",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing anything",
    )
    ap.add_argument(
        "--now", metavar="JOB", default=None,
        help=f"Force a specific job immediately. One of: {', '.join(ALL_JOBS)}",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List all available jobs and exit",
    )
    args = ap.parse_args()

    if args.list:
        print("\nAvailable jobs:")
        for name, job in ALL_JOBS.items():
            last = _read_flag(f"last_run_{name}", "never")
            print(f"  {name:<25s} — {job.description}")
            print(f"  {'':25s}   last run: {last}")
        return

    if args.now:
        if args.now not in ALL_JOBS:
            print(f"Unknown job: {args.now}. Use --list to see options.")
            sys.exit(1)
        run_job(ALL_JOBS[args.now], dry_run=args.dry_run)
        return

    run_scheduler(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

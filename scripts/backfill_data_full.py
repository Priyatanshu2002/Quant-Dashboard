#!/usr/bin/env python3
"""Full-data backfill for the whole screener universe.

For every symbol, fetch and persist the real backend data + calculations the UI
depends on — not just prices:
  1. Company profile (name, sector, industry)  -> company_profiles  (screener names)
  2. Fundamental snapshot + DCF (market cap, P/E, intrinsic value) -> fundamental_snapshots
  3. Quarterly 3-statement (income/balance/cashflow) for US/IN equities -> financial_statements
  4. LLM analyst verdicts for the top candidates by score (cost-controlled)

Idempotent — safe to re-run. Skips crypto/forex/bonds for equity fundamentals.

Usage: .venv/Scripts/python scripts/backfill_data_full.py [--statements-only] [--limit N]
"""
from __future__ import annotations

import argparse
import time

from core.db import get_storage
from core.logging import get_logger, setup_logging
from screener.asset_universe import get_universe

log = get_logger(__name__)

FUNDAMENTAL_CLASSES = ("EQUITY_US", "EQUITY_IN", "ETF")
STATEMENT_CLASSES = ("EQUITY_US", "EQUITY_IN")
POLLITE_S = 0.25


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--statements-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    db = get_storage()
    universe = get_universe()
    symbols = universe.symbols()
    if args.limit:
        symbols = symbols[: args.limit]
    log.info("Backfilling data for %d universe symbols", len(symbols))

    t0 = time.time()
    n_profile = n_snap = n_stmt = 0

    if not args.statements_only:
        from data_ingestion.fundamental_feeds.dcf_scenarios import apply_dcf_to_snapshot
        from data_ingestion.fundamental_feeds.yfinance_earnings import refresh_info_snapshot
        from data_ingestion.fundamental_feeds.yfinance_financials import fetch_company_profile

        for sym in symbols:
            cls = universe.asset_class_of(sym) or ""
            # 1) company profile (name/sector) for everything we can
            try:
                if fetch_company_profile(sym, storage=db):
                    n_profile += 1
            except Exception as e:  # noqa: BLE001
                log.debug("profile %s failed: %s", sym, e)
            # 2) fundamental snapshot + DCF for equities/ETFs
            if cls in FUNDAMENTAL_CLASSES:
                try:
                    snap = refresh_info_snapshot(sym, storage=db) or {}
                    if snap:
                        apply_dcf_to_snapshot(snap)
                        if snap.get("dcf_intrinsic_value"):
                            db.upsert_fundamental_snapshot(snap)
                            n_snap += 1
                except Exception as e:  # noqa: BLE001
                    log.debug("info %s failed: %s", sym, e)
            time.sleep(POLLITE_S)
            if (n_profile + n_snap) % 50 == 0 and n_profile + n_snap:
                log.info("… %d profiles, %d snapshots after %ds",
                         n_profile, n_snap, int(time.time() - t0))

    # 3) quarterly 3-statement for equities
    from data_ingestion.fundamental_feeds.yfinance_financials import fetch_quarterly_statements
    stmt_targets = [s for s in symbols if universe.asset_class_of(s) in STATEMENT_CLASSES]
    for sym in stmt_targets:
        try:
            stmts = fetch_quarterly_statements(sym, quarters=8, storage=db)
            n_stmt += sum(len(v) for v in stmts.values())
        except Exception as e:  # noqa: BLE001
            log.debug("statements %s failed: %s", sym, e)
        time.sleep(POLLITE_S)
        if n_stmt and n_stmt % 1000 == 0:
            log.info("… %d statement rows after %ds", n_stmt, int(time.time() - t0))

    # 4) LLM analyst for top candidates by composite score (cost control)
    try:
        from screener.pipeline import score_universe
        from data_ingestion.sentiment_feeds.llm_analyst import analyze_fundamentals
        top = sorted(score_universe(db), key=lambda s: s.composite_score, reverse=True)[:30]
        for s in top:
            try:
                analyze_fundamentals(s.symbol, storage=db, db=db, force=True)
            except Exception as e:  # noqa: BLE001
                log.debug("LLM analyst %s failed: %s", s.symbol, e)
    except Exception as e:  # noqa: BLE001
        log.warning("LLM analyst sweep failed: %s", e)

    log.info("DONE in %.0fs: %d profiles, %d fundamentals+dcf, %d statement rows",
             time.time() - t0, n_profile, n_snap, n_stmt)


if __name__ == "__main__":
    main()

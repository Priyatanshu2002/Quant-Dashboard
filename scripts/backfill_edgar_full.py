#!/usr/bin/env python3
"""Full-universe SEC EDGAR statement backfill — deep multi-year history.

For every US equity in the screener universe, download the SEC XBRL company-facts
feed and persist the full annual (10-K) + quarterly (10-Q) statement series to
financial_statements (period_type ANNUAL / QUARTERLY). This is the authoritative,
complete-history source (vs yfinance's ~5 quarters) that CFA-standard analysis
requires.

Usage: .venv/Scripts/python scripts/backfill_edgar_full.py [--limit N]
"""
from __future__ import annotations

import time

from core.db import get_storage
from core.logging import get_logger, setup_logging
from screener.asset_universe import get_universe

log = get_logger(__name__)

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
HEADERS = {"User-Agent": "Project Agonistes research@agonistes.local"}


def main() -> None:
    setup_logging()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="only first N symbols (for testing)")
    args = p.parse_args()

    db = get_storage()
    universe = get_universe()
    us_equities = [s for s in universe.symbols() if universe.asset_class_of(s) == "EQUITY_US"]
    if args.limit:
        us_equities = us_equities[: args.limit]

    # ticker → CIK for the whole US market (authoritative SEC mapping)
    from data_ingestion.fundamental_feeds.sec_edgar_watcher import load_cik_map
    cik_map = load_cik_map()

    from data_ingestion.fundamental_feeds.three_statement_parser import parse_company_facts_series

    done, failed = 0, 0
    total_rows = 0
    t0 = time.time()
    for i, sym in enumerate(us_equities, 1):
        cik = cik_map.get(sym.upper())
        if not cik:
            failed += 1
            log.info("[%d/%d] %s: no CIK mapping, skipped", i, len(us_equities), sym)
            continue
        try:
            series = parse_company_facts_series(cik, sym)
            n = 0
            for st, rows in series.items():
                # Only repopulate ANNUAL (10-K) rows with the full expanded line
                # items. Quarterly (10-Q) is left to the complete yfinance data so
                # the Quarterly view is never regressed by sparser SEC tagging.
                annual = [r for r in rows if r.get("period_type") == "ANNUAL"]
                if annual:
                    db.write_financial_statements(sym, st, annual)
                    n += len(annual)
            total_rows += n
            done += 1
            log.info("[%d/%d] %s: wrote %d annual rows", i, len(us_equities), sym, n)
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.warning("[%d/%d] %s failed: %s", i, len(us_equities), sym, e)
        time.sleep(0.5)  # be polite to SEC (≤ ~2 req/s)

    log.info("EDGAR backfill complete in %.0fs: %d ok, %d failed, %d statement rows",
             time.time() - t0, done, failed, total_rows)


if __name__ == "__main__":
    main()

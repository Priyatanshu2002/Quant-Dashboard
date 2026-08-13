#!/usr/bin/env python3
"""Expand the screener universe to the full S&P 500 (keeps other asset classes).

Fetches current S&P 500 constituents from Wikipedia and rewrites the EQUITY_US
section of screener/screener_config.yaml with all of them (symbol/name/sector).

Usage: .venv/Scripts/python scripts/expand_universe.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
import yaml

from core.logging import get_logger, setup_logging

log = get_logger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "screener" / "screener_config.yaml"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500() -> list[dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    }
    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0]
    out = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        out.append({
            "symbol": sym,
            "name": str(row["Security"]).strip(),
            "sector": str(row["GICS Sector"]).strip(),
        })
    return out


def main() -> None:
    setup_logging()
    sp500 = fetch_sp500()
    log.info("Fetched %d S&P 500 constituents", len(sp500))

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    universe = cfg.get("universe", {})
    # keep existing curated symbols that aren't already in the S&P 500 list
    existing_us = universe.get("EQUITY_US", [])
    have = {e["symbol"].upper() for e in sp500}
    curated = [e for e in existing_us if e.get("symbol", "").upper() not in have]
    universe["EQUITY_US"] = curated + sp500
    cfg["universe"] = universe

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    log.info("Wrote %d EQUITY_US instruments to %s (total %d universe)",
             len(universe["EQUITY_US"]), CONFIG_PATH.name,
             sum(len(v) for v in universe.values()))


if __name__ == "__main__":
    main()

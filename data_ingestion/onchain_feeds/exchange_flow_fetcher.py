"""On-chain / exchange-flow fetcher — CoinGecko free API.

Provides BTC dominance, total crypto market-cap change, and top-exchange
volume approximations. No key required (rate-limited to ~10-30 req/min).
"""
from __future__ import annotations

import datetime as dt

import requests

from core.db import Storage, get_storage
from core.events import EVENT_ONCHAIN, EVENT_MACRO, bus
from core.logging import get_logger

log = get_logger(__name__)

API = "https://api.coingecko.com/api/v3"
HEADERS = {"Accept": "application/json"}


def fetch_crypto_global(storage: Storage | None = None) -> dict | None:
    """BTC dominance + 24h total mcap change → macro snapshot fields."""
    storage = storage or get_storage()
    try:
        resp = requests.get(f"{API}/global", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        macro = {
            "ts": dt.datetime.utcnow(),
            "btc_dominance": data.get("market_cap_percentage", {}).get("btc"),
            "crypto_total_mcap_chg_24h": data.get("market_cap_change_percentage_24h_usd"),
        }
        storage.write_macro_snapshot(macro)
        import asyncio
        asyncio.run(bus.publish(EVENT_MACRO, macro))
        return macro
    except Exception as e:  # noqa: BLE001
        log.warning("CoinGecko global fetch failed: %s", e)
        return None


def fetch_exchange_flows(storage: Storage | None = None) -> dict | None:
    """Approximate exchange-flow signal from top-10 exchange 24h volumes."""
    storage = storage or get_storage()
    try:
        resp = requests.get(f"{API}/exchanges",
                            params={"per_page": 10, "page": 1},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        exchanges = resp.json()
        total = sum(float(e.get("trade_volume_24h_btc") or 0) for e in exchanges)
        snap = {
            "ts": dt.datetime.utcnow(),
            "top10_exchange_volume_btc_24h": total,
            "exchange_count": len(exchanges),
        }
        import asyncio
        asyncio.run(bus.publish(EVENT_ONCHAIN, snap))
        return snap
    except Exception as e:  # noqa: BLE001
        log.warning("CoinGecko exchange fetch failed: %s", e)
        return None


if __name__ == "__main__":
    print(fetch_crypto_global())
    print(fetch_exchange_flows())

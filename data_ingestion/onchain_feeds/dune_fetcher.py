"""Dune Analytics fetcher (free tier) — requires DUNE_API_KEY in .env.

Pattern: create a query in Dune, copy its query ID here, run once a day.
Results are stored raw; the exchange_flow_fetcher computes derived features.
"""
from __future__ import annotations

import datetime as dt

import requests

from core.config import DUNE_API_KEY
from core.db import Storage
from core.events import EVENT_ONCHAIN, bus
from core.logging import get_logger

log = get_logger(__name__)

API = "https://api.dune.com/api/v1"

# Example: BTC exchange netflow query (create your own and set the ID).
QUERY_IDS: dict[str, int] = {
    "btc_exchange_netflow": 0,   # TODO: set your Dune query ID
}


def run_dune_query(name: str, storage: Storage | None = None) -> dict | None:
    if not DUNE_API_KEY:
        log.info("DUNE_API_KEY not set — skipping Dune fetch")
        return None
    qid = QUERY_IDS.get(name)
    if not qid:
        log.info("No Dune query ID configured for %s", name)
        return None
    try:
        headers = {"X-Dune-API-Key": DUNE_API_KEY}
        job = requests.post(f"{API}/query/{qid}/execute", headers=headers, timeout=30).json()
        job_id = job.get("execution_id")
        results = requests.get(f"{API}/execution/{job_id}/results",
                               headers=headers, timeout=60).json()
        rows = (results.get("result", {}) or {}).get("rows", [])
        snap = {"ts": dt.datetime.utcnow(), "source": f"dune:{name}", "rows": rows}
        storage.write_macro_snapshot({"ts": snap["ts"]})
        import asyncio
        asyncio.run(bus.publish(EVENT_ONCHAIN, snap))
        return snap
    except Exception as e:  # noqa: BLE001
        log.warning("Dune query %s failed: %s", name, e)
        return None


if __name__ == "__main__":
    print(run_dune_query("btc_exchange_netflow"))

"""Dune Analytics fetcher — modern API, SQL-direct mode.

Requires DUNE_API_KEY in .env (free at https://dune.com → API Keys).

Two modes:

1. SQL-direct (default, recommended for free tier):
   Dune's `POST /api/v1/sql/execute` runs arbitrary SQL with no saved query
   object, so you don't need to build anything in the Dune editor. Queries live
   in the SQL_QUERIES map below (all on the PUBLIC curated tables — `dex.trades`
   etc.). Free-plan note: the premium/gated datasets (stablecoins, balances,
   activity-enriched) are NOT accessible on the free tier.

   Run everything:
       python data_ingestion/onchain_feeds/dune_fetcher.py

2. Query-ID mode (saved queries you own/shared in Dune):
   Set DUNE_QUERY_IDS in .env (name=id,name=id), e.g.
       DUNE_QUERY_IDS=btc_exchange_netflow=1234567

Each execution is submitted, polled to a terminal state, and result rows are
persisted to market_data.raw (via Storage.write_onchain_snapshot) + published
on the onchain event bus.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import requests

from core.config import DUNE_API_KEY
from core.db import Storage, get_storage
from core.events import EVENT_ONCHAIN, bus
from core.logging import get_logger

log = get_logger(__name__)

API = "https://api.dune.com/api/v1"

# ── SQL-direct queries (public curated tables, free-plan safe) ──────────
# Each is a lightweight aggregate that returns a small row set for storage.
SQL_QUERIES: dict[str, str] = {
    "dex_volume_by_pair_7d": """
        SELECT blockchain, token_pair,
               SUM(CAST(amount_usd AS DOUBLE)) AS volume_usd,
               COUNT(*) AS trades
        FROM dex.trades
        WHERE block_time > NOW() - INTERVAL '7' day
          AND token_pair IS NOT NULL
        GROUP BY blockchain, token_pair
        ORDER BY volume_usd DESC
        LIMIT 20
    """,
    "dex_volume_daily_7d": """
        SELECT DATE_TRUNC('day', block_time) AS day,
               SUM(CAST(amount_usd AS DOUBLE)) AS volume_usd,
               COUNT(*) AS trades
        FROM dex.trades
        WHERE block_time > NOW() - INTERVAL '7' day
        GROUP BY 1
        ORDER BY 1
    """,
    "dex_volume_by_blockchain_7d": """
        SELECT blockchain,
               SUM(CAST(amount_usd AS DOUBLE)) AS volume_usd,
               COUNT(*) AS trades
        FROM dex.trades
        WHERE block_time > NOW() - INTERVAL '7' day
        GROUP BY blockchain
        ORDER BY volume_usd DESC
    """,
    "dex_volume_by_protocol_7d": """
        SELECT project, blockchain,
               SUM(CAST(amount_usd AS DOUBLE)) AS volume_usd,
               COUNT(*) AS trades
        FROM dex.trades
        WHERE block_time > NOW() - INTERVAL '7' day
        GROUP BY project, blockchain
        ORDER BY volume_usd DESC
        LIMIT 20
    """,
}


def _headers() -> dict[str, str]:
    return {"X-Dune-API-Key": DUNE_API_KEY, "Content-Type": "application/json"}


def _run_to_rows(sql: str, poll_interval: int = 3,
                 max_wait: int = 600, timeout: int = 30) -> tuple[str, list[dict]]:
    """POST /sql/execute → poll status → fetch results. Returns (execution_id, rows)."""
    resp = requests.post(f"{API}/sql/execute", json={"sql": sql},
                         headers=_headers(), timeout=timeout)
    resp.raise_for_status()
    eid = resp.json().get("execution_id")
    if not eid:
        raise RuntimeError(f"Dune sql/execute returned no execution_id: {resp.json()}")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        st = requests.get(f"{API}/execution/{eid}/status",
                          headers=_headers(), timeout=timeout)
        st.raise_for_status()
        status = st.json()
        state = status.get("state", "")
        if state in ("QUERY_STATE_COMPLETED", "QUERY_STATE_COMPLETED_PARTIAL"):
            break
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELED", "QUERY_STATE_EXPIRED"):
            err = status.get("error") or {}
            raise RuntimeError(f"Dune execution {state}: {err.get('type')} {err.get('message')}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Dune execution {eid} did not finish in {max_wait}s")
    rows: list[dict] = []
    offset, limit = 0, 100
    while True:
        res = requests.get(f"{API}/execution/{eid}/results?limit={limit}&offset={offset}",
                           headers=_headers(), timeout=timeout)
        res.raise_for_status()
        body = res.json()
        if body.get("error"):
            raise RuntimeError(f"Dune results error: {body['error']}")
        rows.extend((body.get("result") or {}).get("rows", []) or [])
        if not body.get("next_offset") and not body.get("next_uri"):
            break
        offset = body.get("next_offset", offset + limit)
    return eid, rows


def run_sql(name: str, storage: Storage | None = None) -> dict | None:
    """Execute a named SQL query from SQL_QUERIES and persist its rows."""
    if not DUNE_API_KEY:
        log.info("DUNE_API_KEY not set — skipping Dune fetch")
        return None
    sql = SQL_QUERIES.get(name)
    if not sql:
        log.warning("No SQL configured for Dune query %s", name)
        return None
    storage = storage or get_storage()
    try:
        eid, rows = _run_to_rows(sql)
        snap = {
            "ts": dt.datetime.utcnow(),
            "source": f"dune:{name}",
            "query": name,
            "execution_id": eid,
            "rows": rows,
        }
        storage.write_onchain_snapshot(name, rows, source="dune", ts=snap["ts"])
        import asyncio
        asyncio.run(bus.publish(EVENT_ONCHAIN, snap))
        log.info("Dune %s: %d rows (execution %s)", name, len(rows), eid)
        return snap
    except Exception as e:  # noqa: BLE001
        log.warning("Dune SQL %s failed: %s", name, e)
        return None


# ── Query-ID mode (saved Dune queries) ──────────────────────────────────
_DEFAULTS: dict[str, int] = {}
DUNE_QUERY_IDS: dict[str, int] = {}


def _load_query_ids() -> dict[str, int]:
    ids: dict[str, int] = dict(_DEFAULTS)
    raw = os.getenv("DUNE_QUERY_IDS", "").strip()
    for token in filter(None, raw.split(",")):
        if "=" not in token:
            continue
        name, _, sid = token.partition("=")
        try:
            ids[name.strip()] = int(sid.strip())
        except ValueError:
            log.warning("Ignoring malformed Dune query id entry: %s", token)
    return ids


def run_dune_query(name: str, storage: Storage | None = None) -> dict | None:
    """Execute a saved query by id and persist its rows."""
    if not DUNE_API_KEY:
        log.info("DUNE_API_KEY not set — skipping Dune fetch")
        return None
    qid = DUNE_QUERY_IDS.get(name)
    if not qid:
        log.info("No Dune query ID configured for %s (set DUNE_QUERY_IDS=%s=<id>)",
                 name, name)
        return None
    storage = storage or get_storage()
    try:
        resp = requests.post(f"{API}/query/{qid}/execute",
                             headers=_headers(), timeout=30)
        resp.raise_for_status()
        eid = resp.json().get("execution_id")
        if not eid:
            raise RuntimeError(f"Dune execute returned no execution_id: {resp.json()}")
        deadline = time.time() + 600
        status = {}
        while time.time() < deadline:
            st = requests.get(f"{API}/execution/{eid}/status",
                              headers=_headers(), timeout=30)
            st.raise_for_status()
            status = st.json()
            state = status.get("state", "")
            if state in ("QUERY_STATE_COMPLETED", "QUERY_STATE_COMPLETED_PARTIAL"):
                break
            if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELED", "QUERY_STATE_EXPIRED"):
                err = status.get("error") or {}
                raise RuntimeError(f"Dune execution {state}: {err.get('type')} {err.get('message')}")
            time.sleep(3)
        rows: list[dict] = []
        offset, limit = 0, 100
        while True:
            res = requests.get(f"{API}/execution/{eid}/results?limit={limit}&offset={offset}",
                               headers=_headers(), timeout=60)
            res.raise_for_status()
            body = res.json()
            rows.extend((body.get("result") or {}).get("rows", []) or [])
            if not body.get("next_offset") and not body.get("next_uri"):
                break
            offset = body.get("next_offset", offset + limit)
        snap = {"ts": dt.datetime.utcnow(), "source": f"dune:{name}",
                "query_id": qid, "execution_id": eid,
                "state": status.get("state"), "rows": rows}
        storage.write_onchain_snapshot(name, rows, source="dune", ts=snap["ts"])
        import asyncio
        asyncio.run(bus.publish(EVENT_ONCHAIN, snap))
        log.info("Dune %s: %d rows (execution %s)", name, len(rows), eid)
        return snap
    except Exception as e:  # noqa: BLE001
        log.warning("Dune query %s failed: %s", name, e)
        return None


def run_all(storage: Storage | None = None) -> list[dict]:
    """Run every configured SQL-direct query and saved query."""
    storage = storage or get_storage()
    out: list[dict] = []
    for name in SQL_QUERIES:
        snap = run_sql(name, storage)
        if snap is not None:
            out.append(snap)
    for name in DUNE_QUERY_IDS:
        snap = run_dune_query(name, storage)
        if snap is not None:
            out.append(snap)
    return out


DUNE_QUERY_IDS = _load_query_ids()


if __name__ == "__main__":
    if not DUNE_API_KEY:
        print("DUNE_API_KEY not set in .env")
    elif SQL_QUERIES:
        snaps = run_all()
        print(f"Ran {len(snaps)}/{len(SQL_QUERIES) + len(DUNE_QUERY_IDS)} queries")
        for s in snaps:
            print(f"  {s['source']}: {len(s['rows'])} rows")
    else:
        print("No DUNE_QUERY_IDS configured (SQL-direct queries already ran).")

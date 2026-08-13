"""Neo4j knowledge-graph writer for Project Agonistes (GAP 1).

Populates the Company knowledge graph that the LangGraph analyst (node_a /
node_b) consumes via ``query_key_relationships``. Node types:

    Company, Person, Sector, Index, ETF

Relationship types:

    COMPETES_WITH, SUPPLIES_TO, PART_OF_SECTOR, PART_OF_INDEX,
    CEO_OF, CFO_OF

Design rules (see implementation_plan_v2.md §Docker/monitoring gaps):
  * Every write uses MERGE → idempotent (safe to re-run on the same data).
  * The ``neo4j`` driver is imported lazily inside functions so the rest of
    the system still starts when the package or server is unavailable.
  * Every public call degrades gracefully: if Neo4j is down it logs a warning
    and returns an empty result instead of raising into the debate pipeline.

Connection is configured via env vars (see core.config / .env.example):
    NEO4J_URI      (default bolt://localhost:7687)
    NEO4J_USER     (default neo4j)
    NEO4J_PASSWORD (default change-me)
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.config import get
from core.logging import get_logger

log = get_logger(__name__)


def _driver():
    """Create (or return cached) a Neo4j driver. Lazy import of the package."""
    from neo4j import GraphDatabase

    uri = get("NEO4J_URI", "bolt://localhost:7687")
    user = get("NEO4J_USER", "neo4j")
    password = get("NEO4J_PASSWORD", "change-me")
    return GraphDatabase.driver(uri, auth=(user, password))


def _safe(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Run fn; on any failure log + return None (graceful degradation)."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning("Neo4j operation '%s' skipped: %s",
                    getattr(fn, "__name__", "op"), exc)
        return None


def neo4j_available() -> bool:
    """True if the server is reachable and the schema constraint query runs."""
    try:
        with _driver().session() as s:
            s.run("RETURN 1").consume()
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def setup_neo4j_schema() -> bool:
    """Create constraints + indexes. Idempotent. Returns False if Neo4j down."""
    statements = [
        # Unique identity keys
        ("CREATE CONSTRAINT company_symbol IF NOT EXISTS "
         "FOR (c:Company) REQUIRE c.symbol IS UNIQUE"),
        ("CREATE CONSTRAINT person_name IF NOT EXISTS "
         "FOR (p:Person) REQUIRE p.name IS UNIQUE"),
        ("CREATE CONSTRAINT sector_name IF NOT EXISTS "
         "FOR (s:Sector) REQUIRE s.name IS UNIQUE"),
        ("CREATE CONSTRAINT index_symbol IF NOT EXISTS "
         "FOR (i:Index) REQUIRE i.symbol IS UNIQUE"),
        ("CREATE CONSTRAINT etf_symbol IF NOT EXISTS "
         "FOR (e:ETF) REQUIRE e.symbol IS UNIQUE"),
        # Lookup indexes
        ("CREATE INDEX company_name_idx IF NOT EXISTS "
         "FOR (c:Company) ON (c.name)"),
        ("CREATE INDEX company_sector_idx IF NOT EXISTS "
         "FOR (c:Company) ON (c.sector)"),
    ]
    try:
        with _driver().session() as s:
            for stmt in statements:
                s.run(stmt).consume()
        log.info("Neo4j schema ensured (%d statements)", len(statements))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Neo4j schema setup failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Company + sector ingestion
# ---------------------------------------------------------------------------

def _yfinance_info(symbol: str) -> dict:
    """Best-effort yfinance .info dict (empty on failure). Lazy import."""
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
        return info if isinstance(info, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.debug("yfinance .info failed for %s: %s", symbol, exc)
        return {}


def ingest_company_to_neo4j(symbol: str, db=None) -> bool:
    """Write a Company node + PART_OF_SECTOR edge. Idempotent (MERGE)."""
    db = db if db is not None else _storage()
    symbol = symbol.upper()
    info = _yfinance_info(symbol)
    fund = db.query_latest_fundamentals(symbol) or {}
    profile = db.get_company_profile(symbol) or {}

    name = (info.get("longName") or info.get("shortName")
            or profile.get("company_name") or symbol)
    sector = info.get("sector") or profile.get("sector") or "UNKNOWN"
    industry = info.get("industry") or profile.get("industry")
    market_cap = info.get("marketCap") or fund.get("market_cap")
    country = info.get("country") or profile.get("country")

    query = (
        "MERGE (c:Company {symbol: $symbol}) "
        "SET c.name = $name, c.sector = $sector, c.industry = $industry, "
        "    c.market_cap = $market_cap, c.country = $country, "
        "    c.updated_at = datetime() "
        "MERGE (sec:Sector {name: $sector}) "
        "MERGE (c)-[:PART_OF_SECTOR]->(sec) "
    )
    params = {
        "symbol": symbol, "name": name, "sector": sector,
        "industry": industry, "market_cap": market_cap, "country": country,
    }

    def _run():
        with _driver().session() as s:
            s.run(query, **params).consume()

    ok = _safe(_run)
    if ok is not None:
        log.info("Neo4j: company %s → %s (%s)", symbol, name, sector)
    return ok is not None


# ---------------------------------------------------------------------------
# Peer + index relationships
# ---------------------------------------------------------------------------

# Seed membership lists (extend freely). Sourced from public S&P 500 /
# NASDAQ-100 and NIFTY 50 constituent lists — representative, not exhaustive.
_SP500_NASDAQ100 = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "AVGO", "JPM", "V", "WMT", "JNJ", "PG", "MA", "HD", "UNH", "COST",
}
_NIFTY50 = {
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
    "KOTAKBANK.NS", "BAJFINANCE.NS", "AXISBANK.NS", "ADANIENT.NS",
}
_INDEX_BY_SYMBOL = {
    **{s: "S&P500" for s in _SP500_NASDAQ100},
    **{s: "NIFTY50" for s in _NIFTY50},
    "SPY": "S&P500", "QQQ": "NASDAQ100",
    "NIFTY50.NS": "NIFTY50", "^NSEI": "NIFTY50",
}


def _peer_candidates(symbol: str, db) -> list[str]:
    """Other Company symbols sharing the same sector, from the local store."""
    profile = db.get_company_profile(symbol)
    if not profile or not profile.get("sector"):
        return []
    sector = profile["sector"]
    peers = []
    for other in db.symbols():
        if other.upper() == symbol.upper():
            continue
        try:
            p = db.get_company_profile(other)
            if p and p.get("sector") == sector:
                peers.append(other.upper())
        except Exception as exc:  # noqa: BLE001
            log.debug("profile lookup failed for %s: %s", other, exc)
            continue
    return peers[:20]


def ingest_peer_relationships(symbol: str, db=None) -> int:
    """Link a Company to sector peers (COMPETES_WITH) and index (PART_OF_INDEX)."""
    db = db if db is not None else _storage()
    symbol = symbol.upper()
    linked = 0

    # Sector peers → COMPETES_WITH (idempotent bidirectional MERGE)
    for peer in _peer_candidates(symbol, db):
        def _link(_peer: str = peer):
            with _driver().session() as s:
                s.run(
                    "MATCH (a:Company {symbol: $a}), (b:Company {symbol: $b}) "
                    "MERGE (a)-[:COMPETES_WITH]->(b) "
                    "MERGE (b)-[:COMPETES_WITH]->(a)",
                    a=symbol, b=_peer,
                ).consume()
        if _safe(_link) is not None:
            linked += 1

    # Index membership → PART_OF_INDEX
    index_name = _INDEX_BY_SYMBOL.get(symbol)
    if index_name:
        def _idx():
            with _driver().session() as s:
                s.run(
                    "MERGE (i:Index {symbol: $idx}) "
                    "MERGE (c:Company {symbol: $sym}) "
                    "MERGE (c)-[:PART_OF_INDEX]->(i)",
                    idx=index_name, sym=symbol,
                ).consume()
        if _safe(_idx) is not None:
            linked += 1

    if linked:
        log.info("Neo4j: %s linked to %d peers/indexes", symbol, linked)
    return linked


# ---------------------------------------------------------------------------
# Management ingestion
# ---------------------------------------------------------------------------

def ingest_management(symbol: str) -> int:
    """Create CEO_OF / CFO_OF edges from yfinance companyOfficers. Idempotent."""
    symbol = symbol.upper()
    info = _yfinance_info(symbol)
    officers = info.get("companyOfficers") or []
    linked = 0
    for off in officers:
        if not isinstance(off, dict):
            continue
        name = off.get("name")
        title = str(off.get("title") or "").upper()
        if not name:
            continue
        rel = None
        if "CEO" in title and "CFO" not in title:
            rel = "CEO_OF"
        elif "CFO" in title:
            rel = "CFO_OF"
        if rel is None:
            continue
        props = {k: v for k, v in off.items()
                 if k in ("age", "totalPay") and v is not None}

        def _persist(_rel: str = rel, _name: str = name,
                     _props: dict = props):
            with _driver().session() as s:
                s.run(
                    f"MATCH (c:Company {{symbol: $sym}}) "
                    f"MERGE (p:Person {{name: $name}}) "
                    f"SET p += $props "
                    f"MERGE (p)-[:{_rel}]->(c)",
                    sym=symbol, name=_name, props=_props,
                ).consume()
        if _safe(_persist) is not None:
            linked += 1

    if linked:
        log.info("Neo4j: %s management linked (%d officers)", symbol, linked)
    return linked


# ---------------------------------------------------------------------------
# Query — consumed by the LangGraph analyst
# ---------------------------------------------------------------------------

def query_key_relationships(symbol: str, top_k: int = 5) -> list[str]:
    """Return a list of human-readable relationship lines for a symbol.

    Empties out (never raises) when the graph / Neo4j is unavailable, so the
    debate pipeline degrades gracefully.
    """
    symbol = symbol.upper()
    query = (
        "MATCH (c:Company {symbol: $symbol}) "
        "OPTIONAL MATCH (c)-[:COMPETES_WITH]->(peer:Company) "
        "OPTIONAL MATCH (c)-[:PART_OF_SECTOR]->(sec:Sector) "
        "OPTIONAL MATCH (c)-[:PART_OF_INDEX]->(idx:Index) "
        "OPTIONAL MATCH (ceo:Person)-[:CEO_OF]->(c) "
        "OPTIONAL MATCH (cfo:Person)-[:CFO_OF]->(c) "
        "WITH c, collect(DISTINCT peer.symbol)[..$k] AS peers, "
        "     collect(DISTINCT sec.name)[..5] AS sectors, "
        "     collect(DISTINCT idx.symbol)[..5] AS indexes, "
        "     collect(DISTINCT ceo.name)[..1] AS ceos, "
        "     collect(DISTINCT cfo.name)[..1] AS cfos "
        "RETURN c.name AS name, c.sector AS sector, c.industry AS industry, "
        "       peers, sectors, indexes, ceos, cfos"
    )
    params = {"symbol": symbol, "k": top_k}

    def _query():
        with _driver().session() as s:
            rec = s.run(query, **params).single()
            return dict(rec) if rec else None

    rec = _safe(_query)
    if not rec:
        return []
    if not any(rec.get(k) for k in ("peers", "sectors", "indexes", "ceos", "cfos")):
        return []

    lines = [(f"{rec.get('name') or symbol} ({rec.get('sector') or 'n/a'}, "
              f"{rec.get('industry') or 'n/a'})")]
    if rec.get("peers"):
        lines.append("Peers/competitors: " + ", ".join(sorted(rec["peers"])))
    if rec.get("sectors"):
        lines.append("Sector: " + ", ".join(sorted(rec["sectors"])))
    if rec.get("indexes"):
        lines.append("Index membership: " + ", ".join(sorted(rec["indexes"])))
    if rec.get("ceos"):
        lines.append("CEO: " + ", ".join(rec["ceos"]))
    if rec.get("cfos"):
        lines.append("CFO: " + ", ".join(rec["cfos"]))
    return lines


def _storage():
    from core.db import get_storage

    return get_storage()

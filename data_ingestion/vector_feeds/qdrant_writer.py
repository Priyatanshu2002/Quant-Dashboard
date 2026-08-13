"""Qdrant vector embedding pipeline for Project Agonistes (GAP 2).

Stores embeddings for SEC filings, earnings-transcript turns, past debate
theses and news so the LangGraph debate can do semantic recall.

Collections (all cosine distance):
    filings              — 10-K / 10-Q text chunks
    earnings_transcripts — quarterly call speaker-turn chunks
    analyst_theses       — archived bull/bear debate outputs
    news                 — news article embeddings

Vector dimension is detected at runtime:
    1536  — OpenAI text-embedding-3-small (when OPENAI_API_KEY is set)
    384   — sentence-transformers all-MiniLM-L6-v2 (local fallback)

Design rules:
  * ``qdrant-client``, ``openai`` and ``sentence-transformers`` are imported
    lazily so the system still starts without them.
  * Deterministic point IDs (uuid5 of the natural key) → upserts are
    idempotent (re-running produces no duplicates).
  * Every public call degrades gracefully when Qdrant / embeddings are down.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from core.config import get
from core.logging import get_logger

log = get_logger(__name__)

EMBED_DIM = 1536
FALLBACK_EMBED_DIM = 384

# Rough char budget approximating a token (≈4 chars/token for English).
_CHUNK_CHARS = 500 * 4
_OVERLAP_CHARS = 50 * 4

COLLECTIONS = ["filings", "earnings_transcripts", "analyst_theses", "news"]


def _qdrant_client():
    from qdrant_client import QdrantClient

    url = get("QDRANT_URL", "http://localhost:6333")
    api_key = get("QDRANT_API_KEY", "") or None
    return QdrantClient(url=url, api_key=api_key)


def _safe(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Run fn; on any failure log + return None (graceful degradation)."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning("Qdrant operation '%s' skipped: %s",
                    getattr(fn, "__name__", "op"), exc)
        return None


def _storage():
    from core.db import get_storage

    return get_storage()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _use_openai() -> bool:
    return bool(get("OPENAI_API_KEY", ""))


def effective_embed_dim() -> int:
    """1536 if OpenAI key present else 384 (local MiniLM)."""
    return EMBED_DIM if _use_openai() else FALLBACK_EMBED_DIM


def embed_text(texts: Iterable[str], *, batch_size: int = 100) -> list[list[float]]:
    """Embed a list of strings → list of vectors.

    Tries OpenAI text-embedding-3-small, falls back to local
    sentence-transformers all-MiniLM-L6-v2. Batches to avoid rate limits.
    """
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    if _use_openai():
        try:
            return _embed_openai(texts, batch_size)
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenAI embeddings unavailable (%s) — falling back "
                        "to local sentence-transformers", exc)
    try:
        return _embed_local(texts)
    except Exception as exc:  # noqa: BLE001
        log.warning("Local embeddings unavailable (%s) — returning no vectors", exc)
        return []


def _embed_openai(texts: list[str], batch_size: int) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=get("OPENAI_API_KEY"))
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model="text-embedding-3-small", input=batch,
            dimensions=EMBED_DIM,
        )
        vectors.extend(d.embedding for d in resp.data)
    return vectors


_embedder = None  # lazy, module-level cache


def _embed_local(texts: list[str]) -> list[list[float]]:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder.encode(texts, batch_size=64).tolist()


# ---------------------------------------------------------------------------
# Collection setup
# ---------------------------------------------------------------------------

def setup_qdrant_collections() -> bool:
    """Create missing collections at the effective dimension. Idempotent.

    Returns False when Qdrant / qdrant-client is unavailable (graceful).
    """
    try:
        from qdrant_client.models import Distance, VectorParams

        client = _qdrant_client()
        dim = effective_embed_dim()
        created = 0
        for name in COLLECTIONS:
            if not client.collection_exists(name):
                client.create_collection(
                    name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
                created += 1
        if created:
            log.info("Qdrant: created %d collection(s) at dim=%d", created, dim)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Qdrant collection setup skipped: %s", exc)
        return False


def _point_id(natural_key: str) -> str:
    """Deterministic UUID from a natural key → idempotent upsert."""
    return str(uuid.UUID(bytes=hashlib.md5(natural_key.encode()).digest()))


def _upsert(collection: str, natural_keys: list[str], vectors: list[list[float]],
            payloads: list[dict]) -> int:
    """Idempotent bulk upsert. Returns count upserted (0 if any failure)."""
    from qdrant_client.models import PointStruct

    client = _qdrant_client()
    points = [
        PointStruct(id=_point_id(key), vector=vec, payload=payload)
        for key, vec, payload in zip(natural_keys, vectors, payloads)
    ]
    client.upsert(collection_name=collection, points=points)
    log.debug("Qdrant: upserted %d points → %s", len(points), collection)
    return len(points)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_chars: int = _CHUNK_CHARS,
               overlap_chars: int = _OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping ~500-token chunks on paragraph/sentence
    boundaries (approx by char count; 50-token overlap)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks, i = [], 0
    while i < len(text):
        window = text[i:i + chunk_chars]
        # Back off to the last sentence/space boundary for a clean break.
        cut = None
        for sep in (". ", "! ", "? ", " "):
            cut = window.rfind(sep, chunk_chars // 2)
            if cut != -1:
                cut += len(sep)
                break
        end = cut or len(window)
        chunks.append(window[:end].strip())
        i += max(end - overlap_chars, 1)
    return [c for c in chunks if c]


def _chunk_speaker_turns(transcript_text: str, max_chars: int = _CHUNK_CHARS) -> list[tuple[str, str]]:
    """Split an earnings transcript into (speaker, text) turn chunks."""
    if not transcript_text:
        return []
    # Naive speaker-turn split on common formats: "SPEAKER:" at line starts.
    pattern = re.compile(r"(?m)^\s*([A-Z][A-Za-z .&'-]{1,40}):\s*")
    matches = list(pattern.finditer(transcript_text))
    turns: list[tuple[str, str]] = []
    if not matches:
        return [("", transcript_text.strip())]
    for idx, m in enumerate(matches):
        speaker = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(transcript_text)
        body = transcript_text[start:end].strip()
        # Sub-chunk long turns.
        for piece in chunk_text(body, chunk_chars=max_chars):
            turns.append((speaker, piece))
    return turns


# ---------------------------------------------------------------------------
# Ingesters
# ---------------------------------------------------------------------------

def ingest_sec_filing(symbol: str, filing_url: str, doc_type: str,
                      text: str | None = None) -> int:
    """Download + chunk + embed an SEC filing into the `filings` collection."""
    symbol = symbol.upper()
    if not text:
        text = _fetch_filing_text(filing_url)
    if not text:
        return 0
    chunks = chunk_text(text)
    if not chunks:
        return 0
    payloads = [
        {"symbol": symbol, "doc_type": doc_type.upper(),
         "chunk_index": i, "filing_url": filing_url}
        for i in range(len(chunks))
    ]
    keys = [f"{symbol}|{doc_type.upper()}|{i}" for i in range(len(chunks))]
    vectors = embed_text(chunks)
    if not vectors:
        return 0
    return _safe(_upsert, "filings", keys, vectors, payloads) or 0


def ingest_earnings_transcript(symbol: str, transcript_text: str,
                               quarter: int | None, year: int | None) -> int:
    """Embed speaker-turn chunks of an earnings call into `earnings_transcripts`."""
    symbol = symbol.upper()
    turns = _chunk_speaker_turns(transcript_text)
    if not turns:
        return 0
    payloads = [
        {"symbol": symbol, "speaker": sp, "quarter": quarter, "year": year,
         "chunk_index": i}
        for i, (sp, _) in enumerate(turns)
    ]
    keys = [f"{symbol}|Q{quarter}-{year}|{i}" for i in range(len(turns))]
    vectors = embed_text([t for _, t in turns])
    if not vectors:
        return 0
    return _safe(_upsert, "earnings_transcripts", keys, vectors, payloads) or 0


def archive_debate_thesis(symbol: str, bull_thesis: str, bear_thesis: str,
                          decision: str, date: str) -> int:
    """Archive the bull/bear thesis of a completed debate into `analyst_theses`.

    Called from node_i_mirofish after each debate cycle so future cycles can
    retrieve "what did the bear say about AAPL last time?".
    """
    symbol = symbol.upper()
    docs, keys, payloads = [], [], []
    if bull_thesis:
        docs.append(bull_thesis)
        keys.append(f"{symbol}|{date}|bull")
        payloads.append({"symbol": symbol, "date": date, "side": "bull",
                         "decision": decision})
    if bear_thesis:
        docs.append(bear_thesis)
        keys.append(f"{symbol}|{date}|bear")
        payloads.append({"symbol": symbol, "date": date, "side": "bear",
                         "decision": decision})
    if not docs:
        return 0
    vectors = embed_text(docs)
    if not vectors:
        return 0
    n = _safe(_upsert, "analyst_theses", keys, vectors, payloads) or 0
    if n:
        log.info("Qdrant: archived %d thesis doc(s) for %s", n, symbol)
    return n


# ---------------------------------------------------------------------------
# Search — consumed by the LangGraph debate nodes
# ---------------------------------------------------------------------------

def semantic_search(collection: str, query_text: str,
                    symbol_filter: str | None = None, top_k: int = 5) -> list[dict]:
    """Semantic search over a collection. Returns list of {payload, score}.

    Never raises: returns [] when Qdrant / embeddings are unavailable so the
    debate pipeline degrades gracefully.
    """
    if collection not in COLLECTIONS:
        return []
    vec = embed_text([query_text])
    if not vec:
        return []

    def _search():
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = _qdrant_client()
        query_filter = None
        if symbol_filter:
            query_filter = Filter(must=[
                FieldCondition(key="symbol",
                               match=MatchValue(value=symbol_filter.upper()))])
        hits = client.search(collection_name=collection, query_vector=vec[0],
                             query_filter=query_filter, limit=top_k)
        return [
            {"payload": h.payload, "score": float(h.score),
             "text": (h.payload or {}).get("text", "")}
            for h in hits
        ]

    return _safe(_search) or []


def _fetch_filing_text(filing_url: str) -> str:
    """Best-effort download of filing text from EDGAR. Returns '' on failure."""
    try:
        import requests

        resp = requests.get(filing_url, timeout=30, headers={
            "User-Agent": "Project Agonistes research@agonistes.local"})
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001
        log.debug("Filing download failed for %s: %s", filing_url, exc)
        return ""

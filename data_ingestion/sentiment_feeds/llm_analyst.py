"""LLM analyst (plan §8.1 LLM_CALL) — deepseek-scored news + fundamental verdicts.

Two analysis kinds, both cached in `llm_analyses` and served to the UI:

  * NEWS        — scores the last ~48h of headline events and writes a
                  sentiment verdict: score [-1,1], label, summary, key
                  points and risks. Batch: 1 LLM call per ~25 headlines.
  * FUNDAMENTAL — equity-analyst style write-up from the latest fundamental
                  snapshot + DCF: rating, thesis, catalysts, risks.

Every path degrades gracefully: when no LLM key is configured or the API
call fails, deterministic fallbacks (lexicon aggregate / rule-based rating)
are stored with model="fallback" so the UI always has a verdict.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import re

from core.db import Storage, get_storage
from core.logging import get_logger
from langgraph_app.src.utils.llm_client import call_openrouter_raw, llm_available

log = get_logger(__name__)

NEWS_FRESH_HOURS = 6
FUND_FRESH_HOURS = 24
MAX_HEADLINES = 25

_SYS_NEWS = (
    "You are a professional equity research sentiment analyst. Given news "
    "headlines for one ticker, produce a JSON object ONLY (no markdown): "
    '{"score": float in [-1,1], "label": "bullish"|"neutral"|"bearish", '
    '"summary": string <= 220 chars, "key_points": [string, ...] (<=5), '
    '"risks": [string, ...] (<=3)}. Score reflects the balance of sentiment; '
    "label from |score|: <0.25 neutral, else bullish/bearish."
)

_SYS_FUND = (
    "You are a senior equity analyst at a multi-asset fund. Given a company's "
    "latest fundamentals (income statement, balance sheet, cash flow, DCF, "
    "valuation multiples), produce a JSON object ONLY (no markdown): "
    '{"score": float in [-1,1] (positive = attractive), "rating": "STRONG_BUY"|"BUY"|"HOLD"|"SELL"|"STRONG_SELL", '
    '"thesis": string <= 300 chars, "catalysts": [string, ...] (<=4), '
    '"risks": [string, ...] (<=4), "fair_value_est": float|null}. '
    "Be conservative: value stocks with no margin of safety get HOLD at best."
)


def _parse_json(text: str) -> dict | None:
    """Tolerant JSON extraction (strip fences, take first balanced object)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth, end = 0, -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _clamp_score(x) -> float:
    try:
        return max(-1.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _label_for(score: float) -> str:
    if score > 0.25:
        return "bullish"
    if score < -0.25:
        return "bearish"
    return "neutral"


async def _llm_json(system: str, user: str, temperature: float = 0.3) -> dict | None:
    if not llm_available():
        return None
    try:
        text = await call_openrouter_raw(system, user, temperature=temperature,
                                         max_tokens=1200)
        return _parse_json(text)
    except Exception as e:  # noqa: BLE001
        log.warning("LLM analyst call failed: %s", e)
        return None


# ── NEWS verdict ────────────────────────────────────────────────────────
def _news_fallback(events: list[dict]) -> dict:
    if not events:
        return {"score": 0.0, "label": "neutral",
                "summary": "No recent news events to analyze.",
                "key_points": [], "risks": [], "analyzed_events": 0}
    scores = [e["score"] for e in events]
    weights = [e.get("source_weight", 1.0) for e in events]
    wsum = sum(weights)
    score = sum(s * w for s, w in zip(scores, weights)) / wsum if wsum else 0.0
    pos = sum(1 for s in scores if s > 0.2)
    neg = sum(1 for s in scores if s < -0.2)
    summary = (f"Lexicon aggregate of {len(events)} headlines: {pos} positive, "
               f"{neg} negative, {len(events) - pos - neg} neutral.")
    return {"score": round(score, 3), "label": _label_for(score),
            "summary": summary, "key_points": [], "risks": [],
            "analyzed_events": len(events)}


async def _news_llm(events: list[dict]) -> dict | None:
    lines = [f"- [{e['source']}] ({e['score']:+.2f}) {e['headline']}" for e in events]
    return await _llm_json(_SYS_NEWS, "\n".join(lines) or "No headlines.")


def analyze_news_sentiment(symbol: str, storage: Storage | None = None,
                           db: Storage | None = None, force: bool = False) -> dict:
    """LLM-scored news verdict for a symbol (fresh-cache aware)."""
    storage = storage or get_storage()
    db = db or storage
    if not force:
        cached = storage.query_latest_llm_analysis(symbol, kind="NEWS")
        if cached:
            age = dt.datetime.utcnow() - dt.datetime.fromisoformat(cached["created_at"])
            if age < dt.timedelta(hours=NEWS_FRESH_HOURS):
                verdict = dict(cached["verdict"])
                verdict["cached"] = True
                verdict["model"] = cached["model"]
                verdict["created_at"] = cached["created_at"]
                return verdict

    events = db.query_sentiment_events(symbol, hours=48)
    events = [e for e in events if e.get("headline")][:MAX_HEADLINES]
    verdict = _news_fallback(events)
    model = "fallback-lexicon"

    llm_out = asyncio.run(_news_llm(events)) if llm_available() else None
    if llm_out:
        verdict = {
            "score": _clamp_score(llm_out.get("score", verdict["score"])),
            "label": str(llm_out.get("label", _label_for(verdict["score"]))).lower(),
            "summary": str(llm_out.get("summary", verdict["summary"]))[:500],
            "key_points": [str(p)[:220] for p in (llm_out.get("key_points") or [])][:5],
            "risks": [str(r)[:220] for r in (llm_out.get("risks") or [])][:3],
            "analyzed_events": len(events),
        }
        model = "deepseek/deepseek-v4-flash-0731"

    storage.upsert_llm_analysis(symbol.upper(), "NEWS", verdict, model)
    verdict["cached"] = False
    verdict["model"] = model
    verdict["created_at"] = dt.datetime.utcnow().isoformat(sep=" ")
    return verdict


# ── FUNDAMENTAL verdict ─────────────────────────────────────────────────
def _fund_fallback(snap: dict, ratios: list[dict] | None) -> dict:
    if not snap:
        return {"score": 0.0, "rating": "HOLD", "thesis": "Insufficient data.",
                "catalysts": [], "risks": [], "fair_value_est": None}
    score = 0.0
    reasons = []
    margin = snap.get("dcf_margin_of_safety")
    if margin is not None:
        score += max(-0.4, min(0.4, margin))
        reasons.append(f"DCF margin of safety {margin * 100:+.1f}%")
    fpe = snap.get("forward_pe")
    if fpe and fpe > 0:
        score += max(-0.2, min(0.2, (15 - fpe) / 30))
        reasons.append(f"forward P/E {fpe:.1f}x")
    growth = snap.get("revenue_yoy_growth")
    if growth is not None:
        score += max(-0.2, min(0.2, growth))
        reasons.append(f"revenue growth {growth * 100:.1f}%")
    roic = snap.get("roic")
    if roic is not None:
        score += max(-0.2, min(0.2, (roic - 0.10) * 2))
        reasons.append(f"ROIC {roic * 100:.1f}%")
    score = max(-1.0, min(1.0, score))
    rating = ("STRONG_BUY" if score > 0.5 else "BUY" if score > 0.2 else
              "SELL" if score < -0.4 else "STRONG_SELL" if score < -0.6 else "HOLD")
    return {"score": round(score, 3), "rating": rating,
            "thesis": "Rule-based aggregate: " + "; ".join(reasons) + ".",
            "catalysts": [], "risks": [],
            "fair_value_est": snap.get("dcf_intrinsic_value")}


async def _fund_llm(snap: dict, ratios: list[dict] | None) -> dict | None:
    lines = [f"{k}: {v}" for k, v in sorted(snap.items())
             if v is not None and k not in ("raw_data", "filing_url", "transcript_summary")]
    if ratios:
        latest = ratios[-1]
        lines.append(f"latest_quarter_ratios: {json.dumps(latest, default=str)}")
    return await _llm_json(_SYS_FUND, "\n".join(lines)[:6000], temperature=0.2)


def analyze_fundamentals(symbol: str, storage: Storage | None = None,
                         db: Storage | None = None, force: bool = False) -> dict:
    """Equity-analyst LLM verdict from the latest fundamental snapshot."""
    storage = storage or get_storage()
    db = db or storage
    if not force:
        cached = storage.query_latest_llm_analysis(symbol, kind="FUNDAMENTAL")
        if cached:
            age = dt.datetime.utcnow() - dt.datetime.fromisoformat(cached["created_at"])
            if age < dt.timedelta(hours=FUND_FRESH_HOURS):
                verdict = dict(cached["verdict"])
                verdict["cached"] = True
                verdict["model"] = cached["model"]
                verdict["created_at"] = cached["created_at"]
                return verdict

    snap = db.query_latest_fundamentals(symbol)
    ratios = db.query_financial_statements(symbol, statement="income", quarters=1)
    if ratios and ratios[0].get("data"):
        ratio_rows = [{"period": r["period"], **r["data"]} for r in ratios]
    else:
        ratio_rows = None

    verdict = _fund_fallback(snap or {}, ratio_rows)
    model = "fallback-rule-based"

    llm_out = asyncio.run(_fund_llm(snap or {}, ratio_rows)) if llm_available() else None
    if llm_out:
        verdict = {
            "score": _clamp_score(llm_out.get("score", verdict["score"])),
            "rating": str(llm_out.get("rating", verdict["rating"])).upper(),
            "thesis": str(llm_out.get("thesis", verdict["thesis"]))[:600],
            "catalysts": [str(c)[:220] for c in (llm_out.get("catalysts") or [])][:4],
            "risks": [str(r)[:220] for r in (llm_out.get("risks") or [])][:4],
            "fair_value_est": llm_out.get("fair_value_est"),
        }
        model = "deepseek/deepseek-v4-flash-0731"

    storage.upsert_llm_analysis(symbol.upper(), "FUNDAMENTAL", verdict, model)
    verdict["cached"] = False
    verdict["model"] = model
    verdict["created_at"] = dt.datetime.utcnow().isoformat(sep=" ")
    return verdict


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["AAPL"]:
        print(f"--- {t} NEWS ---")
        print(json.dumps(analyze_news_sentiment(t), indent=2, default=str))
        print(f"--- {t} FUNDAMENTAL ---")
        print(json.dumps(analyze_fundamentals(t), indent=2, default=str))

#!/usr/bin/env python3
"""arXiv sweep: pull recent + high-relevance papers on DL/transformers in trading.

Saves per-query JSON to data/research/raw/ so later stages don't re-fetch.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path("data/research/raw")
NS = {"a": "http://www.w3.org/2005/Atom"}

QUERIES = {
    "tft_finance": 'all:"temporal fusion transformer" AND (all:financial OR all:stock OR all:trading)',
    "transformer_stock": 'all:transformer AND all:"stock prediction"',
    "transformer_trading": 'all:transformer AND (all:"algorithmic trading" OR all:"trading strategy")',
    "deep_learning_trading": 'all:"deep learning" AND (all:trading OR all:"stock market")',
    "lstm_stock": 'all:LSTM AND (all:"stock market" OR all:"price prediction")',
    "llm_trading": 'all:"large language model" AND (all:trading OR all:finance OR all:"stock")',
    "rl_trading": 'all:"reinforcement learning" AND (all:trading OR all:"portfolio")',
    "attention_finance": 'all:attention AND all:"financial time series"',
    "stock_forecast_recent": 'ti:stock AND abs:deep+learning',
}

def fetch(query: str, max_results: int = 25, sort: str = "relevance",
          retries: int = 5) -> list[dict]:
    q = urllib.parse.quote(query)
    url = (f"https://export.arxiv.org/api/query?search_query={q}"
           f"&max_results={max_results}&sortBy={sort}&sortOrder=descending")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "agonistes-research/0.1"})
            with urllib.request.urlopen(req, timeout=45) as r:
                root = ET.parse(r).getroot()
            out = []
            for entry in root.findall("a:entry", NS):
                arxiv_id = entry.find("a:id", NS).text.strip().split("/abs/")[-1]
                out.append({
                    "arxiv_id": arxiv_id,
                    "title": " ".join(entry.find("a:title", NS).text.strip().split()),
                    "published": entry.find("a:published", NS).text[:10],
                    "authors": [a.find("a:name", NS).text for a in entry.findall("a:author", NS)],
                    "categories": [c.get("term") for c in entry.findall("a:category", NS)],
                    "abstract": " ".join(entry.find("a:summary", NS).text.strip().split()),
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                })
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 10 * (attempt + 1)
            print(f"  retry {attempt + 1} after {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError(f"query failed after {retries} attempts: {last_err}")

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, query in QUERIES.items():
        try:
            papers = fetch(query)
            (OUT / f"{name}.json").write_text(json.dumps(papers, indent=1), encoding="utf-8")
            print(f"{name}: {len(papers)} papers")
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAILED {e}")
        time.sleep(10)  # arXiv rate limit ~1 req/3s; stay well clear

if __name__ == "__main__":
    main()

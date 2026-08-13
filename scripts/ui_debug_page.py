#!/usr/bin/env python3
"""Dump console messages + page text for a given route (debug UI wiring)."""
from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

import websockets


def get_pages() -> list[dict]:
    with urllib.request.urlopen("http://127.0.0.1:9222/json") as r:
        return json.load(r)


async def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3001/"
    pages = [p for p in get_pages() if p["type"] == "page"]
    ws_url = pages[0]["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        mid = 0

        async def cmd(method: str, params: dict | None = None) -> dict:
            nonlocal mid
            mid += 1
            await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == mid:
                    return msg

        await cmd("Page.enable")
        await cmd("Runtime.enable")
        await cmd("Log.enable")
        await cmd("Page.navigate", {"url": url})
        await asyncio.sleep(4)
        text = await cmd("Runtime.evaluate", {"expression": "document.body.innerText",
                                              "returnByValue": True})
        print("=== PAGE TEXT ===")
        print(text["result"]["result"].get("value", "")[:3000])
        print("=== NETWORK/API STATE ===")
        state = await cmd("Runtime.evaluate", {"expression": """
            (async () => {
              try {
                const r = await fetch('/api/screener/top');
                const j = await r.json();
                return 'fetch /api/screener/top -> ' + r.status + ' items=' + (Array.isArray(j) ? j.length : JSON.stringify(j).slice(0,120));
              } catch (e) { return 'fetch failed: ' + e.message; }
            })()
        """, "awaitPromise": True, "returnByValue": True})
        print(state["result"]["result"].get("value"))


if __name__ == "__main__":
    asyncio.run(main())

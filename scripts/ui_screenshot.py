#!/usr/bin/env python3
"""CDP screenshot driver for the Agonistes dashboard (headless Chrome via websockets)."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import urllib.request

import websockets

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "screenshots")


def list_pages(port: int = 9222) -> list[dict]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as r:
        return json.load(r)


async def shot(ws_url: str, url: str, out_path: str, wait_s: float = 6.0) -> None:
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
        await cmd("Emulation.setDeviceMetricsOverride",
                  {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False})
        await cmd("Page.navigate", {"url": url})
        # wait for load + a render settle
        await asyncio.sleep(wait_s)
        # force any lazy content: scroll to bottom then back to top
        await cmd("Runtime.evaluate",
                  {"expression": "window.scrollTo(0, document.body.scrollHeight)"})
        await asyncio.sleep(0.8)
        await cmd("Runtime.evaluate", {"expression": "window.scrollTo(0, 0)"})
        await asyncio.sleep(0.8)
        res = await cmd("Page.captureScreenshot", {"format": "png"})
        data = base64.b64decode(res["result"]["data"])
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"saved {out_path} ({len(data)} bytes)")


async def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    routes = {
        "screener": "http://localhost:3001/",
        "financials": "http://localhost:3001/financials",
        "backtests": "http://localhost:3001/backtest",
        "portfolio": "http://localhost:3001/portfolio",
        "debate": "http://localhost:3001/debate",
    }
    targets = sys.argv[1:] or list(routes)
    for name in targets:
        url = routes[name]
        # fresh tab per page
        pages = [p for p in list_pages() if p["type"] == "page"]
        ws_url = pages[0]["webSocketDebuggerUrl"]
        await shot(ws_url, url, os.path.join(OUT, f"ui_{name}.png"))
        time.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())

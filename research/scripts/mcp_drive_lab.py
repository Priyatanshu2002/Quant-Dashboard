"""Drive the research lab notebook through the Jupyter MCP server (v3).

Executes each EXISTING code cell in place via notebook_code_run_existing
(by 0-based index), so the notebook stays clean (27 cells) with live outputs
recorded. Connects through the mcp_jupyter_notebook MCP server, which talks to
the live Jupyter server the user watches in a browser.

Usage:
    python research/scripts/mcp_drive_lab.py [--cell N] [--from N] [--to N]

Env:
    AG_NB : notebook path relative to the Jupyter root_dir (default 01_empirical_research_lab.ipynb)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
TOKEN = "agonistes"
NB = os.environ.get("AG_NB", "01_empirical_research_lab.ipynb")


def _dump(res) -> str:
    parts = []
    for chunk in res.content:
        text = getattr(chunk, "text", "") or ""
        if text.strip():
            parts.append(text.rstrip("\n"))
    return "\n".join(parts)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=None,
                    help="run only this 1-based code-cell ordinal")
    ap.add_argument("--from", dest="from_cell", type=int, default=1,
                    help="start from this 1-based code-cell ordinal")
    ap.add_argument("--to", dest="to_cell", type=int, default=0,
                    help="run up to this 1-based code-cell ordinal (0=all)")
    args = ap.parse_args()

    params = StdioServerParameters(
        command=str(PY),
        args=["-m", "mcp_jupyter_notebook", "--mode", "server",
              "--base-url", "http://127.0.0.1:8888", "--token", TOKEN,
              "--kernel-name", "python3"],
        env=None,
    )

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as sess:
            await sess.initialize()
            tools = {t.name: t for t in (await sess.list_tools()).tools}
            print(f"[mcp] connected — {len(tools)} tools available", flush=True)

            # open the target notebook first (required by the MCP server)
            open_res = await sess.call_tool("notebook_open", {"notebook_path": NB})
            if "error" in _dump(open_res).lower():
                print("[mcp] notebook_open note:", _dump(open_res)[:200], flush=True)

            # read notebook to map code-cell ordinals -> 0-based indices
            read_res = await sess.call_tool("notebook_read", {"notebook_path": NB})
            nb = json.loads("".join(x.text for x in read_res.content))
            code_idx = 0
            targets = []
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code":
                    code_idx += 1
                    if args.cell and code_idx != args.cell:
                        continue
                    if code_idx < args.from_cell:
                        continue
                    if args.to_cell and code_idx > args.to_cell:
                        continue
                    targets.append((code_idx, i, cell["source"]))

            print(f"[mcp] notebook has {len(nb['cells'])} cells; "
                  f"will execute {len(targets)} code cells", flush=True)

            for code_idx, cell_idx, source in targets:
                head = (source.strip().splitlines() or ["(blank)"])[0][:70]
                print(f"\n{'='*70}\n[CELL {code_idx:02d}] {head}\n{'-'*70}", flush=True)
                res = await sess.call_tool(
                    "notebook_code_run_existing",
                    {"cell_index": cell_idx, "code": source, "notebook_path": NB,
                     "timeout": 3600})
                out = _dump(res)
                # strip the JSON-ish metadata, show just formatted_output when present
                if "formatted_output" in out:
                    # it's the full JSON blob; extract readable fields
                    try:
                        parsed = json.loads(out)
                        print(parsed.get("formatted_output", out))
                        if parsed.get("error_message"):
                            print("!! MCP ERROR:", parsed["error_message"])
                    except Exception:
                        print(out)
                else:
                    print(out)
                print(f"[CELL {code_idx:02d}] OK", flush=True)

    print(f"\n[mcp] done. executed {len(targets)} code cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

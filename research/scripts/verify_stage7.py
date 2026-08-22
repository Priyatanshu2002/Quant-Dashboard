"""Verify Stage 7 results-viewer cells execute correctly.

Runs ONLY the Stage 7 cells (with a minimal context shim providing the imports
they need) through nbclient, so this tests the Stage 7 code in isolation — the
heavy Stage 1-6 preamble and model training are not executed.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "research" / "notebooks" / "01_empirical_research_lab.ipynb"

nb = nbf.read(NB, as_version=4)
marker = "# ==== STAGE 7 : RESEARCH RESULTS VIEWER ===="
stage7 = [c for c in nb.cells
          if c.cell_type == "code" and marker in
          (c.source if isinstance(c.source, str) else "".join(c.source))]

shim = nbf.v4.new_code_cell(
    "from pathlib import Path\n"
    "import pandas as pd\n"
    "import json\n"
    "RESULTS = Path('research/results')\n"
    "def display(obj):\n"
    "    from IPython.display import display as _d\n"
    "    _d(obj)\n"
)
nbk = nbf.v4.new_notebook()
nbk.cells = [shim] + [nbf.from_dict(c) for c in stage7]
for c in nbk.cells:
    c.setdefault("id", "")

client = NotebookClient(nbk, timeout=600, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()

errors = [c for c in nbk.cells if c.cell_type == "code"
          and any(o.get("output_type") == "error" for o in c.get("outputs", []))]
print(f"executed {len(stage7)} Stage 7 code cells")
if errors:
    for c in errors:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                print("ERROR:", o.get("ename"), o.get("evalue"))
    print(f"{len(errors)} Stage 7 cell(s) FAILED")
    sys.exit(1)
print("Stage 7 executed cleanly — no cell errors.")

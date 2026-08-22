"""Cell-by-cell executor for the empirical research lab notebook.

Runs every code cell through a live Jupyter kernel (nbclient — the same engine
JupyterLab uses) and prints each cell's real stdout + captured figures so the
agent can inspect and verify every step BEFORE moving to the next stage.

Usage:
    RUN_MODE=smoke python research/scripts/run_lab_cells.py [--stop-on-error] [--tag N]

Outputs:
  - Executes the notebook in-place, recording outputs into the .ipynb.
  - Writes data/benchmark/* artifacts as each stage completes.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

# nbclient/zmq need the selector event-loop policy on Windows, or the kernel
# fails to start (kc stays None).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "research" / "notebooks" / "01_empirical_research_lab.ipynb"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-on-error", action="store_true",
                    help="halt at first failing cell")
    ap.add_argument("--cell", type=int, default=None,
                    help="only run up to this 1-based code-cell index (safety)")
    ap.add_argument("--tag", type=str, default="",
                    help="optional label appended to printed section markers")
    args = ap.parse_args()

    os.chdir(ROOT)
    nb = nbf.read(NB, as_version=4)

    # Working notebook for execution. Cells are the SAME objects as nb.cells, so
    # nbclient mutates their outputs in place; if --cell truncates the run, the
    # saved notebook still keeps all cells (only executed ones get new outputs).
    work = nbf.v4.new_notebook()
    work.metadata = dict(nb.metadata or {})
    work.cells = list(nb.cells)
    for c in work.cells:
        c.setdefault("id", "")
    if args.cell:
        code_idx = 0
        keep = []
        for c in work.cells:
            if c.cell_type == "code":
                code_idx += 1
                if code_idx > args.cell:
                    continue
            keep.append(c)
        work.cells = keep

    # allow_errors=True lets a failing cell be reported without aborting;
    # --stop-on-error instead halts at the first failure (execute raises).
    client = NotebookClient(work, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}},
                            allow_errors=not args.stop_on_error)

    print(f"[lab] executing notebook: {NB}")
    print(f"[lab] cwd: {ROOT} | stop_on_error={args.stop_on_error} tag={args.tag or '-'}")

    try:
        client.execute()          # creates kernel + runs cells (execute() path works on Windows)
    except Exception as e:        # noqa: BLE001 — CellExecutionError when stop_on_error
        print(f"[lab] execution halted on error: {e}")
    finally:
        try:
            client.km.shutdown_kernel(now=True)
        except Exception:  # noqa: BLE001
            pass

    # Print captured outputs per code cell, and count any that errored.
    failed = 0
    code_idx = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        code_idx += 1
        if args.cell and code_idx > args.cell:
            break
        n_text = n_img = 0
        cell_err = False
        for out in cell.get("outputs", []):
            if out.output_type == "stream" and out.get("text"):
                sys.stdout.write("".join(out["text"]))
            elif out.output_type in ("execute_result", "display_data"):
                if "text/plain" in out.get("data", {}):
                    print("".join(out["data"]["text/plain"]))
                if "image/png" in out.get("data", {}):
                    n_img += 1
            elif out.output_type == "error":
                cell_err = True
                print("!! ERROR: " + "\n".join(out.get("traceback", []))[-1200:])
        if cell_err:
            failed += 1
            print(f"[CELL {code_idx:02d}] FAILED")
        else:
            if n_img:
                print(f"[CELL {code_idx:02d}] >> {n_img} figure(s) rendered")
            print(f"[CELL {code_idx:02d}] OK")

    nbf.write(nb, NB)
    print(f"\n[lab] done. Notebook updated with outputs: {NB}  (failed cells: {failed})")
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

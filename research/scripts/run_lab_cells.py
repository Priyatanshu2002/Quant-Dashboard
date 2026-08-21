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
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

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

    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})

    print(f"[lab] executing notebook: {NB}")
    print(f"[lab] cwd: {ROOT} | stop_on_error={args.stop_on_error} tag={args.tag or '-'}")

    # Execute the kernel (starts an ipykernel on this machine).
    # On the pod we run the same script; DEVICE auto-selects CUDA there.
    client.create_kernel_manager()
    client.start_new_kernel()
    try:
        client.kc.wait_for_ready()
    except Exception as e:  # noqa: BLE001
        print(f"[lab] kernel failed to start: {e}")
        return 2

    code_idx = 0
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            code_idx += 1
            if args.cell and code_idx > args.cell:
                break
            print(f"\n{'='*72}\n[CELL {code_idx:02d}] {cell.source.splitlines()[0][:70]}\n{'-'*72}")
            try:
                client.execute_cell(cell, code_idx)
            except Exception as e:  # noqa: BLE001
                print(f"[CELL {code_idx:02d}] EXECUTION ERROR: {e}")
                if args.stop_on_error:
                    nbf.write(nb, NB)
                    print(f"[lab] halted at cell {code_idx} — notebook saved.")
                    return 3
            # print captured stdout (and note figures)
            n_text = n_img = 0
            for out in cell.get("outputs", []):
                if out.output_type == "stream" and out.get("text"):
                    sys.stdout.write("".join(out["text"]) if out.get("name") == "stdout"
                                     else "".join(out["text"]))
                elif out.output_type in ("execute_result", "display_data"):
                    if "text/plain" in out.get("data", {}):
                        print("".join(out["data"]["text/plain"]))
                    if "text/html" in out.get("data", {}):
                        n_text += 1
                    if "image/png" in out.get("data", {}):
                        n_img += 1
                elif out.output_type == "error":
                    print("!! ERROR: " + "\n".join(out.get("traceback", []))[-1200:])
            if n_img:
                print(f"[CELL {code_idx:02d}] >> {n_img} figure(s) rendered")
            print(f"[CELL {code_idx:02d}] OK")
    finally:
        client.km.shutdown_kernel(now=True)

    nbf.write(nb, NB)
    print(f"\n[lab] done. Notebook updated with outputs: {NB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

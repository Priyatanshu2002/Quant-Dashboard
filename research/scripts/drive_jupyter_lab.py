"""Drive the research lab notebook through a LIVE Jupyter kernel (v2).

Connects to a kernel running on the live Jupyter server (which the user can
watch in a browser) via jupyter_client's BlockingKernelClient — the protocol
correct client. Executes each code cell, captures real stdout + figures, writes
them into the .ipynb, and pushes back to the server contents API so the browser
view updates live.

Usage:
    python research/scripts/drive_jupyter_lab.py [--cell N] [--from N]

Env:
    JP_BASE  : server base URL   (default http://127.0.0.1:8888)
    JP_TOKEN : token             (default agonistes)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import nbformat as nbf

JP_BASE = os.environ.get("JP_BASE", "http://127.0.0.1:8888")
JP_TOKEN = os.environ.get("JP_TOKEN", "agonistes")
NB_NAME = "01_empirical_research_lab.ipynb"


def _http(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{JP_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {JP_TOKEN}")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as r:  # noqa: S310
        raw = r.read()
    return json.loads(raw) if raw else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=None,
                    help="execute only this 1-based code-cell index")
    ap.add_argument("--from", dest="from_cell", type=int, default=1,
                    help="start from this 1-based code-cell index")
    args = ap.parse_args()

    # --- create a kernel on the live server (browser can attach to it) ---
    kernel = _http("POST", "/api/kernels", {"name": "python3"})
    kid = kernel["id"]
    print(f"[drive] created kernel {kid} on {JP_BASE}", flush=True)

    import jupyter_client  # noqa: PLC0415

    cf = jupyter_client.find_connection_file(kid)
    kc = jupyter_client.BlockingKernelClient(connection_file=cf)
    kc.load_connection_file()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=60)
    except Exception as e:  # noqa: BLE001
        print(f"[drive] kernel not ready: {e}")
        return 2
    print("[drive] kernel ready", flush=True)

    # --- load notebook from the server ---
    nb = _http("GET", f"/api/contents/{urllib.parse.quote(NB_NAME)}")
    nb_json = nb["content"]

    # --- execute cells ---
    code_idx = 0
    try:
        for cell in nb_json["cells"]:
            if cell["cell_type"] != "code":
                continue
            code_idx += 1
            if code_idx < args.from_cell:
                continue
            if args.cell and code_idx != args.cell:
                continue
            head = cell["source"].strip().splitlines()[0][:70] if cell["source"] else "(blank)"
            print(f"\n{'='*70}\n[CELL {code_idx:02d}] {head}\n{'-'*70}", flush=True)

            cell["outputs"] = []
            cell["execution_count"] = code_idx
            msg_id = kc.execute(cell["source"], stop_on_error=True, store_history=True)

            # collect outputs until idle
            n_img = 0
            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=180)
                except Exception:
                    print(f"[CELL {code_idx:02d}] timed out waiting for iopub")
                    break
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                mtype = msg["header"]["msg_type"]
                content = msg["content"]
                if mtype == "stream":
                    cell["outputs"].append({"output_type": "stream",
                                            "name": content["name"],
                                            "text": content["text"]})
                elif mtype == "execute_result":
                    cell["outputs"].append({"output_type": "execute_result",
                                            "execution_count": content["execution_count"],
                                            "data": content["data"], "metadata": {}})
                elif mtype == "display_data":
                    cell["outputs"].append({"output_type": "display_data",
                                            "data": content["data"], "metadata": {}})
                elif mtype == "error":
                    cell["outputs"].append({"output_type": "error",
                                            "ename": content["ename"],
                                            "evalue": content["evalue"],
                                            "traceback": content["traceback"]})
                elif mtype == "status" and content["execution_state"] == "idle":
                    break

            # print captured text so the agent can verify
            for out in cell["outputs"]:
                if out["output_type"] == "stream":
                    print("".join(out["text"]), end="")
                elif out["output_type"] in ("execute_result", "display_data"):
                    if "text/plain" in out.get("data", {}):
                        print("".join(out["data"]["text/plain"]))
                    if "image/png" in out.get("data", {}):
                        n_img += 1
                        print(f"[{len(out['data']['image/png'])} B PNG rendered]")
                elif out["output_type"] == "error":
                    print("!! ERROR: " + "\n".join(out.get("traceback", []))[-1200:])
            print(f"[CELL {code_idx:02d}] OK" + (f" ({n_img} figures)" if n_img else ""),
                  flush=True)

            # push outputs back to the server → live browser update
            _http("PUT", f"/api/contents/{urllib.parse.quote(NB_NAME)}",
                  {"type": "notebook", "content": nb_json, "format": "json"})
    finally:
        kc.stop_channels()

    print(f"\n[drive] done. {code_idx} code cells executed on live Jupyter kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

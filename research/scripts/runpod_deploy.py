"""Provision a RunPod GPU pod and deploy the Agonistes empirical research lab.

Control stays on this PC; computation happens on the pod. The pod runs the SAME
6-stage notebook (build_01_lab_notebook.py) in FULL mode, using CUDA via the
trainer's default_device(). We then pull data/benchmark/* artifacts back.

Usage:
    set RUNPOD_API_KEY=<your-key>          # required
    python research/scripts/runpod_deploy.py [--gpu RTX4090] [--cloud SECURE]
                                            [--ssh-key ~/.ssh/id_ed25519.pub]
                                            [--skip-provision --pod-id P]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

# Files to ship to the pod (code + db + notebook + runner). No secrets/.env.
DEPLOY_PATHS = [
    "strategy_builder",
    "core",
    "data/agonistes_dev.db",
    "research/notebooks/01_empirical_research_lab.ipynb",
    "research/scripts/build_01_lab_notebook.py",
    "research/scripts/run_lab_cells.py",
    "pyproject.toml",
]

POD_SETUP = r"""#!/bin/bash
set -e
pip install --quiet torch numpy pandas scikit-learn xgboost lightgbm catboost arch \
    matplotlib nbformat nbconvert ipykernel
# torch CUDA build (pod image already has torch; reinstall not needed). If the
# base image lacks torch, uncomment:  pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
cd /workspace
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo none)"
"""


def _run(args, **kw):
    return subprocess.run([str(PY), *args], capture_output=True, text=True, **kw)


def provision(gpu: str, cloud: str, ssh_pub: str) -> dict:
    env = {"RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"]}
    env_py = "import os,runpod\nk=os.environ['RUNPOD_API_KEY']\n"
    ssh_arg = ""
    if ssh_pub:
        ssh_arg = f", ssh_public_keys=[{Path(ssh_pub).read_text().strip()!r}]"
    code = (env_py +
            "p=runpod.create_pod("
            "name='agonistes-lab',"
            "image_name='runpod/pytorch:2.6.0-py3.11-cuda12.4.1-cudnn9.2.0',"
            f"gpu_type_id={gpu!r},cloud_type={cloud!r},"
            f"gpu_count=1,container_disk_in_gb=30,start_ssh=True"
            f"{ssh_arg}"
            ")\n"
            "print(p['id']); print(p.get('runtime',{}).get('ports',{}))")
    r = subprocess.run([str(PY), "-c", code], capture_output=True, text=True,
                       env={**os.environ, **env})
    if r.returncode != 0:
        raise RuntimeError(f"provision failed:\n{r.stderr}")
    lines = r.stdout.strip().splitlines()
    return {"id": lines[0] if lines else "", "ports": lines[1] if len(lines) > 1 else ""}


def package() -> Path:
    tarpath = ROOT / "data" / "benchmark" / "agonistes_lab.tar.gz"
    with tarfile.open(tarpath, "w:gz") as tf:
        for p in DEPLOY_PATHS:
            src = ROOT / p
            if not src.exists():
                print(f"  ! missing {p}, skipping")
                continue
            if src.is_dir():
                tf.add(src, arcname=p)
            else:
                tf.add(src, arcname=p)
    return tarpath


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTX4090", help="RunPod GPU type")
    ap.add_argument("--cloud", default="SECURE", help="cloud_type (SECURE/COMMUNITY/ALL)")
    ap.add_argument("--ssh-key", default=None, help="path to your ssh public key")
    ap.add_argument("--skip-provision", action="store_true",
                    help="already have a pod; pass --pod-id")
    ap.add_argument("--pod-id", default=None)
    args = ap.parse_args()

    if "RUNPOD_API_KEY" not in os.environ or not os.environ["RUNPOD_API_KEY"]:
        print("FATAL: RUNPOD_API_KEY is not set. "
              "Run `export RUNPOD_API_KEY=<your key>` first.")
        return 1

    print("[1/4] packaging code + db + notebook ...")
    tarpath = package()
    print(f"  -> {tarpath} ({tarpath.stat().st_size/1e6:.1f} MB)")

    pod_id = args.pod_id
    if not args.skip_provision:
        print(f"[2/4] provisioning {args.gpu} pod ...")
        info = provision(args.gpu, args.cloud, args.ssh_key)
        pod_id = info["id"]
        print(f"  pod {pod_id} | ports {info['ports']}")
        print("  NOTE: Wait ~2-4 min for the pod to be READY before connecting.")
    else:
        print(f"[2/4] using existing pod {pod_id}")

    print("[3/4] upload + run ...")
    print(f"  scp -P 22 {tarpath} root@<pod-ip>:/workspace/")
    print("  Run POD_SETUP then: cd /workspace && RUN_MODE=full "
          "python research/scripts/run_lab_cells.py --stop-on-error")
    print("  (See runpod_README.md for the full steps and the data pull-back.)")

    print("[4/4] done.")
    print(f"Pod: {pod_id} | artifacts -> data/benchmark/ on the pod.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# RunPod GPU Deployment — Agonistes Empirical Research Lab

Control stays on your PC; the **computation runs on a RunPod GPU pod**. The pod
runs the same 6-stage notebook (`research/notebooks/01_empirical_research_lab.ipynb`)
in FULL mode using CUDA (`strategy_builder/trainer.py` is device-aware).

## Prerequisites
- A RunPod account + API key.
- SSH public key (`~/.ssh/id_ed25519.pub`) — already on this machine.

## One-time: give the agent your key
```
export RUNPOD_API_KEY=rpa_XXXXXXXXXXXXXXXXXXXXXX
```
(This is the ONLY thing currently missing — the SDK, code, notebook, and DB are
all staged and ready.)

## Deploy
```
# Option A — agent provisions the pod (needs the API key)
python research/scripts/runpod_deploy.py --gpu RTX4090 --ssh-key ~/.ssh/id_ed25519.pub

# Option B — you already created a pod; agent connects over SSH
python research/scripts/runpod_deploy.py --skip-provision --pod-id <pod-id>
```

## On the pod
After upload (agent runs the tar + setup script automatically):
```
cd /workspace
RUN_MODE=full python research/scripts/run_lab_cells.py --stop-on-error
```
This executes the 18 code cells cell-by-cell (Stage 1 → Stage 6), prints real
outputs, and writes `data/benchmark/weights_*.csv`, `leaderboard.json`,
`breakeven.json` on the pod.

## Pull results back to your PC
```
scp -r root@<pod-ip>:/workspace/data/benchmark ./data/benchmark/
```

## GPU choice
- RTX 4090 — best $/perf for the transformer/LSTM models here.
- RTX A6000 / L40 — more VRAM if you scale batch/lookback later.
- Pick SECURE cloud for guaranteed availability; COMMUNITY is cheaper but
  preemptible.

## Notes
- The pod image (`runpod/pytorch:2.6.0-py3.11-cuda12.4.1-cudnn9.2.0`) already has
  CUDA torch. If you use a base image without torch, uncomment the torch install
  in `runpod_deploy.py` `POD_SETUP`.
- No secrets / `.env` are shipped to the pod — only code + DB + notebook.

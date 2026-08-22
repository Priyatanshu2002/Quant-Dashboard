"""Training loop: pooled-Sharpe loss, walk-forward windows, seed ensembling.

Implements the Oxford protocol (arXiv:2603.01820 §2.1):
  * windowed samples: x = features[t-L+1..t], next-day raw return r_{t+1}
  * position signal y = tanh(W h + b) in [-1, 1]
  * portfolio weight w = y * (sigma_tgt / sigma)  (volatility targeting)
  * loss = -annualized Sharpe of pooled portfolio returns across the batch
  * Adam + gradient clipping + early stopping on validation Sharpe
  * expanding-window walk-forward; per-seed models ensembled by validation
    performance (paper: top-S of M seeds)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from strategy_builder.models import SignalHead, build_encoder


def default_device() -> torch.device:
    """CUDA if available, else CPU. Overridable via train_model(device=...)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignalModel(torch.nn.Module):
    """Encoder + linear-tanh head → position signal in [-1, 1]."""

    def __init__(self, model_name: str, n_feat: int, hidden: int, lookback: int,
                 n_assets: int, use_ticker_emb: bool = True):
        super().__init__()
        self.encoder = build_encoder(model_name, in_dim=n_feat, hidden=hidden,
                                     lookback=lookback, n_assets=n_assets,
                                     use_ticker_emb=use_ticker_emb)
        self.head = SignalHead(hidden)

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, ticker_ids))


def pooled_sharpe_loss(signals: torch.Tensor, next_ret: torch.Tensor,
                       vs_factor: torch.Tensor, sigma_tgt: float = 0.10) -> torch.Tensor:
    """Negative annualized Sharpe of the pooled vol-targeted portfolio (eq. 10).

    signals: (B,) positions in [-1,1]; next_ret: (B,) next-day raw returns;
    vs_factor: (B,) 1/sigma. Portfolio return per sample = signal * sigma_tgt * vs * ret.
    """
    port_ret = signals * sigma_tgt * vs_factor * next_ret
    mean = port_ret.mean()
    var = port_ret.var(unbiased=False) + 1e-8
    sharpe = mean / torch.sqrt(var) * math.sqrt(252)
    return -sharpe


class WindowedDataset:
    """Pre-computed windowed samples from a long-format feature panel."""

    def __init__(self, panel: pd.DataFrame, feature_cols: list[str],
                 lookback: int, symbols: list[str]):
        self.lookback = lookback
        self.symbols = symbols
        sym_id = {s: i for i, s in enumerate(symbols)}
        xs, rs, vs, ts, tms, sms = [], [], [], [], [], []
        for sym, g in panel.groupby("symbol", sort=False):
            g = g.sort_values("time")
            feats = g[feature_cols].to_numpy(dtype=np.float32)
            rets = g["ret_1"].shift(-1).to_numpy(dtype=np.float32)   # r_{t+1}
            vsf = g["vs_factor"].to_numpy(dtype=np.float32)
            sid = sym_id[sym]
            for t in range(lookback, len(g) - 1):
                xs.append(feats[t - lookback + 1: t + 1])
                rs.append(rets[t])
                vs.append(vsf[t])
                ts.append(sid)
                tms.append(g["time"].iloc[t])
                sms.append(sym)
        self.x = np.stack(xs)
        self.r = np.asarray(rs, dtype=np.float32)
        self.v = np.asarray(vs, dtype=np.float32)
        self.t = np.asarray(ts, dtype=np.int64)
        self.times = tms
        self.syms = sms

    def __len__(self) -> int:
        return len(self.r)

    def to_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(self.x), torch.from_numpy(self.r),
                           torch.from_numpy(self.v), torch.from_numpy(self.t))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, drop_last=shuffle)


def _validate(model: SignalModel, loader: DataLoader, sigma_tgt: float,
              device: torch.device | None = None) -> float:
    """Mean pooled Sharpe over validation batches (higher = better)."""
    device = device or default_device()
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, rb, vb, tb in loader:
            y = model(xb.to(device), tb.to(device))
            total -= float(pooled_sharpe_loss(y, rb.to(device), vb.to(device), sigma_tgt))
            n += 1
    return total / max(n, 1)


def train_model(model_name: str, train_panel: pd.DataFrame, val_panel: pd.DataFrame,
                feature_cols: list[str], symbols: list[str],
                lookback: int = 64, hidden: int = 32, batch_size: int = 256,
                epochs: int = 60, lr: float = 1e-3, seed: int = 0,
                sigma_tgt: float = 0.10, patience: int = 10,
                grad_clip: float = 1.0, use_ticker_emb: bool = True,
                verbose: bool = False,
                device: torch.device | None = None) -> tuple[SignalModel, float, int]:
    """Train one SignalModel on the pooled-Sharpe objective.

    Returns (model, best_val_sharpe, best_epoch).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_feat = len(feature_cols)
    device = device or default_device()

    train_ds = WindowedDataset(train_panel, feature_cols, lookback, symbols)
    val_ds = WindowedDataset(val_panel, feature_cols, lookback, symbols)
    if len(train_ds) < 64 or len(val_ds) < 32:
        raise ValueError(f"{model_name}: too few samples train={len(train_ds)} val={len(val_ds)}")

    model = SignalModel(model_name, n_feat=n_feat, hidden=hidden, lookback=lookback,
                        n_assets=len(symbols), use_ticker_emb=use_ticker_emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = train_ds.to_loader(batch_size, shuffle=True)
    val_loader = val_ds.to_loader(batch_size, shuffle=False)

    best_sharpe, best_epoch, bad = -1e9, 0, 0
    for ep in range(epochs):
        model.train()
        loss = None
        for xb, rb, vb, tb in train_loader:
            opt.zero_grad()
            y = model(xb.to(device), tb.to(device))
            loss = pooled_sharpe_loss(y, rb.to(device), vb.to(device), sigma_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        val_sharpe = _validate(model, val_loader, sigma_tgt, device)
        if val_sharpe > best_sharpe:
            best_sharpe, best_epoch, bad = val_sharpe, ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    {model_name} seed={seed} ep={ep} train_loss={float(loss):.3f} "
                  f"val_sharpe={val_sharpe:.3f}")
    return model, best_sharpe, best_epoch


def walk_forward_windows(panel: pd.DataFrame, train_months: int = 36,
                         test_months: int = 6, embargo_days: int = 5) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window walk-forward splits over the panel's calendar.

    embargo_days: a purge buffer dropped from the end of each training window so
    no validation observation's label (next-day return) leaks into training.
    This mirrors the purged walk-forward protocol of the reference research
    (Desktop/Research Notebooks and Data, 60-bar embargo) and removes the
    otherwise-ubiquitous 1-day label overlap at each train/val boundary.
    """
    start, end = panel["time"].min(), panel["time"].max()
    windows = []
    cursor = start + pd.DateOffset(months=train_months)
    while cursor + pd.DateOffset(months=test_months) <= end:
        tr_end, te_end = cursor, cursor + pd.DateOffset(months=test_months)
        tr_end_emb = tr_end - pd.Timedelta(days=embargo_days)
        tr = panel[(panel["time"] >= start) & (panel["time"] < tr_end_emb)]
        va = panel[(panel["time"] >= tr_end) & (panel["time"] < te_end)]
        if len(tr) > 500 and len(va) > 100:
            windows.append((tr, va))
        cursor += pd.DateOffset(months=test_months)
    return windows


def run_benchmark_model(model_name: str, panel: pd.DataFrame, feature_cols: list[str],
                        symbols: list[str], lookback: int = 64, hidden: int = 32,
                        seeds: int = 3, top_seeds: int = 2, epochs: int = 60,
                        batch_size: int = 256, lr: float = 1e-3,
                        train_months: int = 36, test_months: int = 6,
                        sigma_tgt: float = 0.10, use_ticker_emb: bool = True,
                        verbose: bool = False,
                        device: torch.device | None = None) -> dict:
    """Walk-forward train + predict; returns ensembled OOS weights + validation log."""
    device = device or default_device()
    windows = walk_forward_windows(panel, train_months, test_months)
    weight_frames: list[pd.DataFrame] = []
    val_log: list[dict] = []
    for wi, (tr, va) in enumerate(windows):
        tr = tr.copy()
        seeds_scores: list[tuple[float, SignalModel]] = []
        for s in range(seeds):
            model, vsharpe, _ = train_model(
                model_name, tr, va, feature_cols, symbols, lookback=lookback,
                hidden=hidden, batch_size=batch_size, epochs=epochs, lr=lr,
                seed=s, sigma_tgt=sigma_tgt, use_ticker_emb=use_ticker_emb,
                verbose=verbose, device=device)
            seeds_scores.append((vsharpe, model))
            val_log.append({"model": model_name, "window": wi, "seed": s,
                            "val_sharpe": round(vsharpe, 4)})
        seeds_scores.sort(key=lambda t: t[0], reverse=True)
        best = seeds_scores[:top_seeds]
        # ensemble: average position signals of the top-S seeds
        ds = WindowedDataset(va, feature_cols, lookback, symbols)
        loader = ds.to_loader(batch_size, shuffle=False)
        y_all: list[np.ndarray] = []
        with torch.no_grad():
            for xb, rb, vb, tb in loader:
                y_sum = torch.zeros(len(xb), device=device)
                for _, m in best:
                    m.eval()
                    y_sum = y_sum + m(xb.to(device), tb.to(device))
                y_all.append((y_sum / len(best)).cpu().numpy())
        y_ens = np.concatenate(y_all)
        # align with the dataset's own (time, symbol) rows
        out_rows = []
        for i, y in enumerate(y_ens):
            out_rows.append({"time": ds.times[i], "symbol": ds.syms[i],
                             "weight": float(y * sigma_tgt * ds.v[i])})
        weight_frames.append(pd.DataFrame(out_rows))
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame(
        columns=["time", "symbol", "weight"])
    return {"weights": weights, "val_log": val_log, "windows": len(windows)}

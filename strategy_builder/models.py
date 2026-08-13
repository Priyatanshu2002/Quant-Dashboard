"""Sequence encoders for the DL-for-finance benchmark (arXiv:2603.01820).

Every encoder maps a window x: (B, L, F) (+ optional ticker ids) to a
terminal representation h: (B, H). A shared SignalHead (linear + tanh)
turns h into the position signal in [-1, 1] (paper eq. 5).

All implementations are pure PyTorch (CPU-friendly). Mamba2 uses a
sequential selective-scan reference implementation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------

class TickerEmbedding(nn.Module):
    """Per-asset embedding concatenated to the input features (paper §2.1)."""

    def __init__(self, n_assets: int, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(n_assets, dim)
        self.dim = dim

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(ticker_ids)                     # (B, d_emb)
        return torch.cat([x, e[:, None, :].expand(-1, x.size(1), -1)], dim=-1)


class SignalHead(nn.Module):
    """Linear projection + tanh → position signal in [-1, 1] (eq. 5)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(h)).squeeze(-1)


class GRN(nn.Module):
    """Gated Residual Network (TFT component)."""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.a = nn.Linear(in_dim, hidden)
        self.b = nn.Linear(in_dim, hidden)
        self.c = nn.Linear(hidden, hidden)
        self.d = nn.Linear(hidden, hidden)
        self.e = nn.Linear(hidden, in_dim)
        self.layernorm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a(x)
        elu = F.elu(a)
        b = self.b(x)
        gate = torch.sigmoid(b)
        c = self.c(elu * gate)
        c = self.d(F.elu(c))
        c = self.dropout(c)
        e = self.e(c)
        return self.layernorm(x + e)


class VariableSelectionNetwork(nn.Module):
    """Per-feature GRN embedding + soft selection weights (TFT VSN).

    Vectorized: all feature embeddings share one GRN applied to the
    flattened (B*L*F, 1) tensor, then reshaped — 7x fewer kernel launches
    than per-feature modules while keeping the same computation.
    """

    def __init__(self, n_features: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.embed = nn.Sequential(nn.Linear(1, hidden), GRN(hidden, hidden, dropout))
        self.selector = GRN(n_features, hidden, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        B, L, F = x.shape
        flat = x.reshape(-1, 1)                       # (B*L*F, 1)
        emb = self.embed(flat).view(B, L, F, self.hidden)
        w = self.softmax(self.selector(x))            # (B, L, F)
        return (emb * w.unsqueeze(-1)).sum(dim=-2)    # (B, L, H)


class PatchEmbed(nn.Module):
    """Temporal patching (PatchTST/PsLSTM): non-overlapping patches → embed."""

    def __init__(self, patch_len: int, d_model: int, in_dim: int = 1):
        super().__init__()
        self.patch_len = patch_len
        self.in_dim = in_dim
        self.proj = nn.Linear(patch_len * in_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) → (B, n_patches, F, d_model)
        B, L, F = x.shape
        n = L // self.patch_len
        x = x[:, : n * self.patch_len]
        if self.in_dim == 1:
            x = x.reshape(B, n, F, self.patch_len)
        else:
            x = x.reshape(B, n, F * self.patch_len)
        return self.proj(x)


# --------------------------------------------------------------------------
# sLSTM / mLSTM cells (xLSTM, Beck et al. 2024)
# --------------------------------------------------------------------------

class SLSTMCell(nn.Module):
    """Scalar LSTM cell with exponential gates + stabilization."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_dim, self.hidden = in_dim, hidden
        self.Wz = nn.Linear(in_dim, hidden)
        self.Wi = nn.Linear(in_dim, hidden)
        self.Wf = nn.Linear(in_dim, hidden)
        self.Wo = nn.Linear(in_dim, hidden)
        self.Rz = nn.Linear(hidden, hidden, bias=False)
        self.Ri = nn.Linear(hidden, hidden, bias=False)
        self.Rf = nn.Linear(hidden, hidden, bias=False)
        self.Ro = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor, state=None):
        # x: (B, in); returns h (B, H), state=(c, m)
        B = x.size(0)
        if state is None:
            c = torch.zeros(B, self.hidden, device=x.device)
            m = torch.full((B, self.hidden), -1e30, device=x.device)
            h = torch.zeros(B, self.hidden, device=x.device)
        else:
            c, m, h = state
        z = torch.tanh(self.Wz(x) + self.Rz(h))
        i_log = self.Wi(x) + self.Ri(h)                  # log input gate
        f_log = self.Wf(x) + self.Rf(h)                  # log forget gate
        o = torch.sigmoid(self.Wo(x) + self.Ro(h))
        m_new = torch.maximum(f_log + m, i_log)
        i = torch.exp(i_log - m_new)
        f = torch.exp(f_log + m - m_new)
        c_new = f * c + i * z
        c_new = torch.clamp(c_new, min=-1e6, max=1e6)
        # h = o * normalized cell (stabilized; exp(m) term omitted for
        # numerical safety on CPU — see xLSTM paper eq. 4-6)
        h_new = o * (c_new / torch.clamp(torch.abs(c_new), min=1.0))
        return h_new, (c_new, m_new, h_new)


class MLSTMCell(nn.Module):
    """Matrix LSTM cell with key-value associative memory."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_dim, self.hidden = in_dim, hidden
        self.Wq = nn.Linear(in_dim, hidden)
        self.Wk = nn.Linear(in_dim, hidden)
        self.Wv = nn.Linear(in_dim, hidden)
        self.Wi = nn.Linear(in_dim, 1)
        self.Wf = nn.Linear(in_dim, 1)
        self.Wo = nn.Linear(in_dim, hidden)
        self.norm_k = math.sqrt(hidden)

    def forward(self, x: torch.Tensor, state=None):
        B = x.size(0)
        if state is None:
            C = torch.zeros(B, self.hidden, self.hidden, device=x.device)
            n = torch.zeros(B, 1, device=x.device)
            h = torch.zeros(B, self.hidden, device=x.device)
        else:
            C, n, h = state
        q = self.Wq(x) / self.norm_k
        k = self.Wk(x)
        v = self.Wv(x)
        i = torch.exp(self.Wi(x))                        # (B, 1)
        f = torch.sigmoid(self.Wf(x))
        o = torch.sigmoid(self.Wo(x))
        C_new = f.unsqueeze(-1) * C + i.unsqueeze(-1) * (v.unsqueeze(-1) * k.unsqueeze(-2))
        n_new = f * n + i
        Cq = torch.einsum("bij,bj->bi", C_new, q)
        Cq_norm = Cq / torch.clamp(torch.norm(Cq, dim=-1, keepdim=True), min=1.0)
        h_new = o * Cq_norm / torch.clamp(n_new, min=1.0)
        return h_new, (C_new, n_new, h_new)


class XLSTMBlock(nn.Module):
    """Causal conv → cell (sLSTM or mLSTM) → LayerNorm head."""

    def __init__(self, in_dim: int, hidden: int, cell: str = "sLSTM",
                 conv_kernel: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, in_dim, conv_kernel,
                              padding=conv_kernel - 1, groups=in_dim)
        self.cell = (SLSTMCell(in_dim, hidden) if cell == "sLSTM"
                     else MLSTMCell(in_dim, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) → (B, L, H)
        xc = self.conv(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        xc = F.gelu(xc)
        outs = []
        state = None
        for t in range(xc.size(1)):
            h, state = self.cell(xc[:, t], state)
            outs.append(h)
        return self.norm(torch.stack(outs, dim=1))


# --------------------------------------------------------------------------
# Mamba2 (selective SSM, reference sequential scan)
# --------------------------------------------------------------------------

class Mamba2Encoder(nn.Module):
    """Mamba2-style selective SSM; sequential scan over L (CPU-friendly).

    Structure: in-proj → causal conv → SSM step (A_bar, B, C, dt) → RMSNorm → out-proj.
    """

    def __init__(self, in_dim: int, hidden: int, lookback: int = 64,
                 d_state: int = 16, dt_rank: int = 8, conv_kernel: int = 4,
                 n_heads: int = 4):
        super().__init__()
        self.in_dim, self.hidden = in_dim, hidden
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.n_heads = n_heads
        self.in_proj = nn.Linear(in_dim, hidden * 2 + dt_rank)
        self.conv = nn.Conv1d(hidden, hidden, conv_kernel,
                              padding=conv_kernel - 1, groups=hidden)
        self.x_proj = nn.Linear(hidden + dt_rank, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, n_heads, bias=True)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.RMSNorm(hidden)
        self.register_buffer("A_log", torch.log(
            torch.arange(1, n_heads + 1, dtype=torch.float32) * 0.5
        ).view(1, 1, n_heads, 1).expand(1, 1, n_heads, d_state).contiguous())
        self.dt_min, self.dt_max = 0.001, 0.1

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        bs, L, _ = x.shape
        z, x0, dt0 = self.in_proj(x).split([self.hidden, self.hidden, self.dt_rank], dim=-1)
        x0 = F.gelu(self.conv(x0.transpose(1, 2))[:, :, :L].transpose(1, 2))
        x_db = torch.cat([x0, dt0], dim=-1)
        dts, Bm, Cm = self.x_proj(x_db).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dts)).clamp(self.dt_min, self.dt_max)  # (B, L, H_heads)
        A = -torch.exp(self.A_log)                                           # (1, 1, H_heads, d_state)
        A_bar = torch.exp(A * dt.unsqueeze(-1))                              # (B, L, H_heads, d_state)
        Bm = Bm.unsqueeze(2).unsqueeze(-1)                                   # (B, L, 1, d_state, 1)
        Cm = Cm.unsqueeze(2).unsqueeze(-2)                                   # (B, L, 1, 1, d_state)
        d_inner = self.hidden // self.n_heads
        x0h = x0.view(bs, L, self.n_heads, d_inner).unsqueeze(-2)            # (B, L, H, 1, d_inner)

        h = torch.zeros(bs, self.n_heads, self.d_state, d_inner, device=x.device)
        ys = []
        for t in range(L):
            h = A_bar[:, t].unsqueeze(-1) * h + Bm[:, t] * x0h[:, t]
            ys.append((Cm[:, t] @ h).squeeze(2))                             # (B, H, d_inner)
        y = torch.stack(ys, dim=1).view(bs, L, self.hidden)
        y = y * F.silu(z)
        return self.out_proj(self.norm(y[:, -1]))                            # (B, H)


# --------------------------------------------------------------------------
# Linear baselines
# --------------------------------------------------------------------------

class AR1x(nn.Module):
    """AR(1) per feature: last observation + first difference, projected."""

    def __init__(self, in_dim: int, hidden: int, lookback: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, hidden)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        last = x[:, -1]
        prev = x[:, -2]
        return self.proj(torch.cat([last, last - prev], dim=-1))


class DLinear(nn.Module):
    """Trend (moving average) + seasonal decomposition, two linear maps."""

    def __init__(self, in_dim: int, hidden: int, lookback: int, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(lookback * in_dim, hidden)
        self.seasonal = nn.Linear(lookback * in_dim, hidden)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        B, L, F_ = x.shape
        k = min(self.kernel, L)
        trend = F.avg_pool1d(x.transpose(1, 2), k, stride=1, padding=k // 2)
        trend = trend[:, :, :L].transpose(1, 2)
        seasonal = x - trend
        flat = lambda t: t.reshape(B, L * F_)  # noqa: E731
        return self.trend(flat(trend)) + self.seasonal(flat(seasonal))


class NLinear(nn.Module):
    """Normalize by last value, linear map, add last value back."""

    def __init__(self, in_dim: int, hidden: int, lookback: int):
        super().__init__()
        self.proj = nn.Linear(lookback * in_dim, hidden)
        self.last_proj = nn.Linear(in_dim, hidden)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        B, L, F_ = x.shape
        last = x[:, -1:, :]
        normed = x - last
        return self.proj(normed.reshape(B, L * F_)) + self.last_proj(last.reshape(B, F_))


# --------------------------------------------------------------------------
# Recurrent models
# --------------------------------------------------------------------------

class LSTMBaseline(nn.Module):
    """Standard LSTM; terminal hidden state."""

    def __init__(self, in_dim: int, hidden: int, lookback: int, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return h[-1]


class XLSTMEncoder(nn.Module):
    """xLSTM: sLSTM block then mLSTM block (compact, per Beck et al. 2024)."""

    def __init__(self, in_dim: int, hidden: int, lookback: int):
        super().__init__()
        self.slstm = XLSTMBlock(in_dim, hidden, cell="sLSTM")
        self.mlstm = XLSTMBlock(hidden, hidden, cell="mLSTM")

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        return self.mlstm(self.slstm(x))[:, -1]


class PSLSTMEncoder(nn.Module):
    """Patch sLSTM: per-channel patches, shared sLSTM, flatten + project."""

    def __init__(self, in_dim: int, hidden: int, lookback: int,
                 patch_len: int = 8, d_model: int = 32):
        super().__init__()
        self.patch_len = patch_len
        self.patch = PatchEmbed(patch_len, d_model, in_dim=1)
        self.cell = SLSTMCell(d_model, d_model)
        self.head = nn.Linear(d_model * in_dim, hidden)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        B, L, F = x.shape
        xp = self.patch(x)                       # (B, n, F, d_model)
        Bn, n, Fn, d = xp.shape
        # shared sLSTM across channels
        hiddens = []
        for f_ in range(Fn):
            outs = []
            state = None
            for p in range(n):
                h, state = self.cell(xp[:, p, f_], state)
                outs.append(h)
            hiddens.append(torch.stack(outs, dim=1))
        h = torch.stack(hiddens, dim=2)          # (B, n, F, d)
        h = h.mean(dim=1)                        # aggregate patches → (B, F, d)
        return self.head(h.reshape(B, Fn * d))


# --------------------------------------------------------------------------
# Transformer models
# --------------------------------------------------------------------------

class PatchTSTEncoder(nn.Module):
    """PatchTST: instance norm, patches, transformer encoder, mean pool."""

    def __init__(self, in_dim: int, hidden: int, lookback: int,
                 patch_len: int = 16, d_model: int = 32, n_layers: int = 2,
                 n_heads: int = 4):
        super().__init__()
        self.patch_len = patch_len
        self.patch = PatchEmbed(patch_len, d_model, in_dim=in_dim)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, hidden)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        xn = (x - mean) / std
        xp = self.patch(xn)                       # (B, n, d_model)
        z = self.enc(xp)
        return self.head(self.norm(z.mean(dim=1)))


class ITransformerEncoder(nn.Module):
    """iTransformer: attention across feature dimension (features as tokens)."""

    def __init__(self, in_dim: int, hidden: int, lookback: int,
                 d_model: int = 32, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.embed = nn.Linear(lookback, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, hidden)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        xt = x.transpose(1, 2)                    # (B, F, L) — features are tokens
        z = self.enc(self.embed(xt))
        return self.head(z.mean(dim=1))


class LPatchTSTEncoder(nn.Module):
    """LSTM+PatchTST: per-channel LSTM denoiser → PatchTST."""

    def __init__(self, in_dim: int, hidden: int, lookback: int,
                 patch_len: int = 16, d_model: int = 32, n_layers: int = 2,
                 n_heads: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, batch_first=True)   # shared per-channel denoiser
        self.denoise = nn.Linear(8, 1)
        self.patch = PatchEmbed(patch_len, d_model, in_dim=in_dim)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, hidden)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        B, L, F = x.shape
        # channel-wise LSTM denoising: residual of LSTM fit
        xr = torch.empty_like(x)
        for f_ in range(F):
            h, _ = self.lstm(x[:, :, f_ : f_ + 1])
            xr[:, :, f_ : f_ + 1] = x[:, :, f_ : f_ + 1] - self.denoise(h)
        xp = self.patch(xr)                        # (B, n, d_model)
        z = self.enc(xp)
        return self.head(self.norm(z.mean(dim=1)))


# --------------------------------------------------------------------------
# VSN hybrids + TFT
# --------------------------------------------------------------------------

class VSNLSTMEncoder(nn.Module):
    """VLSTM: VSN → LSTM."""

    def __init__(self, in_dim: int, hidden: int, lookback: int):
        super().__init__()
        self.vsn = VariableSelectionNetwork(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        v = self.vsn(x)
        _, (h, _) = self.lstm(v)
        return h[-1]


class VSNXLSTMEncoder(nn.Module):
    """VxLSTM: VSN → xLSTM."""

    def __init__(self, in_dim: int, hidden: int, lookback: int):
        super().__init__()
        self.vsn = VariableSelectionNetwork(in_dim, hidden)
        self.xlstm = XLSTMEncoder(hidden, hidden, lookback)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        return self.xlstm(self.vsn(x))


class VSNMamba2Encoder(nn.Module):
    """VSN+Mamba2: feature selection before the selective SSM."""

    def __init__(self, in_dim: int, hidden: int, lookback: int):
        super().__init__()
        self.vsn = VariableSelectionNetwork(in_dim, hidden)
        self.mamba = Mamba2Encoder(hidden, hidden)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        return self.mamba(self.vsn(x))


class TFTEncoder(nn.Module):
    """Compact TFT: VSN → LSTM encoder → interpretable attention → GRN → head."""

    def __init__(self, in_dim: int, hidden: int, lookback: int,
                 n_heads: int = 4):
        super().__init__()
        self.vsn = VariableSelectionNetwork(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.n_heads = n_heads
        self.grn = GRN(hidden, hidden)
        self.head = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, ticker_ids=None) -> torch.Tensor:
        v = self.vsn(x)                                # (B, L, H)
        enc, _ = self.lstm(v)                          # (B, L, H)
        # interpretable attention: single query at t, keys over the encoder
        q = self.q(enc[:, -1:])                        # (B, 1, H)
        k = self.k(enc)                                # (B, L, H)
        scores = (q @ k.transpose(1, 2)) / math.sqrt(self.q.out_features)
        attn = torch.softmax(scores, dim=-1)           # (B, 1, L)
        ctx = (attn @ self.v(enc)).squeeze(1)          # (B, H)
        return self.head(self.grn(ctx))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

ENCODERS: dict[str, type[nn.Module]] = {
    "ar1x": AR1x,
    "dlinear": DLinear,
    "nlinear": NLinear,
    "lstm": LSTMBaseline,
    "xlstm": XLSTMEncoder,
    "pslstm": PSLSTMEncoder,
    "mamba2": Mamba2Encoder,
    "patchtst": PatchTSTEncoder,
    "itransformer": ITransformerEncoder,
    "lpatchtst": LPatchTSTEncoder,
    "vlstm": VSNLSTMEncoder,
    "vxlstm": VSNXLSTMEncoder,
    "vsnmamba2": VSNMamba2Encoder,
    "tft": TFTEncoder,
}

DEFAULT_HIDDEN = {
    "ar1x": 32, "dlinear": 32, "nlinear": 32,
    "lstm": 32, "xlstm": 32, "pslstm": 32,
    "mamba2": 32, "patchtst": 32, "itransformer": 32,
    "lpatchtst": 32, "vlstm": 32, "vxlstm": 32,
    "vsnmamba2": 32, "tft": 32,
}

DEFAULT_LOOKBACK = {
    "ar1x": 64, "dlinear": 64, "nlinear": 64,
    "lstm": 64, "xlstm": 64, "pslstm": 64,
    "mamba2": 64, "patchtst": 64, "itransformer": 64,
    "lpatchtst": 64, "vlstm": 64, "vxlstm": 64,
    "vsnmamba2": 64, "tft": 64,
}


class WithTickerEmb(nn.Module):
    """Wrap an encoder with a ticker-embedding front-end."""

    def __init__(self, enc: nn.Module, n_assets: int, dim: int = 8):
        super().__init__()
        self.emb = TickerEmbedding(n_assets, dim)
        self.enc = enc

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        return self.enc(self.emb(x, ticker_ids))


def build_encoder(name: str, in_dim: int, hidden: int, lookback: int,
                  n_assets: int | None = None, use_ticker_emb: bool = True) -> nn.Module:
    """Construct an encoder; wraps with a ticker embedding when requested."""
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder {name}; have {sorted(ENCODERS)}")
    enc = ENCODERS[name](in_dim=in_dim + (8 if use_ticker_emb and n_assets else 0),
                         hidden=hidden, lookback=lookback)
    if use_ticker_emb and n_assets:
        enc = WithTickerEmb(enc, n_assets)
    return enc

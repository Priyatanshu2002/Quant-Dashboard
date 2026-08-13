# Deep Learning & Transformers in Trading — Research Synthesis

**Project Agonistes · research deliverable · 2026-08-12**
Sources: arXiv, SSRN, MDPI, Springer, JFDS, Google-Scholar-indexed venues. Every claim carries a
ledger citation; the Sources block is machine-rendered.

---

## 1. The headline question: which models have shown the most profitability?

The most rigorous, directly comparable evidence is the **Oxford benchmark**
"Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted
Performance" (Saly-Kaufmann, Wood, Calliess & Zohren; arXiv:2603.01820) — this is the paper your
screenshot table comes from (their architecture-complexity table, listing AR1x, DLinear, NLinear,
Ridge/Lasso ML, XGBoost, HMM+LightGBM, Strategy XGBoost, LSTM, xLSTM, PsLSTM, Mamba2, VSN+Mamba2
and their parameter counts) [1].

Their protocol: 65 cross-asset futures (bonds, commodities, energy, FX, equity indices),
2010–2025 daily data, momentum features → sequence encoder → tanh position head, trained
end-to-end by minimizing **negative annualized Sharpe of the pooled vol-targeted portfolio**
(σ_tgt = 10%), rolling retrain, top-10-of-50-seed ensembling, gross returns with post-hoc
breakeven transaction-cost analysis [1].

**Out-of-sample Sharpe, 2010–2025 (their Table 1/2):**

| Model family | Model | Sharpe | CAGR | Max DD | HAC t | Turnover |
|---|---|---|---|---|---|---|
| Hybrid (VSN) | **VLSTM (VSN+LSTM)** | **2.40** | 26.3% | −22.9% | 8.81 | 967 |
| Hybrid | LPatchTST (LSTM+PatchTST) | 2.31 | 25.5% | −17.4% | 8.81 | 960 |
| TFT | TFT | 2.20 | 24.0% | −23.2% | 8.13 | 913 |
| Recurrent | xLSTM | 1.79 | 19.4% | −14.1% | 6.85 | 483 |
| Recurrent | PsLSTM | 1.74 | 18.7% | −13.1% | 6.83 | 823 |
| Recurrent | VxLSTM (VSN+xLSTM) | 1.69 | 19.4% | **−11.8%** | 6.89 | 776 |
| Recurrent | LSTM | 1.48 | 13.5% | −34.2% | 4.56 | 948 |
| SSM | VSN+Mamba2 | 1.10 | 9.7% | −16.3% | 3.65 | 329 |
| SSM | Mamba2 | 0.78 | 5.9% | −26.3% | 2.31 | 233 |
| Transformer | PatchTST | 0.76 | 8.5% | −17.6% | 3.29 | 624 |
| Linear | AR1x / DLinear / NLinear | 0.64–0.83 | 7.5–8.3% | ~−17% | 2.9–3.1 | ~280–354 |
| Passive | buy-and-hold equal weight | 0.48 | 4.4% | −30.8% | 1.65 | — |
| Transformer | iTransformer | 0.35 | 3.1% | −26.4% | 1.26 | 36 |

**Robust findings:**
1. **Structured temporal representation beats raw capacity.** Models with adaptive gating +
   explicit recurrent state (VLSTM, LPatchTST, TFT, xLSTM family) dominate both linear models and
   generic attention/SSM baselines across every horizon; rankings persist under a reduced
   25-seed/top-5 budget (Table 4) [1].
2. **Linear dynamics are insufficient** — AR1x/DLinear/NLinear occasionally shine in high-volatility
   years (e.g., 2020) but their long-horizon Sharpe stays ~0.6–0.8, near the passive benchmark [1].
3. **Turnover and cost-robustness differentiate.** xLSTM posts the largest breakeven cost buffer
   (e.g., 33.9 bps on Lumber) with roughly half LSTM's turnover (483 vs 948) — higher
   signal-to-trade efficiency; iTransformer's ultra-low turnover (36) coincides with weak returns [1].
4. **Downside matters separately from mean.** VxLSTM has the smallest Max DD (−11.8%) and best
   Calmar (1.64); LPatchTST has the best worst-year profile (min annual Sharpe +0.51) [1].

## 2. The Sharpe-optimized trading lineage (Oxford-Man)

The benchmark's loss function and hybrids descend from a documented line of **direct-Sharpe
optimization** trading models:

- **Deep Momentum Networks** (Lim, Zohren, Roberts 2019, JFDS): LSTM-sized positions trained on
  negative Sharpe; classical TSMOM baselines beaten from ~2003 onward [26].
- **Trading with the Momentum Transformer** (Wood, Giegerich, Roberts, Zohren; arXiv:2112.08534;
  risk.net/JFDS 2023): decoder-only TFT (VSN + LSTM + attention) trained on Sharpe —
  **Sharpe 2.62 (1995–2020)** on 25 commodities + 11 indices + 5 FI + 9 FX; **2.47 through the
  Covid crash vs −1.50 for the LSTM DMN**; net-of-cost: portfolio Sharpe 2.00 at 0 bps, 1.22 at
  1 bps, vs LSTM 0.82/0.20 — attention architectures lose 106% of Sharpe to costs vs LSTM's 228% [2][3].
  Reference code is public [27].
- **Equity replication** (Mason et al., arXiv:2412.12516): on US equities the Momentum Transformer
  still beats basic momentum (+5.21%/yr, +1.3 Sharpe) but lands at Sharpe 1.12 — momentum
  transformers suit futures/indices better than single equities [3].
- **Slow Momentum with Fast Reversion** (Wood, Roberts, Zohren 2022, JFDS 4(1)): changepoint-aware
  deep momentum [2].
- **Spatio-Temporal Momentum** (JFDS 5(3)): joint time-series + cross-sectional momentum learning [26].
- **DeePM** (arXiv:2601.05975): regime-robust variant; Transformer baseline Sharpe 1.02–1.10 with
  ~5–6% annual returns and −26 to −32% DD on macro portfolios [15].

## 3. State-space and modern recurrent variants

- **Mamba/Mamba2 in finance**: in the Oxford benchmark, Mamba2 alone reaches only 0.78 Sharpe;
  the **VSN+Mamba2** front-end (feature selection before the SSM) lifts it to 1.10 — explicit
  feature conditioning partially fixes SSM instability but doesn't close the gap to recurrent
  hybrids [1]. This matches the "asymptotic efficiency ≠ empirical superiority" conclusion [1].
- **xLSTM family** (Beck et al. 2024 architecture): sLSTM/mLSTM exponential gating gives the best
  cost-robustness profile in the benchmark (Sharpe 1.79–1.99 across 2020–2025) [1]; PsLSTM
  (patch-sLSTM) reaches 1.74 with strong downside metrics [1].
- **Quantformer** (arXiv:2404.00424): transformer as a cross-sectional factor; monthly strategy
  Sharpe 0.86–0.94 with 15–18% annual returns; as a factor it beat 100 classical factors
  (0.915 vs best-other 0.243) — but their daily/weekly variants went negative (−0.36 avg) [4].
- **Sharpe-optimized loss design** (Schäfer 2025): a Sharpe-ratio-optimized DL framework with
  risk-sensitive forecasting beats standard loss training on stock performance prediction [25].
- **WaveLSFormer** (arXiv:2601.13435): wavelet front-end + transformer for long-short hourly US
  equity; ROI 0.225→0.607 and **Sharpe 1.024→2.157**, beating wavelet-LSTM (1.879) [5].
- **AttentionLSTM + Sharpe-type losses** (arXiv:2605.28853): AttentionLSTM with
  Omega-CVaR-RiskParity loss → annualized Sharpe 0.29 vs S&P −0.02, +7.86% compounded over
  2007–2023; PatchTST and VSN-LSTM variants went negative — loss design decides [6].

## 4. Temporal Fusion Transformer (Agonistes' chosen model)

- **Original TFT** (Lim, Arik, Loeff, Pfister; arXiv:1912.09363): gated recurrent + interpretable
  attention with variable-selection networks — the architectural base [12].
- **In trading**: Sharpe 2.20–2.27 in the Oxford benchmark (3rd best overall) [1]; Sharpe 2.62 in
  the Momentum Transformer lineage [2][3]; **TFT-ASRO** (Sensors 2025) adds an adaptive-Sharpe
  multi-task objective → 18% Sharpe improvement over prior DL models [17]; a TFT signal strategy
  on crypto (Systems 2025) made 38.6% cumulative with Sharpe 1.06 over 3y [18]; TFT beats N-BEATS
  and ARIMA-family baselines in several single-market studies [16][22].
- **Caveat**: TFT's advantage is strongest with static covariates (asset class, sector) and
  known-future inputs — the Oxford benchmark shows plain attention without recurrent structure
  (iTransformer 0.35, PatchTST 0.76) underperforms badly, and TFT's margin over VLSTM is small [1].

## 5. Time-series foundation models (TSFMs) — the cautionary evidence

- **Zero-shot TSFMs fail in finance**: Chronos/TimesFM zero-shot portfolios score negative or
  near-zero Sharpe (worst variants < −4) vs CatBoost's 6.46 at 512-day windows; fine-tuning
  generally degrades them further [7].
- **Financial pre-training fixes it**: Chronos pre-trained on financial returns reaches
  **Sharpe 5.42, 36.8% annualized** (512d window) [7]; fine-tuned TimesFM (PFN) hits Sharpe 1.68,
  +3.6%/yr market-neutral on S&P 500 vs original TimesFM 0.42 and AR(1) 1.58 [8][24].
- **Takeaway**: the pretraining distribution matters more than the architecture — a generic
  foundation model is not a trading edge out of the box [7][8].

## 6. LLM trading agents — strong backtests, but with a skeptic's caveat

- **FinAgent** (KDD 2024, arXiv:2402.18485): multimodal, tool-augmented LLM agent; beats 12
  baselines with **+36% average profit improvement**; 92.27% return on one dataset; only method
  consistently beating the market [9].
- **Survey** (arXiv:2408.06361 / ACM ICAIF): LLM agents deliver **15–30% annualized return over
  the strongest baseline** in backtests [10]; related systems: TradingGPT (layered memory) [13],
  TradingAgents (multi-agent firm simulation) [14], P1GPT [14-adjacent], Agentic Trading survey [10].
- **StockBench** (arXiv:2510.02209) is the counterweight: in a contamination-free, multi-month,
  real-market benchmark, **most LLMs fail to beat buy-and-hold**; only some show promise — QA
  strength does not transfer to trading [11]. Treat LLM-agent backtests as the least
  externally-validated evidence class.

## 7. Stat-arb / cross-sectional deep learning (SSRN pillar)

- **Deep Learning Statistical Arbitrage** (Guijarro-Ordonez, Pelger, Zanotti; SSRN 3862004,
  BlackRock/Stanford): conditional latent-factor residuals → convolutional transformer signals →
  constrained policy; "consistently high out-of-sample mean returns and Sharpe ratios,
  substantially outperforming all benchmarks" on daily US equities [19].
- **Transformers vs LSTMs for electronic markets** (SSRN 4577922) directly compares the two
  families' trading profitability [20]; **Machine-Learning Signals and Trading Frictions**
  (SSRN 7115197) tests whether ML return predictability survives tradable long-short performance
  after frictions [21]; **DL Statistical Arbitrage benchmarks** show LSTM/Transformer > classical
  pairs in accuracy and stability [22][23].

## 8. What this means for Project Agonistes

1. **Our TFT choice is defensible** (2.20–2.27 Sharpe in the benchmark, interpretable VSN —
   aligns with our LangGraph debate's need for explainable signals) [1][12].
2. **The best-in-class upgrade path is VLSTM or LPatchTST** (+0.1–0.2 Sharpe over TFT, better
   drawdowns) — cheap to add to our `strategy_builder` model zoo [1].
3. **The objective matters more than the model**: train on pooled Sharpe, not MSE; add
   volatility targeting (σ_tgt = 10%), seed ensembling, and breakeven-cost analysis — all now
   implemented in `strategy_builder/` per the protocol [1].
4. **Don't chase SSMs or TSFMs on their own** — Mamba2/Chronos/TimesFM need feature-selection
   front-ends or financial pre-training to become competitive [1][7][8].
5. **Honest validation discipline**: the Oxford benchmark's HAC t-stats, worst-3m/min-annual
   Sharpe, CVaR 5% and per-asset breakeven costs are the right gates before any paper-trading —
   matching Agonistes' existing "no edge found is reported plainly" rule [1].

---

## Sources

[1] https://arxiv.org/abs/2603.01820
[2] https://arxiv.org/abs/2112.08534
[3] https://arxiv.org/abs/2412.12516
[4] https://arxiv.org/abs/2404.00424
[5] https://arxiv.org/abs/2601.13435
[6] https://arxiv.org/abs/2605.28853
[7] https://arxiv.org/abs/2511.18578
[8] https://arxiv.org/abs/2412.09880
[9] https://arxiv.org/abs/2402.18485
[10] https://arxiv.org/abs/2408.06361
[11] https://arxiv.org/abs/2510.02209
[12] https://arxiv.org/abs/1912.09363
[13] https://arxiv.org/abs/2309.03736
[14] https://arxiv.org/abs/2412.20138
[15] https://arxiv.org/abs/2601.05975
[16] https://arxiv.org/abs/2509.10542
[17] https://www.mdpi.com/1424-8220/25/3/976
[18] https://www.mdpi.com/2079-8954/13/6/474
[19] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3862004
[20] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4577922
[21] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7115197
[22] https://link.springer.com/article/10.1186/s40854-026-00929-6
[23] https://ideas.repec.org/p/arx/papers/2411.05790.html
[24] https://tech.preferred.jp/en/blog/timesfm
[25] https://ojs.apspublisher.com/index.php/apemr/article/view/210
[26] https://jfds.pm-research.com/content/5/3/107
[27] https://github.com/kieranjwood/trading-momentum-transformer

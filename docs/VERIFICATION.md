# Project Agonistes — Verification Report

**Date:** 2026-08-12
**Verified by:** automated verification run (subagent, Hermes)
**Project root:** `C:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice`

## 1. Environment

| Item | Value |
|---|---|
| OS / shell | Windows 10 (git-bash / MSYS, POSIX syntax) |
| Python | 3.11.9 (`.venv/Scripts/python`) |
| API backend | `python main.py serve` → uvicorn on `127.0.0.1:8000` |
| Database | SQLite `data/agonistes_dev.db` (live, not a mock) |
| Data present | AAPL, MSFT, NVDA, SPY, TLT, ^NSEI, BTC-USD, ETH-USD (2y daily); ~4.5k feature vectors; macro snapshots; yfinance fundamentals |
| Network | Live (SEC EDGAR companyfacts API reachable) |

## 2. API server — endpoint verification (all against live DB)

Server started with `.venv/Scripts/python main.py serve` (PID 19948), health check returned HTTP 200, then each endpoint was curled and the server killed afterwards. All six returned **HTTP 200 with valid JSON**.

| # | Endpoint | Status | Response excerpt |
|---|---|---|---|
| 1 | `/api/screener/top` | 200 | `[]` (valid JSON; server log: "Screener: scored 7 assets, selected 0 candidates" — no assets passed score threshold) |
| 2 | `/api/backtest/report?symbol=SPY` | 200 | `{"strategy_name": "ma_cross", "period_start": "2024-08-12", "period_end": "2026-08-11", "regime": "FULL", "total_return_pct": 16.26, "cagr": 7.84, "sharpe_ratio": 0.55, "sortino_ratio": 0.94, "calmar_ratio": 2.21, "max_drawdown_pct": -3.54, "max_drawdown_duration_days": 297, "daily_var_95": 0.0003, "volatility_annualized": 4.79, "total_trades": 13, "win_rate": 0.69, ...}` |
| 3 | `/api/backtest/equity?symbol=SPY` | 200 | `[{"t": "2024-08-12", "equity": 1000000.0}, {"t": "2024-08-12", "equity": 1000000.0}, {"t": "2024-08-13", "equity": 999272.57}, {"t": "2024-08-14", "equity": 999274.24}, ...]` |
| 4 | `/api/portfolio/snapshot` | 200 | `{"time": "2026-08-12 06:51:12.28", "nav_usd": 1000000.0, "cash_usd": 1000000.0, "invested_usd": 0.0, "daily_pnl_usd": 0.0, "position_count": 0, ...}` |
| 5 | `/api/financials?symbol=AAPL` | 200 | `{"time": "2026-08-12 06:55:09.19", "symbol": "AAPL", "asset_class": "EQUITY_US", "period_type": "SNAPSHOT", "revenue": 466822987776.0, "revenue_yoy_growth": 0.164, "gross_profit": 227123003392.0, "ebitda": 167959003136.0, "net_income": 128929996800.0, "eps_actual": 8.63, "eps_yoy_growth": 0.287, "total_debt": 84343996416.0, "debt_to_equity": 78.445, "current_ratio": 1.003, ...}` (real yfinance fundamentals from DB) |
| 6 | `/api/debate/recent` | 200 | `[]` (valid JSON; no debate records in DB yet) |

All responses were parseable JSON; no 4xx/5xx, no HTML error pages, no empty server crashes. The server shut down cleanly on kill.

## 3. SEC EDGAR XBRL parser — live MSFT (CIK 0000789019)

Command: `.venv/Scripts/python -c "from data_ingestion.fundamental_feeds.three_statement_parser import parse_company_facts; s = parse_company_facts('0000789019', 'MSFT'); print({k: s.get(k) for k in ('fiscal_year','revenue','net_income','free_cash_flow','revenue_yoy_growth')})"`

Parser output (live fetch of `https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json`):

```
{'fiscal_year': 2026, 'revenue': 89950000000.0, 'net_income': 133749000000.0, 'free_cash_flow': 66987000000.0, 'revenue_yoy_growth': 0.17788686799846665}
```

Raw 10-K facts behind the snapshot (per-concept latest annual entries, verified in the same live feed):

| Concept | Period (end) | Value | Notes |
|---|---|---|---|
| RevenueFromContractWithCustomerExcludingAssessedTax | 2026-06-30 | **331,839,000,000** | True FY2026 revenue (hundreds of billions ✓) |
| NetIncomeLoss | 2026-06-30 | 133,749,000,000 | FY2026 |
| EarningsPerShareDiluted | 2026-06-30 | 17.95 | FY2026 |
| CommonStockSharesOutstanding | 2026-06-30 | 7,427,000,000 | FY2026 |
| NetCashProvidedByUsedInOperatingActivities | 2026-06-30 | 182,935,000,000 | FY2026 |
| PaymentsToAcquirePropertyPlantAndEquipment | 2026-06-30 | 115,948,000,000 | FY2026 |

Cross-checks that pass:
- `free_cash_flow` = OCF − capex = 182,935 − 115,948 = **66,987M** ✓ (snapshot reports 66,987M)
- `revenue_yoy_growth` = 331,839 / 281,698 (FY2025) − 1 = **17.79%** ✓ (snapshot reports 17.79%)

**Finding (parser bug, not data bug):** the snapshot's `revenue` field is **$89.95B — MSFT's FY2017 revenue**, not FY2026. In `three_statement_parser.py`, `_US_GAAP_FIELDS` lists revenue concepts in order (Excluding → Including → `Revenues` → `SalesRevenueNet`) and each concept *overrides* the previous one; MSFT last tagged the legacy `Revenues` concept in FY2017 (89.95B), so the stale value clobbers the correct FY2026 value. Everything else (net income, FCF, growth) is correct FY2026. **Fix suggestion:** stop at the first concept that yields an entry instead of letting later legacy concepts override, or prefer the newest `end` date across concepts. (Not changed in this run — core files untouched per task constraints.)

## 4. DCF on the parsed MSFT snapshot

Command: parse_company_facts('0000789019','MSFT') → `dcf_from_snapshot(s)` (wacc=0.10, terminal_growth=0.025; no market_cap/current_price in snapshot → shares=None → `intrinsic_value_per_share=None` as expected):

```
dcf_result: DCFResult(intrinsic_value_per_share=None,
  enterprise_value=1585236930199.66,          # ≈ $1.585T
  pv_of_projected_fcf=727020560303.09,        # ≈ $727B
  pv_of_terminal_value=858216369896.57,       # ≈ $858B
  wacc=0.1, terminal_growth=0.025, margin_of_safety=None)
```

DCF inputs were the correct FY2026 values (FCF $66.99B, growth 17.79%), so the valuation itself is unaffected by the revenue-field bug above. Note: with shares_outstanding 7.427B present in the snapshot, `dcf_from_snapshot` still yields `intrinsic_ps=None` because it derives shares only from `market_cap/current_price`.

## 5. How to verify again

```bash
# From project root: C:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice

# 1. API server (background), then curl the 6 endpoints:
.venv/Scripts/python main.py serve          # listens on 127.0.0.1:8000
curl -s http://127.0.0.1:8000/api/screener/top
curl -s 'http://127.0.0.1:8000/api/backtest/report?symbol=SPY'
curl -s 'http://127.0.0.1:8000/api/backtest/equity?symbol=SPY'   # first 2 points
curl -s http://127.0.0.1:8000/api/portfolio/snapshot
curl -s 'http://127.0.0.1:8000/api/financials?symbol=AAPL'
curl -s http://127.0.0.1:8000/api/debate/recent
# Every endpoint must return HTTP 200 + valid JSON. Kill the server afterwards.

# 2. SEC EDGAR parser + DCF (live network):
.venv/Scripts/python -c "from data_ingestion.fundamental_feeds.three_statement_parser import parse_company_facts; s = parse_company_facts('0000789019', 'MSFT'); print({k: s.get(k) for k in ('fiscal_year','revenue','net_income','free_cash_flow','revenue_yoy_growth')})"
.venv/Scripts/python -c "
from data_ingestion.fundamental_feeds.three_statement_parser import parse_company_facts
from data_ingestion.fundamental_feeds.dcf_calculator import dcf_from_snapshot
r = dcf_from_snapshot(parse_company_facts('0000789019', 'MSFT'))
print(r.enterprise_value, r.pv_of_terminal_value)"

# Sanity checks: FCF ≈ OCF − capex; revenue_yoy_growth ≈ FY2026/FY2025 − 1;
# revenue should be ~$331.8B for FY2026 (watch for the FY2017 override bug).
```

// Typed API client — the Python backend (main.py serve) exposes these routes.

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE}${path}`);
  if (!resp.ok) throw new Error(`${path}: ${resp.status}`);
  return resp.json() as Promise<T>;
}

async function post<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, { method: "POST" });
  if (!resp.ok) throw new Error(`${path}: ${resp.status}`);
  return resp.json() as Promise<T>;
}

// ── DTOs ──────────────────────────────────────────────────────────────
export interface ScreenerRow {
  symbol: string;
  asset_class: string;
  technical: number;
  fundamental: number;
  sentiment: number;
  macro: number;
  momentum: number;
  composite: number;
}

export interface MarketRow {
  symbol: string;
  name?: string | null;
  asset_class: string;
  sector?: string | null;
  price?: number | null;
  change_pct?: number | null;
  volume?: number | null;
  market_cap?: number | null;
  composite: number;
  technical: number;
  fundamental: number;
  sentiment: number;
  macro: number;
  momentum: number;
}

export interface UniverseEntry {
  symbol: string;
  name?: string | null;
  asset_class: string;
  sector?: string | null;
}

export interface CompanyProfile {
  company_name?: string;
  sector?: string;
  industry?: string;
  country?: string;
  currency?: string;
  website?: string;
  employees?: number;
}

export interface StatementRow {
  period: string;
  [key: string]: unknown;
}

export interface EarningsResult {
  earnings_date: string;
  eps_actual?: number;
  eps_estimate?: number;
  eps_surprise_pct?: number;
}

export interface LLMAnalysis {
  kind: string;
  verdict: {
    score?: number;
    label?: string;
    rating?: string;
    summary?: string;
    thesis?: string;
    key_points?: string[];
    risks?: string[];
    catalysts?: string[];
    fair_value_est?: number | null;
    analyzed_events?: number;
  };
  model: string;
  created_at: string;
  cached?: boolean;
}

export interface FinancialsDTO {
  symbol: string;
  profile: CompanyProfile | null;
  snapshot: Record<string, unknown> | null;
  dcf: {
    intrinsic_value_per_share: number | null;
    margin_of_safety: number | null;
    wacc: number | null;
    terminal_growth: number | null;
    enterprise_value: number | null;
    pv_of_projected_fcf: number | null;
    pv_of_terminal_value: number | null;
    inputs: {
      ttm_free_cash_flow?: number;
      revenue_growth_rate?: number;
      net_debt?: number;
      market_cap?: number;
      current_price?: number;
    };
    sensitivity: {
      waccs: number[];
      terminal_growths: number[];
      grid: (number | null)[][];
    };
  } | null;
  cfa?: ValuationDTO | null; // CFA-standard 3-statement + DCF model (authoritative)
  statements: { income: StatementRow[]; balance: StatementRow[]; cashflow: StatementRow[] };
  annual_statements?: { income: StatementRow[]; balance: StatementRow[]; cashflow: StatementRow[] };
  ratios: { period: string; income: Record<string, unknown>; balance: Record<string, unknown>; cashflow: Record<string, unknown> }[];
  price_change_pct: number | null;
  earnings: { next_date: string | null; results: EarningsResult[] };
  llm_analyses: { news: LLMAnalysis | null; fundamental: LLMAnalysis | null };
}

export interface SentimentEvent {
  ts: string;
  symbol: string;
  source: string;
  score: number;
  source_weight: number;
  headline: string;
  url: string;
}

export interface SentimentDTO {
  symbol: string;
  hours: number;
  aggregate: {
    score: number;
    volume: number;
    positive_pct: number;
    negative_pct: number;
    momentum: number;
  };
  per_source: Record<string, { score: number; volume: number }>;
  series: { ts: string; score: number; volume: number }[];
  events: SentimentEvent[];
  llm: LLMAnalysis | null;
}

export interface BacktestReportDTO {
  strategy_name: string;
  period_start: string;
  period_end: string;
  regime: string;
  total_return_pct: number;
  cagr: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  information_ratio: number;
  max_drawdown_pct: number;
  max_drawdown_duration_days: number;
  daily_var_95: number;
  volatility_annualized: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  expectancy_per_trade_usd: number;
  cost_drag_pct: number;
  alpha_vs_sp500: number;
  performance_by_regime: Record<string, number>;
}

// ── CFA 3-statement + DCF valuation model (/api/model) ──────────────
export interface DiscountRates {
  risk_free: number;
  erp: number;
  erp_source: string;
  beta: number;
  cost_of_equity: number;
  cost_of_debt: number;
  cost_of_debt_after_tax: number;
  tax_rate: number;
  w_e: number | null;
  w_d: number | null;
  wacc: number;
  country: string | null;
  synthetic_rating: string | null;
  cost_of_debt_source: string;
  coverage: string;
}

export interface DcfRow {
  year: number;
  growth: number;
  revenue: number;
  ebit_margin: number;
  ebit: number;
  tax_rate: number;
  nopat: number;
  reinvestment: number;
  invested_capital: number;
  fcff: number;
  roc: number | null;
  wacc: number;
  discount_factor: number;
  pv: number;
}

export interface EquityBridge {
  operating_assets: number;
  debt: number;
  minority_interest: number;
  preferred_stock: number;
  cash: number;
  non_operating_assets: number;
  options_value: number;
  equity_value: number;
  probability_of_failure: number;
}

export interface DcfModel {
  base_revenue: number;
  base_ebit: number;
  base_margin: number;
  sales_to_capital: number;
  riskfree: number;
  initial_wacc: number;
  stable_wacc: number;
  stable_growth: number;
  stable_roc: number;
  stable_reinvestment: number;
  projection: DcfRow[];
  pv_of_projected_fcff: number;
  terminal_cash_flow: number;
  terminal_value: number;
  pv_of_terminal_value: number;
  value_of_operating_assets: number;
  equity_bridge: EquityBridge;
  shares_outstanding: number | null;
  current_price: number | null;
  intrinsic_value_per_share: number | null;
  margin_of_safety: number | null;
}

export interface Scenario {
  intrinsic_value_per_share: number | null;
  margin_of_safety: number | null;
  wacc: number;
  terminal_growth: number;
  revenue_growth_rate: number;
}

export interface ScenarioDict {
  base: Scenario;
  bull: Scenario;
  bear: Scenario;
}

export interface SensitivityGrid {
  waccs: number[];
  growths: number[];
  grid: (number | null)[][];
}

export interface StatementSeries {
  period: string;
  value: number;
}

export interface StatementModel {
  income: StatementSeries[];
  net_income: StatementSeries[];
  operating_cash_flow: StatementSeries[];
  free_cash_flow: StatementSeries[];
  total_assets: StatementSeries[];
  total_debt: StatementSeries[];
  equity: StatementSeries[];
  ltm: Record<string, unknown>;
  balance_equation_check: boolean | null;
  linkage: { fcff_identity: string };
}

export interface QualityModel {
  piotroski: { score: number | null; components: Record<string, boolean>; label?: string };
  altman_z: { z: number | null; zone: string; inputs?: Record<string, number> };
  beneish_m: { m_score: number | null; manipulator: boolean | null; inputs?: Record<string, number> };
  earnings_quality_flags: { flag: string; severity: string }[];
}

export interface HealthFlag {
  ratio: string;
  value: number;
  healthy: boolean;
  benchmark: string;
}

export interface ValuationDTO {
  symbol: string;
  as_of: string;
  profile: CompanyProfile | null;
  statement_model: StatementModel;
  discount_rates: DiscountRates;
  dcf: DcfModel;
  scenarios: ScenarioDict;
  sensitivity: SensitivityGrid;
  ratios: Record<string, number>;
  ratio_health_flags: HealthFlag[];
  quality: QualityModel;
  relative: { multiples: Record<string, number>; mismatches: { signal: string; severity: string }[] };
  assumptions: Record<string, unknown>;
}

// ── Monitoring (/api/monitoring) ────────────────────────────────────
export interface FeedStatus {
  key: string;
  label: string;
  table: string;
  count: number;
  status: "ok" | "empty" | "missing";
}
export interface ServiceStatus {
  name: string;
  port: number;
  running: boolean;
}
export interface MonitoringDTO {
  backend: string;
  db_path: string | null;
  db_size_mb: number | null;
  feeds: FeedStatus[];
  services: ServiceStatus[];
  llm_spend: number | null;
  note: string | null;
}

// ── On-chain (/api/onchain) ─────────────────────────────────────────
export interface OnchainQuery {
  name: string;
  count: number;
  rows: Record<string, unknown>[];
  stored_at: string | null;
}
export interface OnchainDTO {
  source: string;
  queries: OnchainQuery[];
  note?: string | null;
}

// ── Benchmark (/api/benchmark) ──────────────────────────────────────
export interface BenchmarkRow {
  model: string;
  sharpe: number | null;
  cagr: number | null;
  max_dd: number | null;
  calmar: number | null;
  hit_rate: number | null;
  t_hac: number | null;
  info_ratio: number | null;
}
export interface BenchmarkDTO {
  mode: string;
  generated: string | null;
  description?: string | null;
  note?: string | null;
  summary: BenchmarkRow[];
}

export const api = {
  screener: () => get<ScreenerRow[]>("/screener/top"),
  screenerMarket: () => get<{ count: number; rows: MarketRow[] }>("/screener/market"),
  screenerUniverse: () => get<{ count: number; rows: UniverseEntry[] }>("/screener/universe"),
  backtest: (symbol = "SPY") => get<BacktestReportDTO>(`/backtest/report?symbol=${symbol}`),
  backtestEquity: (symbol = "SPY") => get<{ t: string; equity: number }[]>(`/backtest/equity?symbol=${symbol}`),
  portfolio: () => get<Record<string, unknown>>("/portfolio/snapshot"),
  financials: (symbol = "AAPL") => get<FinancialsDTO>(`/financials?symbol=${symbol}`),
  sentiment: (symbol = "AAPL", hours = 72) =>
    get<SentimentDTO>(`/sentiment?symbol=${symbol}&hours=${hours}`),
  debate: (limit = 10) => get<Record<string, unknown>[]>(`/debate/recent?limit=${limit}`),
  monitoring: () => get<MonitoringDTO>("/monitoring"),
  onchain: () => get<OnchainDTO>("/onchain"),
  benchmark: () => get<BenchmarkDTO>("/benchmark"),
  valuation: (symbol = "AAPL") => get<ValuationDTO>(`/model?symbol=${symbol}`),
  refreshFundamentals: (symbol = "AAPL") => post<Record<string, unknown>>(`/fundamentals/refresh?symbol=${symbol}`),
  refreshSentiment: (symbol = "AAPL") => post<Record<string, unknown>>(`/sentiment/refresh?symbol=${symbol}`),
};

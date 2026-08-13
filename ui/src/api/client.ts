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
export interface WaccModel {
  risk_free: number;
  erp: number;
  beta: number;
  cost_of_equity: number;
  cost_of_debt: number;
  cost_of_debt_after_tax: number;
  effective_tax_rate: number;
  equity_value: number;
  debt_value: number;
  w_e: number;
  w_d: number;
  wacc: number;
}

export interface DcfProjection {
  year: number;
  growth: number;
  fcff: number;
  discount_factor: number;
  pv: number;
}

export interface DcfModel {
  base_fcff: number;
  projection_years: number;
  terminal_growth: number;
  net_debt: number;
  projections: DcfProjection[];
  pv_of_projected_fcf: number;
  pv_of_terminal_value: number;
  terminal_value: number;
  enterprise_value: number;
  equity_value: number;
  shares_outstanding: number;
  intrinsic_value_per_share: number | null;
  current_price: number | null;
  margin_of_safety: number | null;
  upside_downsides_pct: number | null;
}

export interface Scenario {
  label: string;
  growth: number;
  margin_adj: number;
  intrinsic_value_per_share: number | null;
  margin_of_safety: number | null;
}

export interface SensitivityGrid {
  waccs: number[];
  terminal_growths: number[];
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

export interface ValuationDTO {
  symbol: string;
  as_of: string;
  profile: CompanyProfile | null;
  statement_model: StatementModel;
  wacc: WaccModel;
  dcf: DcfModel;
  scenarios: Scenario[];
  sensitivity: SensitivityGrid;
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
  valuation: (symbol = "AAPL") => get<ValuationDTO>(`/model?symbol=${symbol}`),
  refreshFundamentals: (symbol = "AAPL") => post<Record<string, unknown>>(`/fundamentals/refresh?symbol=${symbol}`),
  refreshSentiment: (symbol = "AAPL") => post<Record<string, unknown>>(`/sentiment/refresh?symbol=${symbol}`),
};

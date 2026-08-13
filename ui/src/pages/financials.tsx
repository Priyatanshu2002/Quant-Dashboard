import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, ComposedChart, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type FinancialsDTO } from "../api/client";
import Badge from "../components/Badge";
import DCFGauge from "../components/DCFGauge";
import Panel from "../components/Panel";
import Stat from "../components/Stat";
import { fmtNum, fmtPct, fmtPctSigned, fmtUSD, pctColor, sentiColor } from "../lib/format";

type Tab = "income" | "balance" | "cashflow";

const STATEMENT_LABELS: Record<Tab, string> = {
  income: "Income Statement",
  balance: "Balance Sheet",
  cashflow: "Cash Flow",
};

const INCOME_ROWS: [string, string][] = [
  ["total_revenue", "Total Revenue"],
  ["gross_profit", "Gross Profit"],
  ["ebitda", "EBITDA"],
  ["operating_income", "Operating Income"],
  ["net_income", "Net Income"],
  ["eps_diluted", "Diluted EPS"],
  ["income_tax", "Income Tax"],
  ["shares_outstanding", "Diluted Shares"],
];

const BALANCE_ROWS: [string, string][] = [
  ["total_assets", "Total Assets"],
  ["current_assets", "Current Assets"],
  ["cash_and_equivalents", "Cash & Equivalents"],
  ["total_debt", "Total Debt"],
  ["long_term_debt", "Long-Term Debt"],
  ["total_liabilities", "Total Liabilities"],
  ["shareholders_equity", "Shareholders' Equity"],
  ["retained_earnings", "Retained Earnings"],
  ["goodwill", "Goodwill"],
  ["inventory", "Inventory"],
];

const CASHFLOW_ROWS: [string, string][] = [
  ["operating_cash_flow", "Operating Cash Flow"],
  ["capex", "Capital Expenditure"],
  ["free_cash_flow", "Free Cash Flow"],
  ["depreciation", "Depreciation & Amortization"],
  ["stock_based_comp", "Stock-Based Compensation"],
  ["financing_cash_flow", "Financing Cash Flow"],
  ["investing_cash_flow", "Investing Cash Flow"],
];

const ROW_DEFS: Record<Tab, [string, string][]> = {
  income: INCOME_ROWS, balance: BALANCE_ROWS, cashflow: CASHFLOW_ROWS,
};

const RATING_TONE: Record<string, "green" | "red" | "amber" | "blue" | "gray"> = {
  STRONG_BUY: "green", BUY: "green", HOLD: "amber", SELL: "red", STRONG_SELL: "red",
};

const chartTooltip = {
  contentStyle: {
    background: "#101624", border: "1px solid #223049", borderRadius: 8,
    fontSize: 12, color: "#e6ecf7",
  },
  labelStyle: { color: "#8b99b3", fontSize: 11 },
};

function ChartCard({ title, data, dataKey, color, fmt }: {
  title: string; data: Record<string, unknown>[]; dataKey: string;
  color: string; fmt: (v: unknown) => string;
}) {
  return (
    <Panel title={title}>
      <ResponsiveContainer width="100%" height={190}>
        <ComposedChart data={data} margin={{ top: 4, right: 6, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#1a2440" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="period" tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={{ stroke: "#223049" }} />
          <YAxis tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={false}
            tickFormatter={(v: number) => fmt(v)} width={58} />
          <Tooltip {...chartTooltip} formatter={(v) => [fmt(v), title]} />
          <Bar dataKey={dataKey} fill={color} radius={[3, 3, 0, 0]} opacity={0.55} />
          <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </Panel>
  );
}

export default function FinancialsPage() {
  const { symbol: routeSymbol } = useParams();
  const [symbol, setSymbol] = useState((routeSymbol ?? "AAPL").toUpperCase());
  const [data, setData] = useState<FinancialsDTO | null>(null);
  const [tab, setTab] = useState<Tab>("income");
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (routeSymbol) setSymbol(routeSymbol.toUpperCase());
  }, [routeSymbol]);

  useEffect(() => {
    let alive = true;
    setError(null);
    setData(null);
    api.financials(symbol)
      .then((d) => { if (alive) setData(d); })
      .catch((e) => { if (alive) setError(String(e.message ?? e)); });
    return () => { alive = false; };
  }, [symbol]);

  const onRefresh = async () => {
    setRefreshing(true);
    try {
      await api.refreshFundamentals(symbol);
      setData(await api.financials(symbol));
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setRefreshing(false);
    }
  };

  const rows = data?.statements?.[tab] ?? [];
  const snap = data?.snapshot ?? {};
  const dcf = data?.dcf;
  const fund = data?.llm_analyses?.fundamental?.verdict;
  const newsLLM = data?.llm_analyses?.news?.verdict;

  const chartRows = useMemo(() => {
    const defs = ROW_DEFS[tab];
    const first = defs[0][0];
    const second = defs[1]?.[0];
    return rows.map((r) => ({
      period: r.period.slice(2), // "26-06" short label
      [first]: Number(r[first] ?? 0),
      [second]: second ? Number(r[second] ?? 0) : undefined,
    }));
  }, [rows, tab]);

  const trendRows = useMemo(
    () => (data?.ratios ?? []).map((r) => ({
      period: r.period.slice(2),
      gross_margin: Number(r.income.gross_margin ?? 0),
      ebitda_margin: Number(r.income.ebitda_margin ?? 0),
      net_margin: Number(r.income.net_margin ?? 0),
      revenue_yoy: Number(r.income.revenue_yoy_growth ?? 0),
      fcf: Number(r.cashflow.free_cash_flow ?? 0),
    })),
    [data],
  );

  const earningsChart = useMemo(
    () => (data?.earnings?.results ?? []).slice().reverse().map((r) => ({
      date: r.earnings_date.slice(5),
      actual: Number(r.eps_actual ?? 0),
      estimate: Number(r.eps_estimate ?? 0),
      surprise: Number(r.eps_surprise_pct ?? 0),
    })),
    [data],
  );

  const sens = dcf?.sensitivity;
  const sensRows = sens?.grid.map((row, i) => ({
    wacc: `${(sens.waccs[i] * 100).toFixed(0)}%`,
    cells: row.map((v, j) => ({ v, tg: sens.terminal_growths[j] })),
  })) ?? [];

  const sensColors = useMemo(() => {
    if (!sensRows.length) return {};
    const all = sensRows.flatMap((r) => r.cells).map((c) => c.v).filter((v): v is number => v != null);
    if (!all.length) return {};
    const min = Math.min(...all), max = Math.max(...all);
    const out: Record<number, string> = {};
    for (const c of sensRows.flatMap((r) => r.cells)) {
      if (c.v == null) continue;
      const t = (c.v - min) / (max - min || 1);
      out[c.v] = t > 0.55 ? "hi" : t < 0.25 ? "lo" : "";
    }
    return out;
  }, [sensRows]);

  const ratingTone = RATING_TONE[fund?.rating ?? ""] ?? "gray";

  return (
    <div className="stack">
      {/* header */}
      <div className="row-between wrap">
        <div className="row">
          <h1 style={{ margin: 0, fontSize: 24 }}>{symbol}</h1>
          {data?.profile?.company_name && (
            <span className="muted" style={{ fontSize: 15 }}>{data.profile.company_name}</span>
          )}
          {data?.profile?.sector && <span className="chip">{data.profile.sector}</span>}
          {data?.profile?.industry && <span className="chip">{data.profile.industry}</span>}
          {data?.profile?.country && <span className="chip">{data.profile.country}</span>}
        </div>
        <div className="toolbar">
          <input className="input" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && setSymbol((e.target as HTMLInputElement).value.toUpperCase())}
            style={{ width: 120 }} placeholder="TICKER" />
          <Link to={`/news/${symbol}`} className="btn ghost">News & Sentiment →</Link>
          <button className="btn" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? "Refreshing…" : "⟳ Refresh Data"}
          </button>
        </div>
      </div>

      {/* stat strip */}
      <div className="grid grid-5">
        <Stat label="Price" value={
          <span>
            ${fmtNum(snap.current_price, 2)}{" "}
            {data?.price_change_pct != null && (
              <span style={{ color: pctColor(data.price_change_pct), fontSize: 12 }}>
                ({data.price_change_pct >= 0 ? "+" : ""}{data.price_change_pct.toFixed(2)}%)
              </span>
            )}
          </span>
        } />
        <Stat label="Market Cap" value={fmtUSD(snap.market_cap)} />
        <Stat label="Fwd P/E" value={`${fmtNum(snap.forward_pe, 1)}x`} foot={snap.peg_ratio != null ? `PEG ${fmtNum(snap.peg_ratio, 2)}` : undefined} />
        <Stat label="EV/EBITDA" value={`${fmtNum(snap.ev_to_ebitda, 1)}x`} foot="Enterprise multiple" />
        <Stat label="DCF Intrinsic" value={
          <span style={{ color: dcf?.intrinsic_value_per_share != null ? sentiColor((dcf.margin_of_safety ?? 0) * 4) : undefined }}>
            ${fmtNum(dcf?.intrinsic_value_per_share, 2)}
          </span>
        } foot={dcf?.margin_of_safety != null ? `MoS ${fmtPctSigned(dcf.margin_of_safety)}` : "Run refresh"} />
      </div>

      {error && <div className="error-box">⚠ {error} — is the API running on :8000?</div>}

      {!data && !error && <div className="loading">Loading fundamentals for {symbol}…</div>}

      {data && (
        <>
          <div className="grid grid-2">
            {/* DCF panel */}
            <Panel title="DCF Valuation" hint={`WACC ${dcf?.wacc != null ? (dcf.wacc * 100).toFixed(1) : "—"}% · Terminal g ${dcf?.terminal_growth != null ? (dcf.terminal_growth * 100).toFixed(1) : "—"}%`}>
              {dcf ? (
                <>
                  <DCFGauge marginOfSafety={dcf.margin_of_safety ?? 0}
                    intrinsic={dcf.intrinsic_value_per_share} price={snap.current_price as number} />
                  <div className="grid grid-2" style={{ marginTop: 14, gap: 10 }}>
                    <Stat label="PV Projected FCF" value={fmtUSD(dcf.pv_of_projected_fcf)} sm />
                    <Stat label="PV Terminal Value" value={fmtUSD(dcf.pv_of_terminal_value)} sm />
                    <Stat label="TTM FCF" value={fmtUSD(dcf.inputs.ttm_free_cash_flow)} sm />
                    <Stat label="Rev Growth (input)" value={fmtPct(dcf.inputs.revenue_growth_rate)} sm />
                  </div>
                  <div className="chart-tip">Fading-growth DCF: growth converges to terminal over 10 years (plan §8.2).</div>
                </>
              ) : (
                <div className="empty-note">No free-cash-flow data for {symbol} — run “Refresh Data” to compute the DCF.</div>
              )}
            </Panel>

            {/* Sensitivity matrix */}
            <Panel title="Sensitivity — Intrinsic Value / Share" hint="WACC × terminal growth">
              {sensRows.length ? (
                <table className="table sens-table">
                  <thead>
                    <tr>
                      <th>WACC \ g</th>
                      {sens?.terminal_growths.map((tg) => (
                        <th key={tg} className="col">{fmtPct(tg)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sensRows.map((r) => (
                      <tr key={r.wacc}>
                        <td className="mono">{r.wacc}</td>
                        {r.cells.map((c, j) => (
                          <td key={j} className={`mono ${c.v == null ? "faint" : sensColors[c.v] ?? ""} ${c.tg === dcf?.terminal_growth && r.wacc === `${(dcf.wacc! * 100).toFixed(0)}%` ? "base" : ""}`}>
                            {c.v == null ? "—" : `$${c.v.toFixed(2)}`}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="empty-note">Sensitivity requires a computed DCF.</div>
              )}
              <div className="chart-tip">Highlighted cell = base case. Green cells are >75th percentile intrinsic value; red are <25th.</div>
            </Panel>
          </div>

          {/* 3-statement */}
          <Panel title={`3-Statement History — ${STATEMENT_LABELS[tab]}`}
            hint={`${rows.length} quarters · latest ${rows[rows.length - 1]?.period ?? "—"}`}
            right={
              <div className="tabs" style={{ margin: 0, borderBottom: "none" }}>
                {(Object.keys(STATEMENT_LABELS) as Tab[]).map((t) => (
                  <button key={t} className={`tab ${tab === t ? "active" : ""}`}
                    onClick={() => setTab(t)} style={{ padding: "4px 12px" }}>
                    {STATEMENT_LABELS[t]}
                  </button>
                ))}
              </div>
            }>
            {rows.length ? (
              <>
                <div className="grid grid-2" style={{ marginBottom: 14 }}>
                  <ChartCard title={ROW_DEFS[tab][0][1]} data={chartRows} dataKey={ROW_DEFS[tab][0][0]}
                    color="var(--accent)" fmt={fmtUSD} />
                  {ROW_DEFS[tab][1] && (
                    <ChartCard title={ROW_DEFS[tab][1][1]} data={chartRows} dataKey={ROW_DEFS[tab][1][0]}
                      color={tab === "cashflow" ? "var(--green)" : "var(--amber)"} fmt={fmtUSD} />
                  )}
                </div>
                <div style={{ overflowX: "auto" }}>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Line Item</th>
                        {rows.map((r) => <th key={r.period}>{r.period}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {ROW_DEFS[tab].map(([key, label]) => {
                        const vals = rows.map((r) => Number(r[key]));
                        const last = vals[vals.length - 1];
                        const prev = vals[vals.length - 2];
                        const isEPS = key === "eps_diluted" || key === "shares_outstanding";
                        return (
                          <tr key={key}>
                            <td>{label}</td>
                            {vals.map((v, i) => (
                              <td key={i} className={i === vals.length - 1 ? "mono" : "mono faint"}>
                                {isNaN(v) || v === 0 ? "—" : isEPS ? v.toFixed(2) : fmtUSD(v)}
                              </td>
                            ))}
                            {vals.length > 1 && !isNaN(last) && !isNaN(prev) && prev !== 0 && (
                              <td style={{ color: pctColor(last / prev - 1), fontWeight: 700 }}>
                                {fmtPctSigned(last / prev - 1)}
                              </td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <div className="empty-note">No quarterly statements for {symbol} — run “Refresh Data”.</div>
            )}
          </Panel>

          {/* margins + fcf trend */}
          {trendRows.length > 1 && (
            <div className="grid grid-2">
              <Panel title="Margin Trend" hint="8 quarters">
                <ResponsiveContainer width="100%" height={210}>
                  <LineChart data={trendRows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#1a2440" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="period" tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={{ stroke: "#223049" }} />
                    <YAxis tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={false}
                      tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} width={44} />
                    <Tooltip {...chartTooltip} formatter={(v, n) => [`${(Number(v) * 100).toFixed(1)}%`, n]} />
                    <Line type="monotone" dataKey="gross_margin" name="Gross" stroke="#4f8cff" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="ebitda_margin" name="EBITDA" stroke="#f59e0b" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="net_margin" name="Net" stroke="#22c55e" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Panel>
              <Panel title="Free Cash Flow + Revenue Growth" hint="8 quarters">
                <ResponsiveContainer width="100%" height={210}>
                  <ComposedChart data={trendRows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#1a2440" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="period" tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={{ stroke: "#223049" }} />
                    <YAxis yAxisId="l" tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={false}
                      tickFormatter={(v: number) => fmtUSD(v)} width={56} />
                    <YAxis yAxisId="r" orientation="right" tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={false}
                      tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} width={44} />
                    <Tooltip {...chartTooltip} />
                    <Bar yAxisId="l" dataKey="fcf" name="FCF" fill="#22c55e" radius={[3, 3, 0, 0]} opacity={0.5} />
                    <Line yAxisId="r" type="monotone" dataKey="revenue_yoy" name="Rev YoY" stroke="#4f8cff" strokeWidth={2} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </Panel>
            </div>
          )}

          {/* earnings */}
          <Panel title="Earnings" hint={data.earnings.next_date ? `Next: ${data.earnings.next_date}` : "Calendar unavailable"}>
            {earningsChart.length ? (
              <div className="grid grid-2" style={{ alignItems: "center" }}>
                <ResponsiveContainer width="100%" height={200}>
                  <ComposedChart data={earningsChart} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#1a2440" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={{ stroke: "#223049" }} />
                    <YAxis tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={false} width={40} />
                    <Tooltip {...chartTooltip} />
                    <Bar dataKey="actual" name="Actual EPS" fill="#4f8cff" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="estimate" name="Estimate" fill="#5c6b87" radius={[3, 3, 0, 0]} opacity={0.6} />
                    <Line type="monotone" dataKey="surprise" name="Surprise %" stroke="#22c55e" strokeWidth={2} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
                <div className="stack" style={{ gap: 8 }}>
                  {[...(data.earnings.results ?? [])].slice(0, 4).map((r) => (
                    <div key={r.earnings_date} className="row-between" style={{ padding: "6px 10px", background: "var(--panel-2)", borderRadius: 8 }}>
                      <span className="small mono">{r.earnings_date}</span>
                      <span className="small">EPS <b>{r.eps_actual?.toFixed(2)}</b> vs est {r.eps_estimate?.toFixed(2)}</span>
                      <span className="small" style={{ color: pctColor(r.eps_surprise_pct ?? 0), fontWeight: 700 }}>
                        {fmtPctSigned(r.eps_surprise_pct)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="empty-note">No reported quarters stored — “Refresh Data” pulls the earnings calendar + surprises.</div>
            )}
          </Panel>

          {/* AI analyst */}
          <div className="grid grid-2">
            <Panel title="AI Equity Analyst" hint={`${data.llm_analyses.fundamental?.model ?? "—"} · ${data.llm_analyses.fundamental?.created_at ? new Date(data.llm_analyses.fundamental!.created_at.replace(" ", "T")).toLocaleString() : "not run"}`}>
              {fund ? (
                <>
                  <div className="analyst-head">
                    <Badge tone={ratingTone}>{fund.rating ?? "—"}</Badge>
                    <span className="big" style={{ color: sentiColor((fund.score ?? 0) * 3) }}>
                      {fund.score != null ? (fund.score >= 0 ? "+" : "") + fund.score.toFixed(2) : "—"}
                    </span>
                    {fund.fair_value_est != null && (
                      <span className="chip">Fair value ≈ ${Number(fund.fair_value_est).toFixed(2)}</span>
                    )}
                  </div>
                  <p className="analyst-thesis">{fund.thesis}</p>
                  {fund.catalysts && fund.catalysts.length > 0 && (
                    <>
                      <div className="panel-title" style={{ marginTop: 10 }}>Catalysts</div>
                      <ul className="bullet-list">
                        {fund.catalysts.map((c, i) => <li key={i}>{c}</li>)}
                      </ul>
                    </>
                  )}
                  {fund.risks && fund.risks.length > 0 && (
                    <>
                      <div className="panel-title" style={{ marginTop: 10 }}>Risks</div>
                      <ul className="bullet-list">
                        {fund.risks.map((r, i) => <li key={i} className="risk">{r}</li>)}
                      </ul>
                    </>
                  )}
                </>
              ) : (
                <div className="empty-note">No analyst verdict yet — run “Refresh Data” (uses deepseek via Nous Portal with a rule-based fallback).</div>
              )}
            </Panel>

            <Panel title="News Sentiment (LLM)" hint={`${data.llm_analyses.news?.model ?? "—"} · ${data.llm_analyses.news?.analyzed_events ?? 0} headlines`}>
              {newsLLM ? (
                <>
                  <div className="analyst-head">
                    <Badge tone={newsLLM.label === "bullish" ? "green" : newsLLM.label === "bearish" ? "red" : "amber"}>
                      {newsLLM.label ?? "neutral"}
                    </Badge>
                    <span className="big" style={{ color: sentiColor(newsLLM.score ?? 0) }}>
                      {(newsLLM.score ?? 0) >= 0 ? "+" : ""}{(newsLLM.score ?? 0).toFixed(2)}
                    </span>
                  </div>
                  <p className="analyst-thesis">{newsLLM.summary}</p>
                  {newsLLM.key_points && newsLLM.key_points.length > 0 && (
                    <>
                      <div className="panel-title" style={{ marginTop: 10 }}>Key Points</div>
                      <ul className="bullet-list">
                        {newsLLM.key_points.map((p, i) => <li key={i}>{p}</li>)}
                      </ul>
                    </>
                  )}
                  {newsLLM.risks && newsLLM.risks.length > 0 && (
                    <>
                      <div className="panel-title" style={{ marginTop: 10 }}>Risks</div>
                      <ul className="bullet-list">
                        {newsLLM.risks.map((r, i) => <li key={i} className="risk">{r}</li>)}
                      </ul>
                    </>
                  )}
                  <Link to={`/news/${symbol}`} className="btn ghost" style={{ marginTop: 12 }}>Open full news analysis →</Link>
                </>
              ) : (
                <div className="empty-note">No LLM news verdict yet — open the News page and hit Refresh.</div>
              )}
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}

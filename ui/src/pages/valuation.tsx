import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import {
  Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type ValuationDTO } from "../api/client";
import Panel from "../components/Panel";
import Stat from "../components/Stat";
import TickerSearch from "../components/TickerSearch";
import { fmtNum, fmtPct, fmtPctSigned, fmtUSD, pctColor } from "../lib/format";

const chartTooltip = {
  contentStyle: {
    background: "#101624", border: "1px solid #223049", borderRadius: 8,
    fontSize: 12, color: "#e6ecf7",
  },
  labelStyle: { color: "#8b99b3", fontSize: 11 },
};

const WACC_ROWS: [string, (w: ValuationDTO["discount_rates"]) => string | number][] = [
  ["Risk-free rate", (w) => fmtPct(w.risk_free, 2)],
  ["Equity risk premium", (w) => `${fmtPct(w.erp)} (${w.erp_source})`],
  ["Beta", (w) => fmtNum(w.beta, 2)],
  ["Cost of equity (Ke)", (w) => fmtPct(w.cost_of_equity, 2)],
  ["Cost of debt (Kd)", (w) => fmtPct(w.cost_of_debt, 2)],
  ["Cost of debt (after-tax)", (w) => fmtPct(w.cost_of_debt_after_tax, 2)],
  ["Effective tax rate", (w) => fmtPct(w.tax_rate)],
  ["Weight equity / debt", (w) =>
    w.w_e != null && w.w_d != null
      ? `${(w.w_e * 100).toFixed(0)}% / ${(w.w_d * 100).toFixed(0)}%`
      : "n/a"],
  ["Synthetic rating", (w) => w.synthetic_rating ?? "n/a"],
  ["Kd source", (w) => w.cost_of_debt_source],
  ["WACC", (w) => fmtPct(w.wacc, 2)],
];

const RATIO_LABELS: Record<string, string> = {
  gross_margin: "Gross margin", operating_margin: "Operating margin",
  ebitda_margin: "EBITDA margin", net_margin: "Net margin", roe: "ROE",
  roa: "ROA", roic: "ROIC", nopat: "NOPAT", current_ratio: "Current ratio",
  quick_ratio: "Quick ratio", cash_ratio: "Cash ratio", debt_to_equity: "D/E",
  debt_to_capital: "Debt/Capital", debt_to_assets: "Debt/Assets",
  debt_to_ebitda: "Debt/EBITDA", interest_coverage: "Interest coverage",
  net_debt: "Net debt", receivables_turnover: "AR turnover",
  days_sales_outstanding: "DSO (days)", inventory_turnover: "Inventory turnover",
  days_inventory_hand: "DIO (days)", days_payables_outstanding: "DPO (days)",
  cash_conversion_cycle: "Cash conversion cycle", total_asset_turnover: "Asset turnover",
  cash_to_income: "CFO / Net income", sustainable_growth: "Sustainable growth",
  revenue_yoy_growth: "Revenue YoY",
};

export default function ValuationPage() {
  const { symbol: routeSymbol } = useParams();
  const [symbol, setSymbol] = useState((routeSymbol ?? "AAPL").toUpperCase());
  const [data, setData] = useState<ValuationDTO | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (routeSymbol) setSymbol(routeSymbol.toUpperCase());
  }, [routeSymbol]);

  useEffect(() => {
    let alive = true;
    setError(null);
    setData(null);
    api.valuation(symbol)
      .then((d) => { if (alive) setData(d); })
      .catch((e) => { if (alive) setError(String((e as Error).message ?? e)); });
    return () => { alive = false; };
  }, [symbol]);

  const d = data?.dcf;
  const w = data?.discount_rates;
  const mos = d?.margin_of_safety ?? null;
  const sm = data?.statement_model;

  const projChart = useMemo(
    () => (d?.projection ?? []).map((p) => ({
      year: `Y${p.year}`, fcff: p.fcff, pv: p.pv, revenue: p.revenue,
      nopat: p.nopat, roc: p.roc,
    })),
    [d],
  );

  const sens = data?.sensitivity;
  const sensRows = useMemo(() => {
    if (!sens) return [];
    return sens.grid.map((row, i) => ({
      wacc: `${(sens.waccs[i] * 100).toFixed(0)}%`,
      cells: row.map((v, j) => ({ v, g: sens.growths[j] })),
    }));
  }, [sens]);

  const sensColors = useMemo(() => {
    if (!sensRows.length) return {} as Record<number, string>;
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

  const baseWaccLabel = w ? `${(w.wacc * 100).toFixed(0)}%` : "";
  const baseG = d?.stable_growth ?? null;

  function seriesTable(rows: { period: string; value: number }[] | undefined) {
    return (
      <div style={{ overflowX: "auto" }}>
        <table className="table">
          <thead><tr><th>Period</th><th>Value</th></tr></thead>
          <tbody>
            {(rows ?? []).map((r, i) => (
              <tr key={i}>
                <td className="mono faint">{r.period}</td>
                <td className="mono">{fmtUSD(r.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  const scenLabels: { key: "base" | "bull" | "bear"; label: string }[] = [
    { key: "base", label: "Base" }, { key: "bull", label: "Bull" }, { key: "bear", label: "Bear" },
  ];

  return (
    <div className="stack">
      <div className="row-between wrap">
        <div className="row">
          <h1 style={{ margin: 0, fontSize: 24 }}>Valuation</h1>
          {data?.symbol && <span className="chip">{data.symbol}</span>}
          {data?.profile?.company_name && (
            <span className="muted" style={{ fontSize: 15 }}>{data.profile.company_name}</span>
          )}
          {data?.as_of && <span className="chip">As of {data.as_of}</span>}
        </div>
        <div className="toolbar">
          <TickerSearch to="/valuation" placeholder="Search ticker…" width={220} />
        </div>
      </div>

      {error && <div className="error-box">⚠ {error}</div>}
      {data && (data as { error?: string }).error && (
        <div className="error-box">
          ⚠ {(data as { error?: string }).error}
          <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
            This happens when {symbol} doesn't have enough quarterly statements yet — use the
            Financials page → "⟳ Refresh Data" to pull it on demand.
          </div>
        </div>
      )}
      {!data && !error && <div className="loading">Building the valuation model for {symbol}…</div>}
      {data && !(data as { error?: string }).error && !d && !error && (
        <div className="empty-note">No valuation model available for {symbol}.</div>
      )}

      {data && d && w && (
        <>
          <div className="grid grid-5">
            <Stat label="Price" value={
              <span style={{ color: pctColor(mos ?? 0) }}>
                {d.current_price != null ? fmtUSD(d.current_price, 2) : "—"}
              </span>
            } />
            <Stat label="Intrinsic / share" value={fmtUSD(d.intrinsic_value_per_share, 2)} />
            <Stat label="Margin of safety" value={
              <span style={{ color: pctColor(mos ?? 0), fontWeight: 700 }}>
                {mos != null ? fmtPctSigned(mos) : "—"}
              </span>
            } />
            <Stat label="WACC" value={fmtPct(w.wacc, 2)} />
            <Stat label="Op. assets (EV)" value={fmtUSD(d.value_of_operating_assets)} foot={
              d.pv_of_terminal_value && d.value_of_operating_assets
                ? `Terminal ${(d.pv_of_terminal_value / d.value_of_operating_assets * 100).toFixed(0)}% of value`
                : undefined
            } />
          </div>

          <div className="grid grid-2">
            <Panel title="Discount Rates" hint={`CAPM · ${w.country ?? "mature"} · Kd via ${w.cost_of_debt_source}`}>
              <table className="table">
                <thead><tr><th>Input</th><th>Value</th></tr></thead>
                <tbody>
                  {WACC_ROWS.map(([label, fn]) => (
                    <tr key={label}>
                      <td>{label}</td>
                      <td className="mono">{fn(w)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>

            <div className="stack">
              <Panel title="DCF — Equity Bridge" hint="Operating assets → equity value">
                <div className="grid grid-2">
                  <Stat label="PV projected FCFF" value={fmtUSD(d.pv_of_projected_fcff)} sm />
                  <Stat label="PV terminal value" value={fmtUSD(d.pv_of_terminal_value)} sm />
                  <Stat label="Op. assets" value={fmtUSD(d.value_of_operating_assets)} sm />
                  <Stat label="Debt" value={fmtUSD(d.equity_bridge.debt)} sm />
                  <Stat label="Cash + non-op" value={fmtUSD(d.equity_bridge.cash + d.equity_bridge.non_operating_assets)} sm />
                  <Stat label="Minority + pref" value={fmtUSD(d.equity_bridge.minority_interest + d.equity_bridge.preferred_stock)} sm />
                  <Stat label="Equity value" value={fmtUSD(d.equity_bridge.equity_value)} sm />
                  <Stat label="Shares out" value={fmtNum(d.shares_outstanding, 0)} sm />
                </div>
                <div className="chart-tip">
                  Stable g {fmtPct(d.stable_growth, 2)} · stable WACC {fmtPct(d.stable_wacc, 2)} ·
                  stable reinvest {fmtPct(d.stable_reinvestment, 1)} · sales→capital {fmtNum(d.sales_to_capital, 2)}
                </div>
              </Panel>

              <Panel title="Scenarios" hint="Base / Bull / Bear (growth, margin, WACC)">
                <table className="table">
                  <thead><tr><th>Scenario</th><th>Growth</th><th>WACC</th><th>Intrinsic</th><th>MoS</th></tr></thead>
                  <tbody>
                    {scenLabels.map((s) => {
                      const v = data.scenarios[s.key];
                      return (
                        <tr key={s.key}>
                          <td>{s.label}</td>
                          <td className="mono">{fmtPct(v.revenue_growth_rate)}</td>
                          <td className="mono">{fmtPct(v.wacc, 2)}</td>
                          <td className="mono">{fmtUSD(v.intrinsic_value_per_share, 2)}</td>
                          <td className="mono" style={{ color: pctColor(v.margin_of_safety ?? 0) }}>
                            {v.margin_of_safety != null ? fmtPctSigned(v.margin_of_safety) : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </Panel>
            </div>
          </div>

          <Panel title="FCFF Projection — driver-based (10 yr)" hint="Revenue growth → margin → tax → reinvestment → FCFF → ROC">
            <ResponsiveContainer width="100%" height={230}>
              <ComposedChart data={projChart} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#1a2440" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="year" tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={{ stroke: "#223049" }} />
                <YAxis tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={false}
                  tickFormatter={(v: number) => fmtUSD(v)} width={60} />
                <Tooltip {...chartTooltip}
                  formatter={(v, n) => [fmtUSD(v), n === "fcff" ? "FCFF" : n === "pv" ? "PV" : "NOPAT"]} />
                <Bar dataKey="fcff" name="fcff" fill="#22c55e" radius={[3, 3, 0, 0]} opacity={0.5} />
                <Line type="monotone" dataKey="pv" name="pv" stroke="#4f8cff" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ overflowX: "auto", marginTop: 12 }}>
              <table className="table">
                <thead><tr><th>Y</th><th>Growth</th><th>Revenue</th><th>EBIT</th><th>NOPAT</th><th>Reinvest</th><th>FCFF</th><th>ROC</th><th>WACC</th><th>PV</th></tr></thead>
                <tbody>
                  {(d.projection ?? []).map((p) => (
                    <tr key={p.year}>
                      <td className="mono faint">Y{p.year}</td>
                      <td className="mono">{fmtPct(p.growth)}</td>
                      <td className="mono">{fmtUSD(p.revenue)}</td>
                      <td className="mono">{fmtUSD(p.ebit)}</td>
                      <td className="mono">{fmtUSD(p.nopat)}</td>
                      <td className="mono">{fmtUSD(p.reinvestment)}</td>
                      <td className="mono">{fmtUSD(p.fcff)}</td>
                      <td className="mono">{p.roc != null ? fmtPct(p.roc) : "—"}</td>
                      <td className="mono">{fmtPct(p.wacc)}</td>
                      <td className="mono">{fmtUSD(p.pv)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <div className="grid grid-2">
            <Panel title="Sensitivity — Intrinsic / share" hint="WACC × revenue growth">
              {sensRows.length ? (
                <table className="table sens-table">
                  <thead>
                    <tr>
                      <th>WACC \\ g</th>
                      {sens?.growths.map((g) => <th key={g} className="col">{fmtPct(g)}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {sensRows.map((r) => (
                      <tr key={r.wacc}>
                        <td className="mono">{r.wacc}</td>
                        {r.cells.map((c, j) => (
                          <td key={j} className={`mono ${c.v == null ? "faint" : sensColors[c.v] ?? ""} ${r.wacc === baseWaccLabel && c.g === baseG ? "base" : ""}`}>
                            {c.v == null ? "—" : `$${c.v.toFixed(2)}`}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : <div className="empty-note">No sensitivity grid.</div>}
            </Panel>

            <Panel title="Quality & Health" hint="Piotroski · Altman Z · Beneish M">
              <div className="grid grid-3">
                <Stat label="Piotroski F" value={data.quality.piotroski.score != null ? `${data.quality.piotroski.score}/9` : "—"} sm />
                <Stat label="Altman Z" value={data.quality.altman_z.z != null ? fmtNum(data.quality.altman_z.z, 2) : "—"} sm
                  foot={<span style={{ color: data.quality.altman_z.zone === "distress" ? "#f87171" : "#22c55e" }}>{data.quality.altman_z.zone}</span>} />
                <Stat label="Beneish M" value={data.quality.beneish_m.m_score != null ? fmtNum(data.quality.beneish_m.m_score, 2) : "—"} sm
                  foot={data.quality.beneish_m.manipulator ? <span style={{ color: "#f87171" }}>manipulator?</span> : "ok"} />
              </div>
              {data.quality.earnings_quality_flags.length > 0 && (
                <ul className="flag-list">
                  {data.quality.earnings_quality_flags.map((f, i) => (
                    <li key={i} className={f.severity === "warning" ? "warn" : ""}>⚠ {f.flag}</li>
                  ))}
                </ul>
              )}
              {data.relative.mismatches.map((m, i) => (
                <div key={i} className="error-box" style={{ padding: 6, fontSize: 12 }}>⚠ {m.signal}</div>
              ))}
            </Panel>
          </div>

          <Panel title="Ratios" hint="CFA framework">
            <div className="grid grid-3">
              {Object.entries(RATIO_LABELS).map(([key, label]) => {
                const v = data.ratios[key];
                if (v == null) return null;
                const flag = data.ratio_health_flags.find((f) => f.ratio === key);
                const isPct = ["gross_margin", "operating_margin", "ebitda_margin", "net_margin",
                  "roe", "roa", "roic", "sustainable_growth", "revenue_yoy_growth",
                  "cash_to_income", "debt_to_capital", "debt_to_assets"].includes(key);
                return (
                  <div key={key} className="ratio-tile">
                    <div className="ratio-label">{label}</div>
                    <div className={`mono ratio-value ${flag && !flag.healthy ? "warn-text" : ""}`}>
                      {isPct ? fmtPct(v) : fmtNum(v, 2)}
                    </div>
                    {flag && <div className="ratio-note">{flag.healthy ? "✓ ok" : "⚠ " + flag.benchmark}</div>}
                  </div>
                );
              })}
            </div>
          </Panel>

          <Panel title="Relative Valuation" hint="Multiples">
            <div className="grid grid-3">
              {Object.entries(data.relative.multiples ?? {}).map(([k, v]) => (
                <div key={k} className="ratio-tile">
                  <div className="ratio-label">{k.replace(/_/g, " ")}</div>
                  <div className="mono ratio-value">{fmtNum(v, 2)}</div>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Linked 3-Statement History" hint="Trailing quarters from the store">
            <div className="muted" style={{ marginBottom: 10 }}>
              Balance equation (A = L + E):{" "}
              <b>{sm?.balance_equation_check === true ? "✓ balances" : sm?.balance_equation_check === false ? "✗ out of balance" : "n/a"}</b>
            </div>
            <div className="grid grid-2">
              <Panel title="Revenue">{seriesTable(sm?.income)}</Panel>
              <Panel title="Net income">{seriesTable(sm?.net_income)}</Panel>
              <Panel title="Operating cash flow">{seriesTable(sm?.operating_cash_flow)}</Panel>
              <Panel title="Free cash flow">{seriesTable(sm?.free_cash_flow)}</Panel>
              <Panel title="Total assets">{seriesTable(sm?.total_assets)}</Panel>
              <Panel title="Total debt">{seriesTable(sm?.total_debt)}</Panel>
            </div>
          </Panel>

          <Panel title="Assumptions">
            <div className="muted">
              {Object.entries(data.assumptions ?? {})
                .map(([k, v]) => `${k}: ${typeof v === "number" ? fmtNum(v, 4) : String(v)}`).join(" · ")}
              <br />{sm?.linkage?.fcff_identity}
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

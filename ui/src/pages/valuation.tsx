import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import {
  Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type ValuationDTO } from "../api/client";
import Panel from "../components/Panel";
import Stat from "../components/Stat";
import { fmtNum, fmtPct, fmtPctSigned, fmtUSD, pctColor } from "../lib/format";

const chartTooltip = {
  contentStyle: {
    background: "#101624", border: "1px solid #223049", borderRadius: 8,
    fontSize: 12, color: "#e6ecf7",
  },
  labelStyle: { color: "#8b99b3", fontSize: 11 },
};

const WACC_ROWS: [string, (w: ValuationDTO["wacc"]) => string | number][] = [
  ["Risk-free rate", (w) => fmtPct(w.risk_free, 2)],
  ["Equity risk premium", (w) => fmtPct(w.erp)],
  ["Beta", (w) => fmtNum(w.beta, 2)],
  ["Cost of equity (Ke)", (w) => fmtPct(w.cost_of_equity, 2)],
  ["Cost of debt (after-tax)", (w) => fmtPct(w.cost_of_debt_after_tax, 2)],
  ["Effective tax rate", (w) => fmtPct(w.effective_tax_rate)],
  ["Weight equity / debt", (w) =>
    `${(w.w_e * 100).toFixed(0)}% / ${(w.w_d * 100).toFixed(0)}%`],
  ["WACC", (w) => fmtPct(w.wacc, 2)],
];

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
  const w = data?.wacc;
  const mos = d?.margin_of_safety ?? null;
  const sm = data?.statement_model;

  const projChart = useMemo(
    () => (d?.projections ?? []).map((p) => ({
      year: `Y${p.year}`, fcff: p.fcff, pv: p.pv, growth: p.growth,
    })),
    [d],
  );

  const sens = data?.sensitivity;
  const sensRows = useMemo(() => {
    if (!sens) return [];
    return sens.grid.map((row, i) => ({
      wacc: `${(sens.waccs[i] * 100).toFixed(0)}%`,
      cells: row.map((v, j) => ({ v, tg: sens.terminal_growths[j] })),
    }));
  }, [sens]);

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

  const baseWaccLabel = w ? `${(w.wacc * 100).toFixed(0)}%` : "";
  const baseTg = d?.terminal_growth ?? null;

  function seriesTable(rows: { period: string; value: number }[] | undefined) {
    return (
      <div style={{ overflowX: "auto" }}>
        <table className="table">
          <thead>
            <tr><th>Period</th><th>Value</th></tr>
          </thead>
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

  return (
    <div className="stack">
      <div className="row-between wrap">
        <div className="row">
          <h1 style={{ margin: 0, fontSize: 24 }}>CFA Valuation</h1>
          {data?.symbol && <span className="chip">{data.symbol}</span>}
          {data?.profile?.company_name && (
            <span className="muted" style={{ fontSize: 15 }}>{data.profile.company_name}</span>
          )}
          {data?.as_of && <span className="chip">As of {data.as_of}</span>}
        </div>
        <div className="toolbar">
          <input className="input" value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && setSymbol((e.target as HTMLInputElement).value.toUpperCase())}
            style={{ width: 120 }} placeholder="TICKER" />
          <button className="btn" onClick={() => setSymbol((s) => s)}>Load</button>
        </div>
      </div>

      {error && <div className="error-box">⚠ {error}</div>}
      {data && (data as { error?: string }).error && (
        <div className="error-box">
          ⚠ {(data as { error?: string }).error}
          <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
            This happens when {symbol} doesn't have enough quarterly statements yet —
            the data backfill may still be running. Retry in a few minutes, or use the
            Financials page → "⟳ Refresh Data" to pull it on demand.
          </div>
        </div>
      )}
      {!data && !error && <div className="loading">Building CFA 3-statement + DCF model for {symbol}…</div>}
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
            <Stat label="Enterprise value" value={fmtUSD(d.enterprise_value)} foot={
              d.pv_of_terminal_value && d.enterprise_value
                ? `Terminal ${(d.pv_of_terminal_value / d.enterprise_value * 100).toFixed(0)}% of EV`
                : undefined
            } />
          </div>

          <div className="grid grid-2">
            <Panel title="WACC" hint="CAPM cost of equity · after-tax cost of debt">
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
              <Panel title="DCF Valuation" hint={`Two-stage FCFF · ${d.projection_years}yr · terminal g ${fmtPct(d.terminal_growth, 2)}`}>
                <div className="grid grid-2">
                  <Stat label="PV Projected FCF" value={fmtUSD(d.pv_of_projected_fcf)} sm />
                  <Stat label="PV Terminal Value" value={fmtUSD(d.pv_of_terminal_value)} sm />
                  <Stat label="Enterprise value" value={fmtUSD(d.enterprise_value)} sm />
                  <Stat label="Net debt" value={fmtUSD(d.net_debt)} sm />
                  <Stat label="Equity value" value={fmtUSD(d.equity_value)} sm />
                  <Stat label="Shares out" value={fmtNum(d.shares_outstanding, 0)} sm />
                </div>
                <div className="chart-tip">FCFF ≈ Operating Cash Flow − Capex (after-tax interest not modeled).</div>
              </Panel>

              <Panel title="Scenarios" hint="Revenue growth & EBIT margin">
                <table className="table">
                  <thead><tr><th>Scenario</th><th>Growth</th><th>Intrinsic</th><th>MoS</th></tr></thead>
                  <tbody>
                    {(data.scenarios ?? []).map((s) => (
                      <tr key={s.label}>
                        <td>{s.label}</td>
                        <td className="mono">{fmtPct(s.growth)}</td>
                        <td className="mono">{fmtUSD(s.intrinsic_value_per_share, 2)}</td>
                        <td className="mono" style={{ color: pctColor(s.margin_of_safety ?? 0) }}>
                          {s.margin_of_safety != null ? fmtPctSigned(s.margin_of_safety) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Panel>
            </div>
          </div>

          <Panel title="FCFF Projection — fading growth"
            hint={`Base FCFF ${fmtUSD(d.base_fcff)} · WACC ${fmtPct(w.wacc, 2)} · terminal g ${fmtPct(d.terminal_growth, 2)}`}>
            <ResponsiveContainer width="100%" height={230}>
              <ComposedChart data={projChart} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#1a2440" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="year" tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={{ stroke: "#223049" }} />
                <YAxis tick={{ fill: "#5c6b87", fontSize: 10.5 }} tickLine={false} axisLine={false}
                  tickFormatter={(v: number) => fmtUSD(v)} width={60} />
                <Tooltip {...chartTooltip}
                  formatter={(v, n) => [fmtUSD(v), n === "fcff" ? "FCFF" : "PV"]} />
                <Bar dataKey="fcff" name="fcff" fill="#22c55e" radius={[3, 3, 0, 0]} opacity={0.5} />
                <Line type="monotone" dataKey="pv" name="pv" stroke="#4f8cff" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ overflowX: "auto", marginTop: 12 }}>
              <table className="table">
                <thead><tr><th>Year</th><th>Growth</th><th>FCFF</th><th>Discount</th><th>PV</th></tr></thead>
                <tbody>
                  {(d.projections ?? []).map((p) => (
                    <tr key={p.year}>
                      <td className="mono faint">Y{p.year}</td>
                      <td className="mono">{fmtPct(p.growth)}</td>
                      <td className="mono">{fmtUSD(p.fcff)}</td>
                      <td className="mono">{p.discount_factor.toFixed(3)}</td>
                      <td className="mono">{fmtUSD(p.pv)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

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
                        <td key={j}
                          className={`mono ${c.v == null ? "faint" : sensColors[c.v] ?? ""} ${r.wacc === baseWaccLabel && c.tg === baseTg ? "base" : ""}`}>
                          {c.v == null ? "—" : `$${c.v.toFixed(2)}`}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty-note">No sensitivity grid.</div>
            )}
            <div className="chart-tip">Outlined cell = base case. Green &gt;75th percentile intrinsic; red &lt;25th.</div>
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

import { useEffect, useState } from "react";
import { api, BacktestReportDTO } from "../api/client";
import PerformanceCharts, { EquityPoint } from "../components/PerformanceCharts";
import Panel from "../components/Panel";
import Stat from "../components/Stat";
import { fmtPctSigned, fmtUSD } from "../lib/format";

const METRICS: [keyof BacktestReportDTO, string][] = [
  ["total_return_pct", "Total Return"],
  ["cagr", "CAGR"],
  ["sharpe_ratio", "Sharpe"],
  ["sortino_ratio", "Sortino"],
  ["calmar_ratio", "Calmar"],
  ["information_ratio", "Info Ratio"],
  ["max_drawdown_pct", "Max Drawdown"],
  ["win_rate", "Win Rate"],
  ["profit_factor", "Profit Factor"],
  ["expectancy_per_trade_usd", "Expectancy"],
  ["cost_drag_pct", "Cost Drag"],
  ["alpha_vs_sp500", "Alpha vs SP500"],
];

const PCT_KEYS = new Set(["total_return_pct", "cagr", "max_drawdown_pct", "win_rate", "cost_drag_pct", "alpha_vs_sp500"]);

function fmtMetric(key: keyof BacktestReportDTO, v: number): string {
  const isPct = PCT_KEYS.has(key as string);
  if (key === "expectancy_per_trade_usd") return fmtUSD(v, 2);
  return isPct ? fmtPctSigned(v) : v.toFixed(2);
}

export default function BacktestResultsPage() {
  const [symbol, setSymbol] = useState("SPY");
  const [report, setReport] = useState<BacktestReportDTO | null>(null);
  const [equity, setEquity] = useState<EquityPoint[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null); setReport(null); setEquity([]);
    api.backtest(symbol)
      .then(setReport).catch((e) => setError(String((e as Error).message ?? e)));
    api.backtestEquity(symbol)
      .then(setEquity).catch(() => setEquity([]));
  }, [symbol]);

  return (
    <div className="stack">
      <div className="page-head">
        <div className="title">
          <h2>Backtest Engine</h2>
          <p>Strategy validation with full market-friction cost model · MA-cross on {symbol}</p>
        </div>
        <div className="toolbar">
          <input className="input" value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && setSymbol((e.target as HTMLInputElement).value.toUpperCase())}
            style={{ width: 110 }} placeholder="SYMBOL" />
        </div>
      </div>

      {error && <div className="error-box">⚠ {error}</div>}
      {!report && !error && <div className="loading">Running backtest on {symbol}…</div>}

      {report && (
        <>
          <div className="kpi-grid">
            {METRICS.map(([key, label]) => (
              <Stat key={key} label={label}
                value={<span className={PCT_KEYS.has(key as string) && (report[key] as number) < 0 ? "down" : "up"}>{fmtMetric(key, report[key] as number)}</span>} />
            ))}
          </div>

          <Panel title="Performance" hint={`${report.period_start ?? ""} → ${report.period_end ?? ""} · regime: ${report.regime ?? "—"}`}>
            <PerformanceCharts points={equity} />
          </Panel>

          {Object.keys(report.performance_by_regime || {}).length > 0 && (
            <Panel title="Performance by Regime" hint="Sharpe ratio">
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Regime</th><th>Sharpe</th><th>Distribution</th></tr></thead>
                  <tbody>
                    {Object.entries(report.performance_by_regime).map(([regime, sharpe]) => {
                      const n = Number(sharpe);
                      return (
                        <tr key={regime}>
                          <td className="muted">{regime}</td>
                          <td className="mono">{n.toFixed(2)}</td>
                          <td>
                            <div className="btrack" style={{ height: 10, width: 220 }}>
                              <div className="bfill" style={{
                                width: `${Math.min(100, Math.abs(n) * 40)}%`,
                                background: n >= 0 ? "var(--green)" : "var(--red)",
                              }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
        </>
      )}
    </div>
  );
}

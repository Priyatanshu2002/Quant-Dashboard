import { useEffect, useState } from "react";
import { api, BacktestReportDTO } from "../api/client";
import PerformanceCharts from "../components/PerformanceCharts";

const METRICS: [keyof BacktestReportDTO, string][] = [
  ["total_return_pct", "Total Return %"],
  ["cagr", "CAGR %"],
  ["sharpe_ratio", "Sharpe"],
  ["sortino_ratio", "Sortino"],
  ["calmar_ratio", "Calmar"],
  ["information_ratio", "Info Ratio"],
  ["max_drawdown_pct", "Max DD %"],
  ["win_rate", "Win Rate"],
  ["profit_factor", "Profit Factor"],
  ["expectancy_per_trade_usd", "Expectancy $"],
  ["cost_drag_pct", "Cost Drag %"],
  ["alpha_vs_sp500", "Alpha vs SP500 %"],
];

export default function BacktestResultsPage() {
  const [symbol, setSymbol] = useState("SPY");
  const [report, setReport] = useState<BacktestReportDTO | null>(null);

  useEffect(() => {
    api.backtest(symbol).then(setReport).catch(() => setReport(null));
  }, [symbol]);

  return (
    <section>
      <h2>Backtest Results</h2>
      <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
      {report && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, margin: "12px 0" }}>
            {METRICS.map(([key, label]) => (
              <div key={key} style={{ border: "1px solid #ddd", borderRadius: 8, padding: 8 }}>
                <div style={{ fontSize: 11, color: "#777" }}>{label}</div>
                <div style={{ fontSize: 18, fontWeight: 700 }}>
                  {typeof report[key] === "number"
                    ? (report[key] as number).toFixed(2)
                    : JSON.stringify(report[key])}
                </div>
              </div>
            ))}
          </div>
          <PerformanceCharts />
          {Object.keys(report.performance_by_regime || {}).length > 0 && (
            <>
              <h3>Performance by Regime (Sharpe)</h3>
              <table style={{ borderCollapse: "collapse", width: "100%" }}>
                <tbody>
                  {Object.entries(report.performance_by_regime).map(([regime, sharpe]) => (
                    <tr key={regime} style={{ borderBottom: "1px solid #eee" }}>
                      <td style={{ padding: "4px 8px" }}>{regime}</td>
                      <td style={{ padding: "4px 8px", width: 420 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <div
                            style={{
                              width: `${Math.min(100, Math.abs(Number(sharpe)) * 40)}%`,
                              maxWidth: 300,
                              height: 12,
                              borderRadius: 4,
                              background: Number(sharpe) >= 0 ? "#2e7d32" : "#c62828",
                            }}
                          />
                          <b>{Number(sharpe).toFixed(2)}</b>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
    </section>
  );
}

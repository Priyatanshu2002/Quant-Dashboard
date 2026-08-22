import { useEffect, useState } from "react";
import { api, BenchmarkDTO, BenchmarkRow } from "../api/client";
import Panel from "../components/Panel";
import { fmtSigned, fmtPct } from "../lib/format";

function cell(v: number | null, digits = 2, pct = false): string {
  if (v === null || v === undefined || !isFinite(v)) return "~0";
  return pct ? fmtPct(v, digits) : fmtSigned(v, digits);
}

function tone(v: number | null): string {
  if (v === null || v === undefined) return "";
  return v > 0 ? "pos" : v < 0 ? "neg" : "";
}

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkDTO | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.benchmark()
      .then(setData)
      .catch((e) => setError(String((e as Error).message ?? e)));
  }, []);

  return (
    <div className="stack">
      <div className="page-head">
        <div className="title">
          <h2>Model Benchmark Leaderboard</h2>
          <p>OOS walk-forward results (36m train / 6m test), ranked by Sharpe · strategy_builder 37-model harness</p>
        </div>
      </div>

      {error && <div className="error-box">⚠ {error}</div>}
      {!data && !error && <div className="loading">Loading benchmark…</div>}

      {data && (
        <>
          {data.note && <div className="error-box">⚠ {data.note}</div>}
          {data.description && <div className="hint" style={{ marginBottom: 8 }}>{data.description}</div>}

          <Panel title="Full Model Benchmark" hint={data.generated ? `generated ${data.generated.slice(0, 10)} · ${data.summary.length} models` : `${data.summary.length} models`}>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Sharpe</th>
                    <th>CAGR</th>
                    <th>Max DD</th>
                    <th>Calmar</th>
                    <th>Hit %</th>
                    <th>t-HAC</th>
                    <th>Info Ratio</th>
                  </tr>
                </thead>
                <tbody>
                  {[...data.summary]
                    .sort((a, b) => (b.sharpe ?? -Infinity) - (a.sharpe ?? -Infinity))
                    .map((m: BenchmarkRow, i: number) => (
                      <tr key={m.model} className={i === 0 ? "row-top" : ""}>
                        <td className="model">{m.model}</td>
                        <td className={tone(m.sharpe)}>{cell(m.sharpe, 3)}</td>
                        <td className={tone(m.cagr)}>{cell(m.cagr, 1, true)}</td>
                        <td className={tone(m.max_dd)}>{cell(m.max_dd, 1, true)}</td>
                        <td>{cell(m.calmar, 2)}</td>
                        <td>{cell(m.hit_rate, 1, true)}</td>
                        <td className={tone(m.t_hac)}>{cell(m.t_hac, 2)}</td>
                        <td className={tone(m.info_ratio)}>{cell(m.info_ratio, 2)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

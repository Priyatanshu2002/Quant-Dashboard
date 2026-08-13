import { useEffect, useState } from "react";
import { api, ScreenerRow } from "../api/client";
import SignalBreakdown from "../components/SignalBreakdown";

function compositeTone(c: number): string {
  if (c >= 60) return "green";
  if (c >= 45) return "amber";
  return "red";
}

export default function ScreenerPage() {
  const [rows, setRows] = useState<ScreenerRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.screener().then(setRows).catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="stack">
      <div className="page-head">
        <div className="title">
          <h2>Top Candidates</h2>
          <p>Live composite scores across the 33-instrument universe · {rows.length} selected</p>
        </div>
        {rows.length > 0 && (
          <span className="chip">Updated live · {rows.length} passes</span>
        )}
      </div>

      {error && <div className="error-box">⚠ {error} — is the API running on :8000?</div>}
      {rows.length === 0 && !error && (
        <div className="loading">Running screener…</div>
      )}

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr><th>Symbol</th><th>Class</th><th>Composite</th><th>Signal breakdown</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.symbol}>
                <td><span className="symbol-link">{r.symbol}</span></td>
                <td className="muted">{r.asset_class}</td>
                <td>
                  <span className={`badge ${compositeTone(r.composite)}`}>{r.composite.toFixed(1)}</span>
                </td>
                <td><SignalBreakdown row={r} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

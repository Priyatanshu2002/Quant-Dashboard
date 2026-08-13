import { useEffect, useState } from "react";
import { api, ScreenerRow } from "../api/client";
import SignalBreakdown from "../components/SignalBreakdown";

export default function ScreenerPage() {
  const [rows, setRows] = useState<ScreenerRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.screener().then(setRows).catch((e) => setError(String(e)));
  }, []);

  return (
    <section>
      <h2>Live Screener — Top Candidates</h2>
      {error && <p style={{ color: "red" }}>{error} (start `python main.py serve`)</p>}
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr style={{ textAlign: "left", borderBottom: "2px solid #333" }}>
            <th>Symbol</th><th>Class</th><th>Composite</th><th>Breakdown</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.symbol} style={{ borderBottom: "1px solid #eee" }}>
              <td><b>{r.symbol}</b></td>
              <td>{r.asset_class}</td>
              <td><b>{r.composite.toFixed(1)}</b></td>
              <td style={{ width: 420 }}>
                <SignalBreakdown row={r} />
              </td>
            </tr>
          ))}
          {rows.length === 0 && !error && (
            <tr><td colSpan={4} style={{ color: "#888" }}>No candidates yet — run `python main.py smoke`</td></tr>
          )}
        </tbody>
      </table>
    </section>
  );
}

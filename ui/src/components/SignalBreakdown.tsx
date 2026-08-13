import { ScreenerRow } from "../api/client";

const LABELS: [keyof Omit<ScreenerRow, "symbol" | "asset_class" | "composite">, string][] = [
  ["technical", "Technical"],
  ["fundamental", "Fundamental"],
  ["sentiment", "Sentiment"],
  ["macro", "Macro"],
  ["momentum", "Momentum"],
];

/** Horizontal bar chart: how each signal category contributed to the composite. */
export default function SignalBreakdown({ row }: { row: ScreenerRow }) {
  return (
    <div>
      {LABELS.map(([key, label]) => (
        <div key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
          <span style={{ width: 86, fontSize: 11, color: "#555" }}>{label}</span>
          <div style={{ flex: 1, background: "#eee", borderRadius: 3, height: 10 }}>
            <div style={{ width: `${row[key]}%`, background: "#1a5cff", height: 10, borderRadius: 3 }} />
          </div>
          <span style={{ width: 34, fontSize: 11 }}>{row[key].toFixed(0)}</span>
        </div>
      ))}
    </div>
  );
}

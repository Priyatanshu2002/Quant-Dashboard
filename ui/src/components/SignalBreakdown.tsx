import { ScreenerRow } from "../api/client";

const LABELS: [keyof Omit<ScreenerRow, "symbol" | "asset_class" | "composite">, string][] = [
  ["technical", "Technical"],
  ["fundamental", "Fundamental"],
  ["sentiment", "Sentiment"],
  ["macro", "Macro"],
  ["momentum", "Momentum"],
];

const COLORS = ["var(--accent)", "var(--purple)", "var(--cyan)", "var(--amber)", "var(--green)"];

/** Horizontal bar chart: how each signal category contributed to the composite. */
export default function SignalBreakdown({ row }: { row: ScreenerRow }) {
  return (
    <div style={{ minWidth: 240 }}>
      {LABELS.map(([key, label], i) => (
        <div key={key} className="bar-row" style={{ gridTemplateColumns: "96px 1fr 34px", margin: "7px 0" }}>
          <span className="bname">{label}</span>
          <div className="btrack">
            <div className="bfill" style={{ width: `${Math.min(100, row[key])}%`, background: COLORS[i % COLORS.length] }} />
          </div>
          <span className="bval">{row[key].toFixed(0)}</span>
        </div>
      ))}
    </div>
  );
}

/** Side-by-side Bull vs Bear transcript for one gating_log row. */
export default function DebateTranscript({ row }: { row: Record<string, unknown> }) {
  const bull = (row.bull_summary as string) || "—";
  const bear = (row.bear_summary as string) || "—";
  const decision = row.decision === "TRADE" ? "✅ TRADE" : "⛔ NO_TRADE";

  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12, marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <b>{row.symbol as string}</b>
        <span>
          Bull {Number(row.bull_confidence).toFixed(2)} vs Bear {Number(row.bear_confidence).toFixed(2)}
          {" · "}{decision}
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 8 }}>
        <div style={{ background: "#eafbea", borderRadius: 8, padding: 10 }}>
          <b>🐂 BULL</b>
          <p style={{ fontSize: 13 }}>{bull}</p>
        </div>
        <div style={{ background: "#fdeaea", borderRadius: 8, padding: 10 }}>
          <b>🐻 BEAR</b>
          <p style={{ fontSize: 13 }}>{bear}</p>
        </div>
      </div>
    </div>
  );
}

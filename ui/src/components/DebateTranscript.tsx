/** Side-by-side Bull vs Bear transcript for one gating_log row. */
export default function DebateTranscript({ row }: { row: Record<string, unknown> }) {
  const bull = (row.bull_summary as string) || "—";
  const bear = (row.bear_summary as string) || "—";
  const trade = row.decision === "TRADE";
  const bullConf = Number(row.bull_confidence ?? 0);
  const bearConf = Number(row.bear_confidence ?? 0);

  return (
    <div className="panel">
      <div className="row-between wrap">
        <div className="row">
          <span className="symbol-link">{row.symbol as string}</span>
          <span className={`badge ${trade ? "green" : "red"}`}>{trade ? "TRADE" : "NO_TRADE"}</span>
        </div>
        <div className="row muted small">
          <span>Bull <b className="mono">{bullConf.toFixed(2)}</b></span>
          <span>·</span>
          <span>Bear <b className="mono">{bearConf.toFixed(2)}</b></span>
          {row.time ? <span>· <span className="faint">{String(row.time).slice(0, 16).replace("T", " ")}</span></span> : null}
        </div>
      </div>
      <div className="grid grid-2" style={{ marginTop: 12 }}>
        <div style={{ background: "var(--green-soft)", border: "1px solid rgba(52,211,153,0.25)", borderRadius: 10, padding: 12 }}>
          <b style={{ color: "var(--green)", fontSize: 12, letterSpacing: 1 }}>🐂 BULL</b>
          <p style={{ fontSize: 13, margin: "6px 0 0", color: "var(--text)" }}>{bull}</p>
        </div>
        <div style={{ background: "var(--red-soft)", border: "1px solid rgba(248,113,113,0.25)", borderRadius: 10, padding: 12 }}>
          <b style={{ color: "var(--red)", fontSize: 12, letterSpacing: 1 }}>🐻 BEAR</b>
          <p style={{ fontSize: 13, margin: "6px 0 0", color: "var(--text)" }}>{bear}</p>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { api } from "../api/client";

const METRICS: [string, string, (v: Record<string, unknown>) => string][] = [
  ["nav_usd", "NAV ($)", (v) => Number(v.nav_usd ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 })],
  ["cash_usd", "Cash ($)", (v) => Number(v.cash_usd ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 })],
  ["invested_usd", "Invested ($)", (v) => Number(v.invested_usd ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 })],
  ["daily_pnl_usd", "Daily PnL ($)", (v) => Number(v.daily_pnl_usd ?? 0).toFixed(2)],
  ["unrealized_pnl_usd", "Unrealized PnL ($)", (v) => Number(v.unrealized_pnl_usd ?? 0).toFixed(2)],
  ["realized_pnl_usd", "Realized PnL ($)", (v) => Number(v.realized_pnl_usd ?? 0).toFixed(2)],
  ["var_95_usd", "VaR 95% ($)", (v) => Number(v.var_95_usd ?? 0).toFixed(2)],
  ["gross_exposure_usd", "Gross Exposure ($)", (v) => Number(v.gross_exposure_usd ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 })],
  ["position_count", "Open Positions", (v) => String(v.position_count ?? 0)],
];

export default function PortfolioPage() {
  const [snap, setSnap] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    api.portfolio().then(setSnap).catch(() => setSnap(null));
  }, []);

  return (
    <section>
      <h2>Paper Portfolio</h2>
      {snap ? (
        <>
          <p style={{ color: "#777", fontSize: 13 }}>
            Snapshot: {String(snap.time ?? "n/a").slice(0, 19).replace("T", " ")}
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
            {METRICS.map(([key, label, fmt]) => (
              <div key={key} style={{ border: "1px solid #ddd", borderRadius: 8, padding: 8 }}>
                <div style={{ fontSize: 11, color: "#777" }}>{label}</div>
                <div style={{ fontSize: 17, fontWeight: 700 }}>{fmt(snap)}</div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <p style={{ color: "#888" }}>No portfolio snapshot yet.</p>
      )}
    </section>
  );
}

import { useEffect, useState } from "react";
import { api } from "../api/client";
import Stat from "../components/Stat";
import Panel from "../components/Panel";

const METRICS: [string, string, (v: Record<string, unknown>) => string, boolean][] = [
  ["nav_usd", "NAV", (v) => Number(v.nav_usd ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 }), false],
  ["cash_usd", "Cash", (v) => Number(v.cash_usd ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 }), false],
  ["invested_usd", "Invested", (v) => Number(v.invested_usd ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 }), false],
  ["gross_exposure_usd", "Gross Exposure", (v) => Number(v.gross_exposure_usd ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 }), false],
  ["daily_pnl_usd", "Daily PnL", (v) => Number(v.daily_pnl_usd ?? 0).toFixed(2), true],
  ["unrealized_pnl_usd", "Unrealized PnL", (v) => Number(v.unrealized_pnl_usd ?? 0).toFixed(2), true],
  ["realized_pnl_usd", "Realized PnL", (v) => Number(v.realized_pnl_usd ?? 0).toFixed(2), true],
  ["var_95_usd", "VaR 95%", (v) => Number(v.var_95_usd ?? 0).toFixed(2), true],
  ["position_count", "Open Positions", (v) => String(v.position_count ?? 0), false],
];

export default function PortfolioPage() {
  const [snap, setSnap] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.portfolio().then(setSnap).catch((e) => setError(String((e as Error).message ?? e)));
  }, []);

  return (
    <div className="stack">
      <div className="page-head">
        <div className="title">
          <h2>Paper Portfolio</h2>
          <p>Live position snapshot from the portfolio manager</p>
        </div>
        {snap?.time ? <span className="chip">Snapshot {String(snap.time).slice(0, 19).replace("T", " ")}</span> : null}
      </div>

      {error && <div className="error-box">⚠ {error}</div>}
      {!snap && !error && <div className="loading">Loading portfolio snapshot…</div>}

      {snap && Object.keys(snap).length > 0 && (
        <>
          <div className="kpi-grid">
            {METRICS.map(([key, label, fmt, signed]) => (
              <Stat key={key} label={label} value={
                <span className={signed && Number(snap[key] ?? 0) < 0 ? "down" : signed ? "up" : ""}>
                  {fmt(snap)}
                </span>
              } />
            ))}
          </div>
          <Panel title="Positions & Risk">
            <div className="empty-note">
              Position-level table, Black-Litterman weights and risk-budget usage land here once the
              portfolio manager emits them — currently the snapshot holds aggregate NAV / PnL / VaR.
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

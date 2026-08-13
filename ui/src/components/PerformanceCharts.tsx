import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, AreaChart, Area } from "recharts";
import { fmtUSD } from "../lib/format";

export interface EquityPoint { t: string; equity: number; }

const tooltip = {
  contentStyle: { background: "var(--raised)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12, color: "var(--text)" },
  labelStyle: { color: "var(--text-muted)", fontSize: 11 },
};

/** Equity curve + computed underwater drawdown from real backtest points. */
export default function PerformanceCharts({ points }: { points: EquityPoint[] }) {
  const data = points.map((p, i) => {
    let equity = p.equity;
    let peak = equity;
    for (let j = 0; j <= i; j++) peak = Math.max(peak, points[j].equity);
    return { ...p, drawdown: peak > 0 ? (equity / peak - 1) * 100 : 0 };
  });

  if (!data.length) {
    return <div className="empty-note">No equity curve data for this symbol — run a backtest.</div>;
  }

  return (
    <div className="grid grid-2" style={{ marginTop: 4 }}>
      <div>
        <div className="panel-title" style={{ marginBottom: 8 }}>Equity Curve</div>
        <ResponsiveContainer width="100%" height={230}>
          <AreaChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.35} />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--border-faint)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="t" tick={{ fill: "var(--text-faint)", fontSize: 10.5 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
            <YAxis tick={{ fill: "var(--text-faint)", fontSize: 10.5 }} tickLine={false} axisLine={false} tickFormatter={(v: number) => fmtUSD(v)} width={70} domain={["auto", "auto"]} />
            <Tooltip {...tooltip} formatter={(v) => [fmtUSD(Number(v)), "Equity"]} />
            <Area type="monotone" dataKey="equity" stroke="var(--accent)" strokeWidth={2} fill="url(#eqFill)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div>
        <div className="panel-title" style={{ marginBottom: 8 }}>Drawdown (underwater)</div>
        <ResponsiveContainer width="100%" height={230}>
          <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--border-faint)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="t" tick={{ fill: "var(--text-faint)", fontSize: 10.5 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
            <YAxis tick={{ fill: "var(--text-faint)", fontSize: 10.5 }} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v.toFixed(0)}%`} width={48} />
            <Tooltip {...tooltip} formatter={(v) => [`${Number(v).toFixed(2)}%`, "Drawdown"]} />
            <Line type="monotone" dataKey="drawdown" stroke="var(--red)" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

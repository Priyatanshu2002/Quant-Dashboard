import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

/** Placeholder equity + drawdown charts (wire to /api/backtest/equity when serving). */
export default function PerformanceCharts() {
  const data = Array.from({ length: 120 }, (_, i) => ({
    day: i,
    equity: 100 * Math.exp(0.0015 * i + 0.05 * Math.sin(i / 9)),
    drawdown: -Math.max(0, Math.sin(i / 23) * 8),
  }));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      <div>
        <h4>Equity Curve</h4>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis domain={["auto", "auto"]} />
            <Tooltip />
            <Line type="monotone" dataKey="equity" stroke="#1a5cff" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div>
        <h4>Drawdown (underwater curve)</h4>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="drawdown" stroke="#cf222e" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import {
  BarChart3, Boxes, Layers, PieChart, RefreshCw,
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, OnchainDTO, OnchainQuery } from "../api/client";
import Panel from "../components/Panel";
import Stat from "../components/Stat";

const fmtUsd = (n: number) =>
  n >= 1e9 ? `$${(n / 1e9).toFixed(2)}B`
  : n >= 1e6 ? `$${(n / 1e6).toFixed(1)}M`
  : n >= 1e3 ? `$${(n / 1e3).toFixed(1)}K`
  : `$${n.toFixed(2)}`;

const fmtNum = (n: number) => n.toLocaleString();

// Friendly labels + icons for each persisted Dune query.
const QUERY_META: Record<string, { title: string; icon: React.ElementType }> = {
  dex_volume_by_pair_7d: { title: "Top Pairs by Volume (7d)", icon: Layers },
  dex_volume_daily_7d: { title: "Daily DEX Volume (7d)", icon: BarChart3 },
  dex_volume_by_blockchain_7d: { title: "Volume by Blockchain (7d)", icon: Boxes },
  dex_volume_by_protocol_7d: { title: "Volume by Protocol (7d)", icon: PieChart },
};

function OnchainTable({ q }: { q: OnchainQuery }) {
  if (!q.rows.length) {
    return <div className="chart-tip">No rows yet — run the Dune fetcher to populate.</div>;
  }
  const cols = Object.keys(q.rows[0]);
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {q.rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => {
                const v = r[c];
                const isNum = typeof v === "number";
                return <td key={c} className={isNum ? "mono" : undefined}>
                  {isNum ? fmtNum(v as number) : String(v)}
                </td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function OnchainPage() {
  const [data, setData] = useState<OnchainDTO | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = () => {
    setLoading(true);
    api.onchain()
      .then(setData)
      .catch((e) => setError(String((e as Error).message ?? e)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const daily = useMemo(() => {
    const q = data?.queries.find((x) => x.name === "dex_volume_daily_7d");
    return (q?.rows ?? []).map((r) => ({
      day: String(r.day ?? "").slice(0, 10),
      volume_usd: r.volume_usd as number,
      trades: r.trades as number,
    }));
  }, [data]);

  const live = data?.queries.filter((q) => q.count > 0).length ?? 0;
  const totalVolume = useMemo(() => {
    // 7d total = sum of daily volume rows.
    return daily.reduce((acc, d) => acc + (d.volume_usd || 0), 0);
  }, [daily]);

  return (
    <div className="stack">
      <div className="page-head">
        <div className="title">
          <h2>On-chain Intelligence</h2>
          <p>Dune Analytics · public curated datasets · live DEX flow</p>
        </div>
        <div className="right">
          <span className="status"><span className={`dot ${live > 0 ? "green" : "gray"}`} /> Dune · {live}/4 feeds</span>
          <button className="btn" onClick={load} disabled={loading}>
            <RefreshCw size={14} style={{ marginRight: 6 }} />{loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {error && <div className="error-box">⚠ {error}</div>}
      {!data && !error && <div className="loading">Reading on-chain snapshots…</div>}

      {data && (
        <>
          <div className="kpi-grid">
            <Stat label="DEX volume (7d)" value={fmtUsd(totalVolume)} foot="across chains" />
            <Stat label="Daily snapshots" value={`${daily.length}`} foot="days captured" />
            <Stat label="Queries live" value={`${live}/4`} foot="from Dune API" />
            <Stat label="Source" value={<span className="mono">Dune</span>} foot="SQL-direct" />
          </div>

          {daily.length > 0 && (
            <Panel title="DEX Volume Trend" hint="7-day daily aggregated USD volume">
              <div className="chart">
                <ResponsiveContainer width="100%" height={240}>
                  <AreaChart data={daily} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="vol" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#6366f1" stopOpacity={0.4} />
                        <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="day" tick={{ fontSize: 11, fill: "var(--muted)" }} />
                    <YAxis tickFormatter={(v) => fmtUsd(v)} tick={{ fontSize: 11, fill: "var(--muted)" }} width={70} />
                    <Tooltip
                      contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8 }}
                      formatter={(v: number) => [fmtUsd(v), "Volume"]}
                    />
                    <Area type="monotone" dataKey="volume_usd" stroke="#6366f1" strokeWidth={2} fill="url(#vol)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          )}

          <div className="grid grid-2">
            {data.queries.map((q) => {
              const meta = QUERY_META[q.name] ?? { title: q.name, icon: BarChart3 };
              const Icon = meta.icon;
              return (
                <Panel key={q.name} title={meta.title} hint={`${q.count} rows · ${q.stored_at ? `snapshot ${q.stored_at}` : "empty"}`}>
                  <div className="panel-head" style={{ marginBottom: 8 }}>
                    <div className="icon"><Icon /></div>
                  </div>
                  <OnchainTable q={q} />
                </Panel>
              );
            })}
          </div>

          {data.note && <div className="chart-tip">{data.note}</div>}
        </>
      )}
    </div>
  );
}

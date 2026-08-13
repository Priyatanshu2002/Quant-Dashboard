import { useEffect, useState } from "react";
import {
  Activity, BarChart3, Briefcase, Database, FileText, Gauge, LineChart,
  MessageSquare, Newspaper, Scale, Server, TrendingUp,
} from "lucide-react";
import { api, MonitoringDTO, ServiceStatus } from "../api/client";
import Panel from "../components/Panel";
import Stat from "../components/Stat";

const FEED_ICONS: Record<string, React.ElementType> = {
  prices: TrendingUp, features: Gauge, fundamentals: BarChart3, statements: FileText,
  profiles: Briefcase, sentiment: Newspaper, llm_analyses: MessageSquare,
  earnings: Activity, macro: LineChart, debate: Scale, portfolio: Briefcase,
};

export default function MonitoringPage() {
  const [data, setData] = useState<MonitoringDTO | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.monitoring().then(setData).catch((e) => setError(String((e as Error).message ?? e)));
  }, []);

  const ok = data?.feeds.filter((f) => f.status === "ok").length ?? 0;
  const runningSvcs = data?.services.filter((s) => s.running).length ?? 0;

  return (
    <div className="stack">
      <div className="page-head">
        <div className="title">
          <h2>System Monitoring</h2>
          <p>Data-feed coverage · storage health · infrastructure status</p>
        </div>
        {data && (
          <span className="status"><span className="dot green" /> {ok}/{data.feeds.length} feeds live</span>
        )}
      </div>

      {error && <div className="error-box">⚠ {error}</div>}
      {!data && !error && <div className="loading">Reading system state…</div>}

      {data && (
        <>
          <div className="kpi-grid">
            <Stat label="Storage backend" value={<span className="mono">{data.backend}</span>} foot={data.db_size_mb != null ? `${data.db_size_mb} MB` : undefined} />
            <Stat label="Data feeds" value={`${ok}/${data.feeds.length}`} foot="live sources" />
            <Stat label="Infrastructure" value={`${runningSvcs}/${data.services.length}`} foot="containerized services up" />
            <Stat label="LLM spend" value={data.llm_spend != null ? `$${data.llm_spend}` : "—"} foot="not tracked yet" />
          </div>

          <Panel title="Data Feed Coverage" hint="Row counts per source">
            <div className="feed-status">
              {data.feeds.map((f) => {
                const Icon = FEED_ICONS[f.key] ?? Database;
                const tone = f.status === "ok" ? "green" : f.status === "empty" ? "amber" : "red";
                return (
                  <div key={f.key} className="feed-tile">
                    <div className="icon"><Icon /></div>
                    <div className="info" style={{ flex: 1 }}>
                      <div className="name">{f.label}</div>
                      <div className="meta mono">{f.count.toLocaleString()} rows</div>
                    </div>
                    <span className={`badge ${tone}`}>{f.status}</span>
                  </div>
                );
              })}
            </div>
          </Panel>

          <div className="grid grid-2">
            <Panel title="Infrastructure" hint="Containerized services">
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Service</th><th>Port</th><th>Status</th></tr></thead>
                  <tbody>
                    {data.services.map((s: ServiceStatus) => (
                      <tr key={s.name}>
                        <td>{s.name}</td>
                        <td className="mono">{s.port}</td>
                        <td>
                          <span className="status">
                            <span className={`dot ${s.running ? "green" : "gray"}`} />
                            {s.running ? "Up" : "Down"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>

            <Panel title="Storage" hint="Primary database">
              <Stat label="Database path" value={<span className="mono small" style={{ fontSize: 12 }}>{data.db_path ?? "—"}</span>} sm />
              <div style={{ height: 12 }} />
              <Stat label="Size" value={data.db_size_mb != null ? `${data.db_size_mb} MB` : "—"} sm />
              <div style={{ height: 12 }} />
              <div className="chart-tip">{data.note}</div>
              <div style={{ height: 12 }} />
              <div className="feed-status" style={{ gridTemplateColumns: "1fr" }}>
                <div className="feed-tile">
                  <div className="icon"><Server /></div>
                  <div className="info">
                    <div className="name">SQLite (dev)</div>
                    <div className="meta">TimescaleDB backend partially wired — {data.backend} active</div>
                  </div>
                </div>
              </div>
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}

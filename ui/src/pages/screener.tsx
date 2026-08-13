import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowUpDown, ChevronDown, ChevronUp, Search } from "lucide-react";
import { api, MarketRow } from "../api/client";
import TickerSearch from "../components/TickerSearch";
import { fmtUSD } from "../lib/format";

type SortKey = keyof Pick<MarketRow, "symbol" | "asset_class" | "price" | "change_pct" | "volume" | "market_cap" | "composite" | "technical" | "fundamental" | "sentiment" | "macro" | "momentum">;

const CLASSES = ["EQUITY_US", "EQUITY_IN", "CRYPTO", "ETF", "BOND", "FOREX"];

function compositeTone(c: number): string {
  if (c >= 60) return "green";
  if (c >= 45) return "amber";
  return "red";
}

export default function ScreenerPage() {
  const nav = useNavigate();
  const [rows, setRows] = useState<MarketRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [assetClass, setAssetClass] = useState("ALL");
  const [sortKey, setSortKey] = useState<SortKey>("composite");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  useEffect(() => {
    let alive = true;
    api.screenerMarket()
      .then((d) => { if (alive) setRows(d.rows); })
      .catch((e) => { if (alive) setError(String((e as Error).message ?? e)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    let out = rows.filter((r) => {
      if (assetClass !== "ALL" && r.asset_class !== assetClass) return false;
      if (!q) return true;
      return r.symbol.toUpperCase().includes(q) || (r.name ?? "").toUpperCase().includes(q);
    });
    out = [...out].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * sortDir;
      return String(av ?? "").localeCompare(String(bv ?? "")) * sortDir;
    });
    return out;
  }, [rows, query, assetClass, sortKey, sortDir]);

  const toggleSort = (k: SortKey) => {
    if (k === sortKey) setSortDir((d) => (d === 1 ? -1 : 1));
    else { setSortKey(k); setSortDir(k === "symbol" || k === "asset_class" ? 1 : -1); }
  };

  const SortHead = ({ k, children }: { k: SortKey; children: string }) => (
    <th style={{ cursor: "pointer" }} onClick={() => toggleSort(k)}>
      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
        {children}
        {sortKey === k ? (sortDir === 1 ? <ChevronUp size={12} /> : <ChevronDown size={12} />) : <ArrowUpDown size={11} opacity={0.5} />}
      </span>
    </th>
  );

  const open = (symbol: string) => nav(`/financials/${symbol}`);

  return (
    <div className="stack">
      <div className="page-head">
        <div className="title">
          <h2>Market Screener</h2>
          <p>{loading ? "Loading market…" : `${rows.length} instruments across 6 asset classes — click any row to drill in`}</p>
        </div>
        {!loading && <span className="chip">{rows.length} symbols</span>}
      </div>

      <div className="toolbar">
        <TickerSearch to="/financials" placeholder="Jump to a ticker…" width={300} />
        <div className="row" style={{ flex: 1, maxWidth: 260 }}>
          <Search size={15} style={{ color: "var(--text-faint)" }} />
          <input className="input" style={{ flex: 1 }} placeholder="Filter table…"
            value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
        <select className="input" value={assetClass} onChange={(e) => setAssetClass(e.target.value)}>
          <option value="ALL">All asset classes</option>
          {CLASSES.map((c) => <option key={c} value={c}>{c.replace("_", " ")}</option>)}
        </select>
        <span className="chip">{filtered.length} shown</span>
      </div>

      {error && <div className="error-box">⚠ {error}</div>}
      {loading && <div className="loading">Scoring the universe…</div>}

      {!loading && (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <SortHead k="symbol">Symbol</SortHead>
                <th>Name</th>
                <SortHead k="asset_class">Class</SortHead>
                <th>Sector</th>
                <SortHead k="price">Price</SortHead>
                <SortHead k="change_pct">Chg %</SortHead>
                <SortHead k="market_cap">Mkt Cap</SortHead>
                <SortHead k="volume">Volume</SortHead>
                <SortHead k="composite">Score</SortHead>
                <SortHead k="technical">Tech</SortHead>
                <SortHead k="fundamental">Fund</SortHead>
                <SortHead k="sentiment">Sent</SortHead>
                <SortHead k="macro">Macro</SortHead>
                <SortHead k="momentum">Mom</SortHead>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.symbol} onClick={() => open(r.symbol)} style={{ cursor: "pointer" }}>
                  <td><span className="symbol-link">{r.symbol}</span></td>
                  <td className="muted" style={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis" }}>{r.name ?? "—"}</td>
                  <td><span className="chip">{r.asset_class.replace("_", " ")}</span></td>
                  <td className="muted small">{r.sector ?? "—"}</td>
                  <td className="mono">{r.price != null ? fmtUSD(r.price, 2) : "—"}</td>
                  <td className="mono" style={{ color: r.change_pct == null ? "var(--text-faint)" : r.change_pct >= 0 ? "var(--green)" : "var(--red)", fontWeight: 700 }}>
                    {r.change_pct != null ? `${r.change_pct >= 0 ? "+" : ""}${r.change_pct.toFixed(2)}%` : "—"}
                  </td>
                  <td className="mono">{r.market_cap ? fmtUSD(r.market_cap) : "—"}</td>
                  <td className="mono">{r.volume != null ? r.volume.toLocaleString() : "—"}</td>
                  <td><span className={`badge ${compositeTone(r.composite)}`}>{r.composite.toFixed(1)}</span></td>
                  <td className="mono">{r.technical.toFixed(0)}</td>
                  <td className="mono">{r.fundamental.toFixed(0)}</td>
                  <td className="mono">{r.sentiment.toFixed(0)}</td>
                  <td className="mono">{r.macro.toFixed(0)}</td>
                  <td className="mono">{r.momentum.toFixed(0)}</td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={14} className="empty-note">No instruments match — clear filters.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

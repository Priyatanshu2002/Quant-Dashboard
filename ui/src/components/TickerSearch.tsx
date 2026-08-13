import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import { api, UniverseEntry } from "../api/client";

interface Props {
  to: string; // route to navigate to, e.g. "/financials" (symbol appended)
  placeholder?: string;
  width?: number;
}

/** Searchable ticker dropdown — find any instrument by symbol or company name,
 * no need to memorize tickers. Opens on focus, filters as you type, Enter/click
 * navigates to the given route with the selected symbol. */
export default function TickerSearch({ to, placeholder = "Search symbol or name…", width = 300 }: Props) {
  const nav = useNavigate();
  const [list, setList] = useState<UniverseEntry[]>([]);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.screenerUniverse().then((d) => setList(d.rows)).catch(() => {});
  }, []);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const q = query.trim().toUpperCase();
  const filtered = q
    ? list.filter((r) => r.symbol.toUpperCase().includes(q) || (r.name ?? "").toUpperCase().includes(q)).slice(0, 12)
    : list.slice(0, 12);

  const go = (symbol: string) => {
    setQuery(""); setOpen(false);
    nav(`${to}/${symbol}`);
  };

  return (
    <div ref={boxRef} style={{ position: "relative", width }}>
      <div className="row" style={{ position: "relative" }}>
        <Search size={15} style={{ color: "var(--text-faint)", position: "absolute", left: 10, pointerEvents: "none" }} />
        <input
          className="input"
          style={{ width: "100%", paddingLeft: 32 }}
          placeholder={placeholder}
          value={query}
          onChange={(e) => { setQuery(e.target.value); setOpen(true); setHighlight(0); }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setHighlight((h) => Math.min(h + 1, filtered.length - 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setHighlight((h) => Math.max(h - 1, 0)); }
            else if (e.key === "Enter" && filtered[highlight]) { go(filtered[highlight].symbol); }
            else if (e.key === "Escape") setOpen(false);
          }}
        />
      </div>
      {open && filtered.length > 0 && (
        <div style={{
          position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, zIndex: 50,
          background: "var(--raised)", border: "1px solid var(--border)", borderRadius: 10,
          boxShadow: "var(--shadow-2)", overflow: "hidden", maxHeight: 340, overflowY: "auto",
        }}>
          {filtered.map((r, i) => (
            <button key={r.symbol} type="button"
              onMouseEnter={() => setHighlight(i)}
              onClick={() => go(r.symbol)}
              style={{
                display: "flex", alignItems: "center", gap: 10, width: "100%", textAlign: "left",
                padding: "9px 12px", background: i === highlight ? "var(--accent-soft)" : "transparent",
                border: "none", cursor: "pointer", color: "var(--text)", fontFamily: "var(--font)", fontSize: 13,
              }}>
              <span style={{ fontFamily: "var(--mono)", fontWeight: 700, color: "var(--accent-strong)", width: 78 }}>{r.symbol}</span>
              <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--text-muted)" }}>
                {r.name || "—"}
              </span>
              <span className="chip">{r.asset_class.replace("_", " ")}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// Shared formatting helpers (tabular-friendly, compact).

export function fmtUSD(v: unknown, digits = 0): string {
  const n = Number(v ?? 0);
  if (!isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${n.toFixed(digits)}`;
}

export function fmtNum(v: unknown, digits = 1): string {
  const n = Number(v ?? 0);
  if (!isFinite(n)) return "—";
  return n.toFixed(digits);
}

export function fmtPct(v: unknown, digits = 1): string {
  const n = Number(v ?? 0);
  if (!isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtPctSigned(v: unknown, digits = 1): string {
  const n = Number(v ?? 0);
  if (!isFinite(n)) return "—";
  const s = n > 0 ? "+" : "";
  return `${s}${(n * 100).toFixed(digits)}%`;
}

export function fmtSigned(v: unknown, digits = 2): string {
  const n = Number(v ?? 0);
  if (!isFinite(n)) return "—";
  const s = n > 0 ? "+" : "";
  return `${s}${n.toFixed(digits)}`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.replace(" ", "T"));
  if (isNaN(d.getTime())) return iso.slice(0, 10);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.replace(" ", "T"));
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export function compactNum(v: unknown): string {
  const n = Number(v ?? 0);
  if (!isFinite(n)) return "—";
  return n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(Math.round(n));
}

export function pctColor(v: unknown): string {
  const n = Number(v ?? 0);
  if (n > 0) return "var(--green)";
  if (n < 0) return "var(--red)";
  return "var(--text-muted)";
}

export function sentiColor(score: number): string {
  if (score > 0.25) return "var(--green)";
  if (score < -0.25) return "var(--red)";
  return "var(--amber)";
}

export function sentiLabel(score: number): string {
  if (score > 0.25) return "Bullish";
  if (score < -0.25) return "Bearish";
  return "Neutral";
}

interface Props {
  marginOfSafety: number; // (intrinsic - price) / price, e.g. -0.459
  intrinsic?: number | null;
  price?: number | null;
}

const ZONES = [
  { min: -Infinity, max: -0.15, cls: "over", label: "Overvalued" },
  { min: -0.15, max: 0.05, cls: "fair", label: "Fair Value" },
  { min: 0.05, max: Infinity, cls: "under", label: "Undervalued" },
];

/** Plan §11.2 DCFGauge — undervalued (green) / fair (amber) / overvalued (red). */
export default function DCFGauge({ marginOfSafety, intrinsic, price }: Props) {
  const mos = Number(marginOfSafety ?? 0);
  const clamped = Math.max(-0.3, Math.min(0.3, mos));
  const pct = ((clamped + 0.3) / 0.6) * 100;

  const zone = ZONES.find((z) => mos >= z.min && mos < z.max) ?? ZONES[1];
  const label = zone.label;
  const color = zone.cls === "over" ? "var(--red)" : zone.cls === "under" ? "var(--green)" : "var(--amber)";

  return (
    <div className="dcf-wrap">
      <div className="row-between">
        <span className="big" style={{ color }}>{label.toUpperCase()}</span>
        <span className="mono" style={{ color, fontSize: 20, fontWeight: 800 }}>
          {mos >= 0 ? "+" : ""}{(mos * 100).toFixed(1)}%
        </span>
      </div>
      <div className="dcf-gauge">
        <div className="dcf-marker" style={{ left: `${pct}%` }} />
      </div>
      <div className="dcf-scale">
        <span>-30%</span><span>-15%</span><span>0%</span><span>+15%</span><span>+30%</span>
      </div>
      <div className="dcf-zones">
        {ZONES.map((z) => (
          <div key={z.label} className={`dcf-zone ${z.cls}`}>
            <div className="zlabel">{z.label}</div>
            <div className="zval">
              {z.min === -Infinity ? "< -15%" : z.max === Infinity ? "> +5%" : `${(z.min * 100).toFixed(0)}% … ${(z.max * 100).toFixed(0)}%`}
            </div>
          </div>
        ))}
      </div>
      {intrinsic != null && price != null && (
        <div className="row" style={{ marginTop: 12, fontSize: 12.5, color: "var(--text-muted)" }}>
          <span>Intrinsic: <b className="mono" style={{ color: "var(--text)" }}>${Number(intrinsic).toFixed(2)}</b></span>
          <span>·</span>
          <span>Price: <b className="mono" style={{ color: "var(--text)" }}>${Number(price).toFixed(2)}</b></span>
        </div>
      )}
    </div>
  );
}

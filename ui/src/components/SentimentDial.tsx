import { sentiColor, sentiLabel } from "../lib/format";

interface Props {
  score: number; // -1..1
  label?: string;
  size?: number;
}

/** Arc dial showing a -1..+1 sentiment score with a colored sweep. */
export default function SentimentDial({ score, label, size = 150 }: Props) {
  const clamped = Math.max(-1, Math.min(1, score));
  const color = sentiColor(clamped);
  const r = 58;
  const cx = 75;
  const cy = 75;
  const startAngle = 180; // left
  const endAngle = 0;     // right (sweep across the top)
  const frac = (clamped + 1) / 2; // 0..1

  const polar = (angleDeg: number, radius: number) => {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy - radius * Math.sin(rad) };
  };
  const arcPath = (fromAngle: number, toAngle: number, radius: number) => {
    const from = polar(fromAngle, radius);
    const to = polar(toAngle, radius);
    const large = toAngle - fromAngle > 180 ? 1 : 0;
    return `M ${from.x} ${from.y} A ${radius} ${radius} 0 ${large} 1 ${to.x} ${to.y}`;
  };

  const sweepAngle = startAngle - (startAngle - endAngle) * frac;
  const tip = polar(sweepAngle, r - 12);

  return (
    <div className="dial" style={{ width: size, height: size }}>
      <svg viewBox="0 0 150 150" width={size} height={size}>
        <path d={arcPath(startAngle, endAngle, r)} stroke="#1d2942" strokeWidth={10} fill="none" strokeLinecap="round" />
        <path d={arcPath(startAngle, sweepAngle, r)} stroke={color} strokeWidth={10} fill="none" strokeLinecap="round" style={{ transition: "all .5s ease" }} />
        <circle cx={tip.x} cy={tip.y} r={5.5} fill={color} />
      </svg>
      <div className="dial-center">
        <div className="dial-score" style={{ color }}>{clamped >= 0 ? "+" : ""}{clamped.toFixed(2)}</div>
        <div className="dial-label">{label ?? sentiLabel(clamped)}</div>
      </div>
    </div>
  );
}

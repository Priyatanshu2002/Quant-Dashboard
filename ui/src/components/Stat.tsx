interface Props {
  label: string;
  value: React.ReactNode;
  foot?: React.ReactNode;
  color?: string;
  sm?: boolean;
}

/** Label + value tile. */
export default function Stat({ label, value, foot, color, sm }: Props) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className={`value ${sm ? "sm" : ""}`} style={color ? { color } : undefined}>
        {value}
      </div>
      {foot && <div className="foot">{foot}</div>}
    </div>
  );
}

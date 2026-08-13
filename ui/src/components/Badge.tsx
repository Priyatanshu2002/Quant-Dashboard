interface Props {
  tone?: "green" | "red" | "amber" | "blue" | "gray";
  children: React.ReactNode;
}

/** Small colored pill. */
export default function Badge({ tone = "gray", children }: Props) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

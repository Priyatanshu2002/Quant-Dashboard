import type { ReactNode } from "react";

interface Props {
  title?: string;
  hint?: string;
  children: ReactNode;
  className?: string;
  right?: ReactNode;
}

/** Card container with an uppercase title bar (design-system panel). */
export default function Panel({ title, hint, children, className = "", right }: Props) {
  return (
    <section className={`panel ${className}`}>
      {title && (
        <h3 className="panel-title">
          <span>{title}</span>
          <span className="row">
            {hint && <span className="hint">{hint}</span>}
            {right}
          </span>
        </h3>
      )}
      {children}
    </section>
  );
}

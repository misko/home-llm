import type { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  accent?: "blue" | "green" | "amber" | "neutral";
}

export function MetricCard({ label, value, detail, accent = "neutral" }: Props) {
  return (
    <article className={`metric-card accent-${accent}`}>
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      {detail && <span className="metric-detail">{detail}</span>}
    </article>
  );
}

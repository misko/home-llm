import type { ReactNode } from "react";

export function EmptyState({ icon, title, detail, action }: {
  icon: ReactNode;
  title: string;
  detail: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <span className="empty-icon">{icon}</span>
      <h2>{title}</h2>
      <p>{detail}</p>
      {action}
    </div>
  );
}

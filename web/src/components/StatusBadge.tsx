import { Activity, AlertTriangle, LoaderCircle, Moon } from "lucide-react";

interface Props {
  ready?: boolean;
  active?: boolean;
  phase?: string;
}

export function StatusBadge({ ready, active, phase }: Props) {
  if (ready) return <span className="status-badge ready"><Activity size={14} /> Ready</span>;
  if (phase === "starting" || phase === "stopping") {
    return <span className="status-badge working"><LoaderCircle className="spin" size={14} /> {phase}</span>;
  }
  if (active || phase === "failed") {
    return <span className="status-badge danger"><AlertTriangle size={14} /> {phase ?? "Unavailable"}</span>;
  }
  return <span className="status-badge idle"><Moon size={14} /> Inactive</span>;
}

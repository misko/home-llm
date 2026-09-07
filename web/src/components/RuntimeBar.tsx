import { Cpu, Gauge, RefreshCw } from "lucide-react";
import { useRuntime } from "../hooks/useConsoleData";
import { StatusBadge } from "./StatusBadge";

export function RuntimeBar({ onSwitch }: { onSwitch: () => void }) {
  const runtime = useRuntime();
  const data = runtime.data;
  const memory = data?.memory_used_mib != null && data.memory_total_mib
    ? `${(data.memory_used_mib / 1024).toFixed(1)} / ${(data.memory_total_mib / 1024).toFixed(0)} GiB`
    : "Awaiting telemetry";

  return (
    <header className="runtime-strip">
      <div className="runtime-identity">
        <span className="eyebrow">ACTIVE DEPLOYMENT</span>
        <strong>{data?.public_alias ?? "No active model"}</strong>
        <span className="muted">{data?.model_name ?? (runtime.isError ? "Console API unavailable" : "")}</span>
      </div>
      <div className="runtime-facts">
        {runtime.isFetching && !data ? <span className="status-badge working"><RefreshCw className="spin" size={14} /> Connecting</span> : <StatusBadge ready={data?.ready} active={data?.active} phase={data?.phase} />}
        <span className="metric"><Cpu size={13} /><b>{memory}</b></span>
        <span className="metric"><Gauge size={13} /><b>{data?.context_size ? `${Math.round(data.context_size / 1024)}K` : "—"}</b> context</span>
        <button className="primary-button compact" onClick={onSwitch}>Switch model</button>
      </div>
    </header>
  );
}

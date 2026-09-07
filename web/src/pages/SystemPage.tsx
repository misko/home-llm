import {
  Check,
  Clipboard,
  Cpu,
  Fingerprint,
  Globe2,
  LockKeyhole,
  Server,
  ShieldCheck,
} from "lucide-react";
import { useState } from "react";
import { shortHash } from "../api/format";
import { MetricCard } from "../components/MetricCard";
import { useRuntime, useSystem } from "../hooks/useConsoleData";

export function SystemPage() {
  const system = useSystem();
  const runtime = useRuntime();
  const [copied, setCopied] = useState(false);
  const example = [
    "curl http://127.0.0.1:14000/v1/chat/completions \\",
    "  -H 'Content-Type: application/json' \\",
    `  -d '{"model":"${runtime.data?.public_alias ?? "local-fast"}","messages":[{"role":"user","content":"Hello"}]}'`,
  ].join("\n");

  return (
    <section className="page workspace-wide">
      <div className="page-head">
        <div>
          <span className="eyebrow">CONTROL PLANE</span>
          <h1>System</h1>
          <p>Gateway, hardware, catalog, and reviewed runtime identity.</p>
        </div>
      </div>
      <div className="metric-grid">
        <MetricCard label="Gateway" value={runtime.data?.healthy ? "Healthy" : "Waiting"} detail="127.0.0.1:14000" accent={runtime.data?.healthy ? "green" : "amber"} />
        <MetricCard label="GPU" value={system.data?.gpu_name ?? "—"} detail={system.data?.driver_version ? `Driver ${system.data.driver_version}` : "Hardware discovery"} accent="blue" />
        <MetricCard label="Catalog" value={system.data?.catalog_valid ? "Valid" : "Unavailable"} detail={shortHash(system.data?.catalog_revision)} accent={system.data?.catalog_valid ? "green" : "amber"} />
      </div>
      <div className="system-grid">
        <article className="panel identity-panel">
          <div className="panel-head">
            <div><h2>Runtime identity</h2><p>Reviewed executable evidence for the active deployment</p></div>
            <Fingerprint size={21} />
          </div>
          <dl>
            <dt><span><Server size={14} /> Runtime lock</span></dt><dd>{system.data?.runtime_lock_id ?? "—"}</dd>
            <dt><span><ShieldCheck size={14} /> Executable SHA-256</span></dt><dd><code>{system.data?.runtime_sha256 ?? "—"}</code></dd>
            <dt><span><Cpu size={14} /> Version evidence</span></dt><dd>{system.data?.runtime_version ?? "—"}</dd>
            <dt><span><LockKeyhole size={14} /> Data plane</span></dt><dd>{system.data?.data_root_label ?? "Managed local storage"}</dd>
          </dl>
        </article>
        <article className="panel api-panel">
          <div className="panel-head">
            <div><h2>OpenAI-compatible API</h2><p>Stable loopback endpoint for local clients</p></div>
            <Globe2 size={21} />
          </div>
          <div className="endpoint-box"><span>Base URL</span><code>http://127.0.0.1:14000/v1</code></div>
          <div className="code-block">
            <button onClick={() => {
              void navigator.clipboard?.writeText(example);
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}>
              {copied ? <Check size={14} /> : <Clipboard size={14} />} {copied ? "Copied" : "Copy"}
            </button>
            <pre>{example}</pre>
          </div>
          <p className="modal-note">The gateway remains bound to loopback. Remote access requires a separately reviewed TLS proxy and authentication policy.</p>
        </article>
      </div>
    </section>
  );
}

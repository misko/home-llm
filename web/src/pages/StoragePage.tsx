import { Database, HardDrive, Layers3, ShieldCheck } from "lucide-react";
import { formatBytes, shortHash } from "../api/format";
import { MetricCard } from "../components/MetricCard";
import { useStorage } from "../hooks/useConsoleData";

export function StoragePage() {
  const storage = useStorage();
  const data = storage.data;
  const budget = 2_700_000_000_000;
  const usedPercent = data ? Math.min(100, (data.used_bytes / budget) * 100) : 0;

  return (
    <section className="page workspace-wide">
      <div className="page-head">
        <div>
          <span className="eyebrow">CONTENT-ADDRESSED STORAGE</span>
          <h1>Storage</h1>
          <p>Physical capacity, immutable artifacts, and the protected operating reserve.</p>
        </div>
      </div>

      <div className="capacity-panel panel">
        <div>
          <div className="panel-head">
            <div>
              <h2>2.7 TB model budget</h2>
              <p>Hot artifacts, acquisition cache, work space, and safety reserve</p>
            </div>
            <strong>{data ? formatBytes(data.free_bytes) : "—"} free</strong>
          </div>
          <div className="capacity-track"><span style={{ width: `${usedPercent}%` }} /></div>
          <div className="capacity-legend">
            <span><i className="blue" /> Physical used {data ? formatBytes(data.used_bytes) : "—"}</span>
            <span><i className="amber" /> Reserve {data ? formatBytes(data.reserve_bytes) : "—"}</span>
            <span><i /> Available {data ? formatBytes(data.free_bytes) : "—"}</span>
          </div>
        </div>
      </div>

      <div className="metric-grid">
        <MetricCard label="Logical artifacts" value={data ? formatBytes(data.logical_bytes) : "—"} detail={`${data?.artifact_count ?? 0} registered artifacts`} accent="blue" />
        <MetricCard label="Unique CAS bytes" value={data ? formatBytes(data.unique_blob_bytes) : "—"} detail="SHA-256 byte authority" accent="green" />
        <MetricCard label="Protected reserve" value={data ? formatBytes(data.reserve_bytes) : "—"} detail="Never consumed by normal ingestion" accent="amber" />
      </div>

      <div className="panel">
        <div className="panel-head">
          <div><h2>Installed artifacts</h2><p>Loader-ready views backed by immutable CAS blobs</p></div>
          <span className="verified-label"><ShieldCheck size={15} /> Verified on access</span>
        </div>
        {storage.isError ? (
          <div className="error-panel">{storage.error.message}</div>
        ) : (
          <div className="artifact-list">
            {data?.artifacts.map((artifact) => (
              <article key={artifact.id}>
                <span className="artifact-icon"><Database size={19} /></span>
                <div className="artifact-name"><strong>{artifact.id}</strong><span>{artifact.model_id}</span></div>
                <span className="format-chip">{artifact.format.toUpperCase()} · {artifact.quantization}</span>
                <div className="artifact-size"><strong>{formatBytes(artifact.logical_bytes)}</strong><span>logical</span></div>
                <div className="artifact-hash"><code>{shortHash(artifact.manifest_sha256, 12)}</code><span>manifest</span></div>
                {artifact.active && <span className="active-label"><Layers3 size={13} /> Active</span>}
              </article>
            ))}
          </div>
        )}
      </div>

      <div className="policy-note">
        <HardDrive size={18} />
        <div>
          <strong>Safe cleanup remains preview-first</strong>
          <p>The first console release keeps garbage collection read-only. Use the existing CLI to review and apply collection until the destructive confirmation flow receives dedicated E2E coverage.</p>
        </div>
      </div>
    </section>
  );
}

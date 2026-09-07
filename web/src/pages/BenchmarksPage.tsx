import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BarChart3, CheckCircle2, ChevronRight, FlaskConical, Gauge, Play, RefreshCw, TriangleAlert } from "lucide-react";
import { consoleApi } from "../api/client";
import { formatDuration, formatTimestamp } from "../api/format";
import { MetricCard } from "../components/MetricCard";
import { Modal } from "../components/Modal";
import { useRuns, useRuntime } from "../hooks/useConsoleData";

export function BenchmarksPage() {
  const runs = useRuns();
  const runtime = useRuntime();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [runDialog, setRunDialog] = useState(false);
  const [suite, setSuite] = useState("smoke");
  const [operation, setOperation] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const detail = useQuery({ queryKey: ["console", "run", selectedRun], queryFn: () => consoleApi.run(selectedRun!), enabled: Boolean(selectedRun) });
  const clean = runs.data?.runs.filter((run) => run.status === "completed" && run.error_count === 0 && run.warmup_error_count === 0) ?? [];
  const fastest = useMemo(() => clean.filter((run) => run.mean_tokens_per_second != null).sort((a, b) => (b.mean_tokens_per_second ?? 0) - (a.mean_tokens_per_second ?? 0))[0], [clean]);

  async function launch() {
    if (!runtime.data?.deployment_id) return;
    setError(null);
    try {
      const created = await consoleApi.runBenchmark(runtime.data.deployment_id, suite, runtime.data.revision);
      setOperation(created.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Benchmark could not be started.");
    }
  }

  return (
    <section className="page workspace-wide">
      <div className="page-head"><div><span className="eyebrow">IMMUTABLE EVIDENCE</span><h1>Benchmarks</h1><p>Run cataloged suites and compare only measurements with identical verified contracts.</p></div><button className="primary-button" disabled={!runtime.data?.ready} onClick={() => setRunDialog(true)}><Play size={15} /> Run benchmark</button></div>
      <div className="metric-grid">
        <MetricCard label="Indexed runs" value={runs.data?.runs.length ?? "—"} detail="Verified bundles" accent="blue" />
        <MetricCard label="Clean runs" value={clean.length} detail="Zero measured or warmup errors" accent="green" />
        <MetricCard label="Fastest decode" value={fastest?.mean_tokens_per_second ? `${fastest.mean_tokens_per_second.toFixed(1)} tok/s` : "—"} detail={fastest?.deployment_id ?? "No performance runs"} accent="amber" />
      </div>
      <div className="panel benchmark-panel">
        <div className="panel-head"><div><h2>Run history</h2><p>Most recent immutable benchmark evidence</p></div><span className="table-count">{runs.data?.runs.length ?? 0} runs</span></div>
        {runs.isPending ? <div className="skeleton table" /> : runs.isError ? <div className="error-panel"><TriangleAlert size={22} /> {runs.error.message}</div> : (
          <div className="table-scroll"><table><thead><tr><th>Run</th><th>Deployment</th><th>Suite</th><th>Status</th><th>Pass</th><th>Latency</th><th>Decode</th><th /></tr></thead><tbody>
            {runs.data?.runs.map((run) => <tr key={run.run_id}>
              <td><button className="run-link" onClick={() => setSelectedRun(run.run_id)}>{run.run_id}</button><small>{formatTimestamp(run.started_at)}</small></td>
              <td>{run.deployment_id}</td><td><code>{run.suite_id}@{run.suite_version}</code></td>
              <td><span className={`run-status ${run.status === "completed" ? "clean" : "warning"}`}>{run.status === "completed" ? <CheckCircle2 size={13} /> : <TriangleAlert size={13} />}{run.status}</span></td>
              <td>{run.pass_rate == null ? "Unscored" : `${Math.round(run.pass_rate * 100)}%`}</td><td>{formatDuration(run.mean_latency_ms)}</td><td>{run.mean_tokens_per_second ? `${run.mean_tokens_per_second.toFixed(1)} tok/s` : "—"}</td>
              <td><button className="icon-button" aria-label={`Inspect ${run.run_id}`} onClick={() => setSelectedRun(run.run_id)}><ChevronRight size={17} /></button></td>
            </tr>)}
          </tbody></table></div>
        )}
      </div>

      <Modal open={runDialog} title="Run a benchmark" onClose={() => !operation && setRunDialog(false)}>
        <div className="form-stack"><label>Suite<select value={suite} onChange={(event) => setSuite(event.target.value)}><option value="smoke">Smoke · functional contract</option><option value="perf-4090">Performance · RTX 4090</option></select></label>
          <div className="selection-summary"><FlaskConical size={20} /><div><strong>{runtime.data?.public_alias}</strong><span>{runtime.data?.deployment_id}</span></div></div>
          <p className="modal-note">Performance benchmarking temporarily reserves inference admission so unrelated chat traffic cannot alter the measurement.</p>
          {error && <div className="inline-error">{error}</div>}{operation && <div className="operation-banner running"><RefreshCw className="spin" size={17} /><div><strong>Benchmark queued</strong><span>Operation {operation}</span></div></div>}
          <div className="modal-actions"><button className="ghost-button" disabled={Boolean(operation)} onClick={() => setRunDialog(false)}>Cancel</button><button className="primary-button" disabled={Boolean(operation)} onClick={() => void launch()}><Play size={15} /> Start suite</button></div>
        </div>
      </Modal>

      <Modal open={Boolean(selectedRun)} title="Run evidence" onClose={() => setSelectedRun(null)}>
        {detail.isPending ? <div className="skeleton tall" /> : detail.data && <div className="run-detail">
          <div className="detail-title"><div><span className="eyebrow">{detail.data.suite_id}@{detail.data.suite_version}</span><h3>{detail.data.deployment_id}</h3></div><span className="run-status clean"><CheckCircle2 size={13} />{detail.data.status}</span></div>
          <div className="review-grid"><span><b>Pass rate</b>{detail.data.pass_rate == null ? "Unscored" : `${Math.round(detail.data.pass_rate * 100)}%`}</span><span><b>Samples</b>{detail.data.sample_count}</span><span><b>Mean latency</b>{formatDuration(detail.data.mean_latency_ms)}</span><span><b>Decode</b>{detail.data.mean_tokens_per_second ? `${detail.data.mean_tokens_per_second.toFixed(1)} tok/s` : "—"}</span></div>
          {detail.data.telemetry.length > 1 && <div className="chart-wrap" aria-label="GPU telemetry"><ResponsiveContainer width="100%" height={190}><AreaChart data={detail.data.telemetry}><defs><linearGradient id="memory" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#67a7ff" stopOpacity={0.45}/><stop offset="95%" stopColor="#67a7ff" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#202a38" vertical={false}/><XAxis dataKey="timestamp" hide/><YAxis stroke="#718198" fontSize={11}/><Tooltip contentStyle={{ background: "#111925", border: "1px solid #2a394d" }}/><Area type="monotone" dataKey="memory_used_mib" stroke="#67a7ff" fill="url(#memory)" name="Memory MiB" /></AreaChart></ResponsiveContainer></div>}
          <div className="case-list">{detail.data.cases.map((item) => <div key={item.case_id}><span><BarChart3 size={15} />{item.case_id}</span><b>{item.pass_rate == null ? `${item.mean_tokens_per_second?.toFixed(1) ?? "—"} tok/s` : `${Math.round(item.pass_rate * 100)}%`}</b></div>)}</div>
          <details className="evidence"><summary><Gauge size={14} /> Reproducibility identity</summary><dl><dt>Suite SHA-256</dt><dd>{detail.data.suite_sha256}</dd><dt>Request contract</dt><dd>{detail.data.request_contract_sha256}</dd></dl></details>
        </div>}
      </Modal>
    </section>
  );
}

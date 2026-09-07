import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Box, Check, Cpu, Gauge, Play, RefreshCw, ShieldCheck, Sparkles, Square, TriangleAlert } from "lucide-react";
import { consoleApi } from "../api/client";
import { createClientId } from "../api/id";
import type { DeploymentSummary } from "../api/types";
import { formatBytes, shortHash } from "../api/format";
import { Modal } from "../components/Modal";
import { StatusBadge } from "../components/StatusBadge";
import { consoleKeys, usePortfolio, useRuntime } from "../hooks/useConsoleData";

export function ModelsPage() {
  const portfolio = usePortfolio();
  const runtime = useRuntime();
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<DeploymentSummary | null>(null);
  const [operationId, setOperationId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [activationKey, setActivationKey] = useState<string | null>(null);
  const [stopKey, setStopKey] = useState<string | null>(null);
  const operation = useQuery({
    queryKey: ["console", "operation", operationId],
    queryFn: () => consoleApi.operation(operationId!),
    enabled: Boolean(operationId),
    refetchInterval: ({ state }) => ["succeeded", "failed", "cancelled", "interrupted"].includes(state.data?.state ?? "") ? false : 750,
  });
  useEffect(() => {
    if (operation.data?.state === "succeeded") {
      void queryClient.invalidateQueries({ queryKey: consoleKeys.runtime });
      void queryClient.invalidateQueries({ queryKey: consoleKeys.portfolio });
      setSelected(null);
      setOperationId(null);
    }
  }, [operation.data?.state, queryClient]);

  const operationTerminal = ["succeeded", "failed", "cancelled", "interrupted"].includes(operation.data?.state ?? "");
  const operationPending = submitting || Boolean(operationId && !operationTerminal);
  const operationStatusUnavailable = Boolean(operationId && operation.isError);
  const modalCanClose = !operationPending || operationStatusUnavailable;

  function review(deployment: DeploymentSummary) {
    setActionError(null);
    setOperationId(null);
    setActivationKey(createClientId());
    setSelected(deployment);
  }

  const active = useMemo(() => portfolio.data?.models.flatMap((model) => model.deployments).find((deployment) => deployment.active), [portfolio.data]);

  async function activate() {
    if (!selected || !portfolio.data || !runtime.data || submitting) return;
    setActionError(null);
    setSubmitting(true);
    const idempotencyKey = activationKey ?? createClientId();
    setActivationKey(idempotencyKey);
    try {
      const created = await consoleApi.activate(selected.id, portfolio.data.catalog_revision, runtime.data.revision, idempotencyKey);
      setOperationId(created.id);
      setActivationKey(null);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Activation could not be started.");
    } finally {
      setSubmitting(false);
    }
  }

  async function stop() {
    if (!runtime.data || submitting) return;
    setActionError(null);
    setSubmitting(true);
    const idempotencyKey = stopKey ?? createClientId();
    setStopKey(idempotencyKey);
    try {
      const created = await consoleApi.stop(runtime.data.revision, idempotencyKey);
      setOperationId(created.id);
      setStopKey(null);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Stop could not be started.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="page workspace-wide">
      <div className="page-head">
        <div><span className="eyebrow">CATALOG · DEPLOYMENT PROFILES</span><h1>Models</h1><p>Choose a reviewed serving profile. Switching changes the active model for every connected client.</p></div>
        {active && <button className="danger-button" disabled={operationPending} onClick={() => void stop()}><Square size={14} /> Stop active</button>}
      </div>
      {actionError && !selected && <div className="inline-error" role="alert">{actionError}</div>}
      {operationStatusUnavailable && !selected && <div className="inline-error" role="alert">
        Operation status is temporarily unavailable. The operation may still be running.
        <button onClick={() => void operation.refetch()}>Retry status</button>
        <button onClick={() => setOperationId(null)}>Dismiss status</button>
      </div>}
      {!selected && operation.data && <div className={`operation-banner ${operation.data.state}`} role={operation.data.state === "failed" ? "alert" : "status"} aria-live="polite">
        {operation.data.state === "failed" ? <TriangleAlert size={17} /> : <RefreshCw className="spin" size={17} />}
        <div><strong>{operation.data.state.replaceAll("_", " ")}</strong><span>{operation.data.message}</span></div>
        {operationTerminal && <button className="ghost-button" onClick={() => setOperationId(null)}>Dismiss</button>}
      </div>}
      {portfolio.isPending ? <div className="card-grid"><div className="skeleton tall" /><div className="skeleton tall" /></div> : portfolio.isError ? (
        <div className="error-panel"><ShieldCheck size={24} /><div><strong>Catalog API unavailable</strong><p>{portfolio.error.message}</p></div><button onClick={() => void portfolio.refetch()}><RefreshCw size={15} /> Retry</button></div>
      ) : <div className="model-grid">{portfolio.data?.models.map((model) => (
        <article className="model-card" data-testid={`model-${model.id}`} key={model.id}>
          <header>
            <span className="model-glyph"><Cpu size={21} /></span>
            <div><h2>{model.display_name}</h2><span>{model.total_params_b}B total{model.active_params_b && model.active_params_b !== model.total_params_b ? ` · ${model.active_params_b}B active` : ""}</span></div>
            <span className="license-chip">{model.license.name}</span>
          </header>
          <p>{model.description}</p>
          <div className="capability-row">{model.capabilities.slice(0, 6).map((capability) => <span key={capability}>{capability.replaceAll("_", " ")}</span>)}</div>
          <div className="deployment-list">{model.deployments.map((deployment) => (
            <div className={deployment.active ? "deployment-row active" : "deployment-row"} key={deployment.id}>
              <div className="deployment-main">
                <strong>{deployment.public_alias}</strong>
                <span>{deployment.reasoning_mode === "off" ? "Standard" : `Reasoning ${deployment.reasoning_mode}`} · {Math.round(deployment.context_size / 1024)}K · {deployment.artifact.quantization}</span>
              </div>
              <div className="deployment-size"><b>{formatBytes(deployment.artifact.size_bytes)}</b><span>{deployment.artifact.registered ? "Installed" : "Not installed"}</span></div>
              {deployment.active ? <StatusBadge ready={deployment.ready} active phase={deployment.phase} /> : (
                <button className="row-action" disabled={!deployment.artifact.registered} onClick={() => review(deployment)}><Play size={14} /> Activate</button>
              )}
            </div>
          ))}</div>
          <footer><span><Box size={14} /> {model.modalities.join(" + ")}</span><span><Gauge size={14} /> Native {Math.round(model.native_context / 1024)}K</span></footer>
        </article>
      ))}</div>}

      <Modal open={Boolean(selected)} title={`Activate ${selected?.public_alias ?? "deployment"}`} onClose={() => modalCanClose && setSelected(null)}>
        {selected && <div className="activation-review">
          <div className="transition-line"><span>{runtime.data?.public_alias ?? "No model"}</span><span>→</span><strong>{selected.public_alias}</strong></div>
          <div className="review-grid">
            <span><b>Context</b>{Math.round(selected.context_size / 1024)}K tokens</span>
            <span><b>Artifact</b>{formatBytes(selected.artifact.size_bytes)}</span>
            <span><b>Cache</b>{selected.kv_cache}</span>
            <span><b>Manifest</b><code>{shortHash(selected.artifact.manifest_sha256)}</code></span>
          </div>
          <p className="modal-note">This switch affects every client and may interrupt active generations. LLM Lab verifies the artifact and runtime before stopping the current deployment, then restores it if readiness fails.</p>
          {operation.data && <div className={`operation-banner ${operation.data.state}`} role={operation.data.state === "failed" ? "alert" : "status"} aria-live="polite">
            {operation.data.state === "succeeded" ? <Check size={17} /> : operation.data.state === "failed" ? <TriangleAlert size={17} /> : <RefreshCw className="spin" size={17} />}
            <div><strong>{operation.data.state.replaceAll("_", " ")}</strong><span>{operation.data.message}</span></div>
          </div>}
          {actionError && <div className="inline-error" role="alert">{actionError}</div>}
          {operationStatusUnavailable && <div className="inline-error" role="alert">
            Operation status is temporarily unavailable. The operation may still be running.
            <button onClick={() => void operation.refetch()}>Retry status</button>
          </div>}
          {operation.data?.error && <div className="inline-error" role="alert">{operation.data.error.message}</div>}
          <div className="modal-actions"><button className="ghost-button" disabled={!modalCanClose} onClick={() => setSelected(null)}>Cancel</button><button className="primary-button" disabled={operationPending} onClick={() => void activate()}><Sparkles size={15} /> {operation.data?.state === "failed" ? "Retry activation" : "Verify and activate"}</button></div>
        </div>}
      </Modal>
    </section>
  );
}

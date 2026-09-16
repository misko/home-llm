export type Phase = "inactive" | "starting" | "ready" | "failed" | "stopping";

export interface LicenseSummary {
  name: string;
  commercial_use: boolean;
  osi_approved: boolean;
  acceptance_required: boolean;
  url?: string | null;
}

export interface ArtifactSummary {
  id: string;
  format: string;
  quantization?: string | null;
  size_bytes: number;
  registered: boolean;
  verified?: boolean | null;
  manifest_sha256?: string | null;
}

export interface DeploymentSummary {
  id: string;
  public_alias: string;
  backend: string;
  context_size: number;
  reasoning_mode: string;
  parallel: number;
  kv_cache: string;
  active: boolean;
  ready: boolean;
  phase: Phase;
  artifact: ArtifactSummary;
}

export interface ModelSummary {
  id: string;
  display_name: string;
  family: string;
  description: string;
  total_params_b?: number | null;
  active_params_b?: number | null;
  native_context: number;
  modalities: string[];
  capabilities: string[];
  license: LicenseSummary;
  deployments: DeploymentSummary[];
}

export interface ModelPortfolio {
  schema_version: number;
  catalog_revision: string;
  models: ModelSummary[];
}

export interface RuntimeStatus {
  schema_version: number;
  revision: string;
  active: boolean;
  ready: boolean;
  running: boolean;
  healthy: boolean;
  phase: Phase;
  deployment_id?: string | null;
  public_alias?: string | null;
  model_name?: string | null;
  activated_at?: string | null;
  context_size?: number | null;
  memory_used_mib?: number | null;
  memory_total_mib?: number | null;
  gpu_utilization_percent?: number | null;
  operation?: OperationSummary | null;
}

export interface StorageArtifact {
  id: string;
  model_id: string;
  format: string;
  quantization?: string | null;
  logical_bytes: number;
  registered: boolean;
  active: boolean;
  manifest_sha256?: string | null;
}

export interface StorageStatus {
  schema_version: number;
  total_bytes: number;
  free_bytes: number;
  used_bytes: number;
  reserve_bytes: number;
  logical_bytes: number;
  unique_blob_bytes: number;
  artifact_count: number;
  artifacts: StorageArtifact[];
}

export interface BenchmarkRun {
  run_id: string;
  suite_id: string;
  suite_version: string;
  deployment_id: string;
  model_id?: string | null;
  status: string;
  started_at?: string | null;
  finished_at?: string | null;
  pass_rate?: number | null;
  sample_count: number;
  error_count: number;
  warmup_error_count: number;
  mean_latency_ms?: number | null;
  mean_tokens_per_second?: number | null;
  peak_memory_mib?: number | null;
}

export interface BenchmarkRuns {
  schema_version: number;
  runs: BenchmarkRun[];
}

export interface BenchmarkCaseResult {
  case_id: string;
  pass_rate?: number | null;
  sample_count: number;
  error_count: number;
  mean_latency_ms?: number | null;
  mean_tokens_per_second?: number | null;
}

export interface BenchmarkRunDetail extends BenchmarkRun {
  suite_sha256?: string | null;
  request_contract_sha256?: string | null;
  cases: BenchmarkCaseResult[];
  telemetry: Array<{
    timestamp: string;
    memory_used_mib?: number | null;
    gpu_utilization_percent?: number | null;
    power_draw_w?: number | null;
  }>;
}

export type OperationState =
  | "queued"
  | "validating"
  | "switching"
  | "waiting_ready"
  | "rolling_back"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "cancelling"
  | "interrupted";

export interface OperationSummary {
  id: string;
  kind: "activate" | "stop" | "benchmark" | "verify";
  state: OperationState;
  requested_deployment_id?: string | null;
  previous_deployment_id?: string | null;
  created_at: string;
  updated_at: string;
  progress?: number | null;
  message?: string | null;
  error?: { code: string; message: string; retryable: boolean } | null;
  result?: Record<string, unknown> | null;
}

export interface SystemStatus {
  schema_version: number;
  gateway_version: string;
  catalog_valid: boolean;
  catalog_revision: string;
  runtime_lock_id?: string | null;
  runtime_sha256?: string | null;
  runtime_version?: string | null;
  gpu_name?: string | null;
  driver_version?: string | null;
  data_root_label: string;
}

export interface ConsoleEvent {
  id: string;
  type: "snapshot" | "runtime.changed" | "operation.changed";
  runtime?: RuntimeStatus;
  operation?: OperationSummary;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  deployment_id?: string;
  created_at: string;
  tool_calls?: ToolCall[];
  tool_executions?: AgentToolExecution[];
  sources?: AgentSource[];
  attachments?: ChatAttachment[];
}

export interface ChatAttachment {
  id: string;
  name: string;
  mime_type: string;
  size: number;
  data_url: string;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export type AgentToolStatus = "running" | "completed" | "blocked" | "failed";

export interface AgentToolError {
  code?: string;
  message: string;
  retryable?: boolean;
}

export interface AgentToolExecution {
  id: string;
  name: string;
  arguments?: unknown;
  status: AgentToolStatus;
  result?: unknown;
  error?: AgentToolError;
}

export interface AgentSource {
  title: string;
  url: string;
  snippet?: string;
}

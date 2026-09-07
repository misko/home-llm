import type { ModelPortfolio, RuntimeStatus } from "../api/types";

export function runtimeFixture(overrides: Partial<RuntimeStatus> = {}): RuntimeStatus {
  return {
    schema_version: 1,
    revision: "rt1-local-fast",
    active: true,
    ready: true,
    running: true,
    healthy: true,
    phase: "ready",
    deployment_id: "ling-3.0-tiny-4090-8k",
    public_alias: "local-fast",
    model_name: "Ling 3.0 Tiny",
    activated_at: "2026-09-06T20:00:00Z",
    context_size: 8192,
    memory_used_mib: 6144,
    memory_total_mib: 24564,
    gpu_utilization_percent: 4,
    operation: null,
    ...overrides,
  };
}

export function portfolioFixture(museActive = false): ModelPortfolio {
  return {
    schema_version: 1,
    catalog_revision: "cat1-console-e2e",
    models: [
      {
        id: "ling-3.0-tiny",
        display_name: "Ling 3.0 Tiny",
        family: "ling-3.0",
        description: "Fast local reasoning and tool use.",
        total_params_b: 7.9,
        active_params_b: 1.3,
        native_context: 262144,
        modalities: ["text"],
        capabilities: ["chat", "reasoning", "tools"],
        license: {
          name: "MIT",
          commercial_use: true,
          osi_approved: true,
          acceptance_required: false,
        },
        deployments: [
          {
            id: "ling-3.0-tiny-4090-8k",
            public_alias: "local-fast",
            backend: "llama_cpp",
            context_size: 8192,
            reasoning_mode: "off",
            parallel: 1,
            kv_cache: "q8_0/q8_0",
            active: !museActive,
            ready: !museActive,
            phase: museActive ? "inactive" : "ready",
            artifact: {
              id: "ling-3.0-tiny-q4-k-m",
              format: "gguf",
              quantization: "Q4_K_M",
              size_bytes: 4_823_895_906,
              registered: true,
              manifest_sha256: "a".repeat(64),
            },
          },
        ],
      },
      {
        id: "muse-glimmer-30b",
        display_name: "Muse Glimmer 30B",
        family: "muse-glimmer",
        description: "Multimodal agent and tool-use model for a 24 GB GPU.",
        total_params_b: 29.6,
        active_params_b: 29.6,
        native_context: 131072,
        modalities: ["text", "image"],
        capabilities: ["chat", "reasoning", "tools", "vision"],
        license: {
          name: "Apache-2.0",
          commercial_use: true,
          osi_approved: true,
          acceptance_required: false,
        },
        deployments: [
          {
            id: "muse-glimmer-30b-4090-8k",
            public_alias: "local-agent",
            backend: "llama_cpp",
            context_size: 8192,
            reasoning_mode: "auto",
            parallel: 1,
            kv_cache: "q8_0/q8_0",
            active: museActive,
            ready: museActive,
            phase: museActive ? "ready" : "inactive",
            artifact: {
              id: "muse-glimmer-30b-kquant",
              format: "gguf",
              quantization: "KQuant-17GB-Q4_K_M",
              size_bytes: 18_157_050_004,
              registered: true,
              manifest_sha256: "b".repeat(64),
            },
          },
        ],
      },
    ],
  };
}

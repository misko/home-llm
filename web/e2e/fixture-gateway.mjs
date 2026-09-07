import { createServer } from "node:http";

const runtime = {
  schema_version: 1,
  revision: "rt1-proxy-fixture",
  active: true,
  ready: true,
  running: true,
  healthy: true,
  phase: "ready",
  deployment_id: "ling-3.0-tiny-4090-8k",
  public_alias: "local-fast",
  model_name: "Ling 3.0 Tiny",
  context_size: 8192,
  operation: null,
};

const portfolio = {
  schema_version: 1,
  catalog_revision: "cat1-proxy-fixture",
  models: [{
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
    deployments: [{
      id: "muse-glimmer-30b-4090-8k",
      public_alias: "local-agent",
      backend: "llama_cpp",
      context_size: 8192,
      reasoning_mode: "auto",
      parallel: 1,
      kv_cache: "q8_0/q8_0",
      active: false,
      ready: false,
      phase: "inactive",
      artifact: {
        id: "muse-glimmer-30b-kquant",
        format: "gguf",
        quantization: "KQuant-17GB-Q4_K_M",
        size_bytes: 18157050004,
        registered: true,
        manifest_sha256: "b".repeat(64),
      },
    }],
  }],
};

function json(response, status, document) {
  const body = JSON.stringify(document);
  response.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  response.end(body);
}

const server = createServer((request, response) => {
  if (request.url === "/health") {
    json(response, 200, { status: "ok" });
  } else if (request.url === "/api/v1/runtime") {
    json(response, 200, runtime);
  } else if (request.url === "/api/v1/models") {
    json(response, 200, portfolio);
  } else if (request.url === "/api/v1/events") {
    response.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    });
    response.end("retry: 60000\n\n");
  } else {
    json(response, 404, { detail: "Not Found" });
  }
});

server.listen(14100, "127.0.0.1");
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}

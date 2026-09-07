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
    deployments: [{
      id: "ling-3.0-tiny-4090-8k",
      public_alias: "local-fast",
      backend: "llama_cpp",
      context_size: 8192,
      reasoning_mode: "off",
      parallel: 1,
      kv_cache: "q8_0/q8_0",
      active: true,
      ready: true,
      phase: "ready",
      artifact: {
        id: "ling-3.0-tiny-q4-k-m",
        format: "gguf",
        quantization: "Q4_K_M",
        size_bytes: 4823895906,
        registered: true,
        manifest_sha256: "a".repeat(64),
      },
    }],
  }, {
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
  } else if (request.url === "/api/v1/agent/turns" && request.method === "POST") {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      const turn = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      if (turn.toolset !== "standard-readonly") {
        json(response, 400, { error: { code: "invalid_toolset", message: "Expected the read-only research toolset." } });
        return;
      }
      if (turn.instructions !== "Use concise language and cite sources.") {
        json(response, 400, { error: { code: "invalid_instructions", message: "Expected top-level research instructions." } });
        return;
      }
      if (turn.messages?.some((message) => message.role === "system")) {
        json(response, 400, { error: { code: "system_message_forbidden", message: "Instructions must not be encoded as a system message." } });
        return;
      }
      response.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      });
      const events = [
        {
          schema_version: 1,
          type: "turn.started",
          run_id: "run-search-fixture",
          sequence: 1,
          model: "local-fast",
          toolset: "standard-readonly",
          tools: ["web_search", "web_fetch", "calculator", "current_time"],
        },
        {
          schema_version: 1,
          type: "tool.started",
          run_id: "run-search-fixture",
          sequence: 2,
          call_id: "search-fixture",
          name: "web_search",
          arguments: { query: "best local LLM tools" },
          round: 1,
        },
        {
          schema_version: 1,
          type: "tool.completed",
          run_id: "run-search-fixture",
          sequence: 3,
          call_id: "search-fixture",
          name: "web_search",
          result: {
            query: "best local LLM tools",
            results: [{
              title: "SearXNG search documentation",
              url: "https://docs.searxng.org/dev/search_api.html",
              snippet: "SearXNG provides a JSON search API suitable for a private research tool.",
            }],
          },
          round: 1,
          duration_ms: 12.5,
        },
        {
          schema_version: 1,
          type: "tool.started",
          run_id: "run-search-fixture",
          sequence: 4,
          call_id: "calculator-fixture",
          name: "calculator",
          arguments: { expression: "2 + 2" },
          round: 1,
        },
        {
          schema_version: 1,
          type: "tool.completed",
          run_id: "run-search-fixture",
          sequence: 5,
          call_id: "calculator-fixture",
          name: "calculator",
          result: { expression: "2 + 2", result: 4 },
          round: 1,
          duration_ms: 0.2,
        },
        {
          schema_version: 1,
          type: "assistant.delta",
          run_id: "run-search-fixture",
          sequence: 6,
          content: "SearXNG is a strong self-hosted search option. ",
          round: 2,
        },
        {
          schema_version: 1,
          type: "assistant.delta",
          run_id: "run-search-fixture",
          sequence: 7,
          content: "I found it through the read-only research tool.",
          round: 2,
        },
        {
          schema_version: 1,
          type: "turn.completed",
          run_id: "run-search-fixture",
          sequence: 8,
          model: "local-fast",
          rounds: 2,
          finish_reason: "stop",
          usage: { prompt_tokens: 27, completion_tokens: 18, total_tokens: 45 },
        },
      ];
      let index = 0;
      const writeNext = () => {
        if (index === events.length) {
          response.end();
          return;
        }
        const event = events[index++];
        response.write(`id: ${event.sequence}\nevent: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
        setTimeout(writeNext, 15);
      };
      writeNext();
    });
  } else {
    json(response, 404, { detail: "Not Found" });
  }
});

server.listen(14100, "127.0.0.1");
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}

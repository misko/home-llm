import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const gatewayTarget = process.env.LLM_LAB_GATEWAY_URL ?? "http://127.0.0.1:14000";

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  build: {
    outDir: "../src/llm_lab/web_dist",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    allowedHosts: ["kalman"],
    proxy: {
      "/api": gatewayTarget,
      "/health": gatewayTarget,
      "/v1": gatewayTarget,
    },
  },
});

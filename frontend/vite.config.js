import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend; in Docker the same path is
// proxied by nginx, so no base-url configuration is needed at runtime.
const API_TARGET = process.env.VITE_API_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: Number(process.env.PORT || 5173),
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: Number(process.env.PORT || 4173),
    proxy: { "/api": { target: API_TARGET, changeOrigin: true } },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The UI dev server proxies to the Python API (main.py serve, default :8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});

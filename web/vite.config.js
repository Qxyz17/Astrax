import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /v1 to the local llama-server so the browser never
// needs to deal with CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/v1": "http://127.0.0.1:8080",
    },
  },
});

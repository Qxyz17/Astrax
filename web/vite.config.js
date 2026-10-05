import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Strip HTML to plain text so a fetched page can be given to the model as
// context. No external dependency: a small regex-based cleaner is enough for
// the pages users paste.
function htmlToText(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function fetchUrlMiddleware() {
  return {
    name: "astrax-fetch-url",
    configureServer(server) {
      server.middlewares.use("/api/fetch-url", async (req, res) => {
        try {
          const requestUrl = new URL(req.url, "http://localhost");
          const target = requestUrl.searchParams.get("url");
          if (!target || !/^https?:\/\//i.test(target)) {
            res.statusCode = 400;
            res.end("invalid url");
            return;
          }
          const response = await fetch(target, {
            headers: { "User-Agent": "Mozilla/5.0 AstraxReader" },
            signal: AbortSignal.timeout(20000),
          });
          const body = await response.text();
          const text = htmlToText(body).slice(0, 8000);
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.end(JSON.stringify({ url: target, text }));
        } catch (error) {
          res.statusCode = 502;
          res.end("fetch failed: " + error.message);
        }
      });
    },
  };
}

// The dev server proxies /v1 to the local llama-server and adds a URL fetch
// endpoint so pasted links can be read as context.
export default defineConfig({
  plugins: [react(), fetchUrlMiddleware()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/v1": "http://127.0.0.1:8080",
    },
  },
});

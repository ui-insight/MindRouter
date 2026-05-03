#!/usr/bin/env node
// MindRouter MCP Bridge — SSE-to-stdio for Claude Desktop
//
// Connects to MindRouter's hosted MCP SSE server and bridges it to
// stdio transport so Claude Desktop can use it as a local MCP server.
// No npm dependencies — uses Node.js built-in fetch API.

const SSE_URL = process.env.SSE_URL || "https://mindrouter.uidaho.edu/mcp/sse";
const API_KEY = process.env.API_KEY || "";

if (!API_KEY) {
  console.error("[mindrouter] No API key configured. Set your MindRouter API key in the extension settings.");
  process.exit(1);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  const sseResp = await fetch(SSE_URL, {
    headers: { Authorization: `Bearer ${API_KEY}`, Accept: "text/event-stream" },
  });

  if (!sseResp.ok) {
    console.error(`[mindrouter] SSE connect failed: ${sseResp.status}`);
    process.exit(1);
  }

  const reader = sseResp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let messageEndpoint = null;

  const sseTask = (async () => {
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const parts = buf.split("\n");
        buf = parts.pop() || "";

        let eventType = null;

        for (const line of parts) {
          const trimmed = line.trim();
          if (!trimmed) { eventType = null; continue; }

          if (trimmed.startsWith("event:")) {
            eventType = trimmed.slice(6).trim();
          } else if (trimmed.startsWith("data:")) {
            const data = trimmed.slice(5).trim();

            if (eventType === "endpoint" && data) {
              try { messageEndpoint = new URL(data, SSE_URL).toString(); }
              catch { messageEndpoint = data; }
            } else if (eventType === "message" && data.startsWith("{")) {
              process.stdout.write(data + "\n");
            } else if (!messageEndpoint && data.startsWith("http")) {
              try { messageEndpoint = new URL(data, SSE_URL).toString(); }
              catch { messageEndpoint = data; }
            }
          }
        }
      }
    } catch (e) {
      console.error(`[mindrouter] SSE error: ${e.message}`);
    }
  })();

  let tries = 0;
  while (!messageEndpoint && tries++ < 100) await sleep(100);
  if (!messageEndpoint) {
    console.error("[mindrouter] Timeout waiting for server endpoint");
    process.exit(1);
  }

  console.error("[mindrouter] Connected");

  process.stdin.setEncoding("utf-8");
  let stdinBuf = "";

  process.stdin.on("data", async (chunk) => {
    stdinBuf += chunk;
    const lines = stdinBuf.split("\n");
    stdinBuf = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;

      try {
        const resp = await fetch(messageEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${API_KEY}`,
          },
          body: line,
        });
        if (!resp.ok) {
          console.error(`[mindrouter] POST ${resp.status}: ${await resp.text()}`);
        }
      } catch (e) {
        console.error(`[mindrouter] POST error: ${e.message}`);
      }
    }
  });

  process.on("SIGTERM", () => { reader.cancel(); process.exit(0); });
  process.on("SIGINT", () => { reader.cancel(); process.exit(0); });

  await sseTask;
}

main().catch(e => {
  console.error(`[mindrouter] Fatal: ${e.message}`);
  process.exit(1);
});

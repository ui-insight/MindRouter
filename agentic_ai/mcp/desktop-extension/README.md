# MindRouter Desktop Extension

Claude Desktop extension (`.mcpb`) that connects Claude to MindRouter's hosted MCP server.

## Install

1. Download [`mindrouter.mcpb`](../mindrouter.mcpb)
2. Double-click the file (or drag it into Claude Desktop)
3. Enter your MindRouter API key when prompted

Generate an API key at [mindrouter.uidaho.edu](https://mindrouter.uidaho.edu) under **API Keys**.

## What it does

Bridges Claude Desktop to MindRouter's SSE MCP server via stdio transport. Provides access to MindRouter tools (e.g., `web_search`) without any local dependencies beyond Claude Desktop itself.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| **API Key** | *(required)* | Your MindRouter API key (`mr2_...`) |
| **Server URL** | `https://mindrouter.uidaho.edu/mcp/sse` | MCP SSE endpoint. Change if you run your own MindRouter instance. |

## Rebuilding

From this directory:

```bash
zip -r ../mindrouter.mcpb manifest.json server/
```

## How it works

The extension runs a lightweight Node.js script that:
1. Opens an SSE connection to the MindRouter MCP server
2. Receives the session endpoint URL
3. Reads JSON-RPC messages from stdin (Claude Desktop)
4. POSTs them to the MCP server
5. Writes server responses back to stdout

No npm dependencies — uses Node.js built-in `fetch` API. Claude Desktop provides the Node.js runtime.

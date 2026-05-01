# MindRouter MCP Servers

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers that expose MindRouter APIs as tools for agentic systems like Claude Code, CoWork, Cursor, and other MCP-compatible clients.

## Connection Options

### Hosted SSE Server (Recommended)

MindRouter hosts a built-in MCP server — no local Python, no dependencies, no processes to manage. Just configure your MCP client:

```json
{
  "mcpServers": {
    "mindrouter-search": {
      "type": "sse",
      "url": "https://mindrouter.uidaho.edu/mcp/search/sse",
      "headers": {
        "Authorization": "Bearer mr2_your_key_here"
      }
    }
  }
}
```

### Local stdio Server

Run a local MCP server that forwards requests to MindRouter over HTTPS. Use this if you prefer self-hosted tooling.

#### 1. Install dependencies

```bash
pip install "mcp[cli]" httpx
```

#### 2. Configure your API key

Get a MindRouter API key from the [dashboard](https://mindrouter.uidaho.edu/dashboard/api-keys).

```bash
export MINDROUTER_API_KEY=mr2_your_key_here
```

#### 3. Add to your MCP client

```bash
claude mcp add mindrouter-search -- python3 /path/to/agentic_ai/mcp/search/server.py
```

Or add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "mindrouter-search": {
      "command": "python3",
      "args": ["/path/to/agentic_ai/mcp/search/server.py"],
      "env": {
        "MINDROUTER_API_KEY": "mr2_your_key_here",
        "MINDROUTER_BASE_URL": "https://mindrouter.uidaho.edu"
      }
    }
  }
}
```

#### 4. Verify

Inside Claude Code, run `/mcp` to confirm the server is connected and the `web_search` tool is available.

## Available Tools

| Tool | MCP Name | Description |
|------|----------|-------------|
| **Web Search** | `web_search` | Search the web via MindRouter (Brave Search / SearXNG) |

## Environment Variables (Local Server Only)

| Variable | Default | Description |
|----------|---------|-------------|
| `MINDROUTER_API_KEY` | *(required)* | Your MindRouter API key |
| `MINDROUTER_BASE_URL` | `https://mindrouter.uidaho.edu` | MindRouter instance URL |

## Architecture

```
Agent (Claude Code, CoWork, Cursor, etc.)
  ├── SSE transport  ──► MindRouter SSE MCP Server (hosted) ──► Search backend
  └── stdio transport ──► MCP Server (local) ──► HTTPS ──► MindRouter /v1/search
```

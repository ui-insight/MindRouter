# MindRouter MCP Servers

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers that expose MindRouter APIs as tools for agentic systems like Claude Code, CoWork, Cursor, and other MCP-compatible clients.

## Available Servers

| Server | Directory | Description |
|--------|-----------|-------------|
| **Web Search** | `search/` | Search the web via MindRouter's search API (Brave Search) |

## Quick Start

### 1. Install dependencies

```bash
cd mcp/search
pip install -r requirements.txt
```

### 2. Configure your API key

Get a MindRouter API key from the [dashboard](https://mindrouter.uidaho.edu/dashboard/api-keys).

```bash
export MINDROUTER_API_KEY=mr2_your_key_here
```

### 3. Add to Claude Code

```bash
claude mcp add mindrouter-search -- python3 /path/to/mcp/search/server.py
```

Or add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "mindrouter-search": {
      "command": "python3",
      "args": ["/path/to/mcp/search/server.py"],
      "env": {
        "MINDROUTER_API_KEY": "mr2_your_key_here",
        "MINDROUTER_BASE_URL": "https://mindrouter.uidaho.edu"
      }
    }
  }
}
```

### 4. Verify

Inside Claude Code, run `/mcp` to confirm the server is connected and the `web_search` tool is available.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINDROUTER_API_KEY` | *(required)* | Your MindRouter API key |
| `MINDROUTER_BASE_URL` | `https://mindrouter.uidaho.edu` | MindRouter instance URL |

## How It Works

MCP servers run locally on your machine. When an agent needs to search the web, it calls the `web_search` tool via MCP, which forwards the query to MindRouter's `/v1/search` API over HTTPS and returns the results.

```
Agent (Claude Code, CoWork, etc.)
  └── MCP protocol (stdio) ──► MCP Server (local)
                                  └── HTTPS ──► MindRouter /v1/search
                                                  └── Brave Search API
```

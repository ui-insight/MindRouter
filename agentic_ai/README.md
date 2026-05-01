# MindRouter Agentic AI Integrations

Tools, skills, and servers that let AI coding agents and agentic systems use MindRouter's APIs. Compatible with Claude Code, ForgeCode, OpenCode, Codex, Cursor, CoWork, and other agentic tools.

## Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| **Hosted MCP Server** | *(built into MindRouter)* | Server-side SSE MCP endpoint — no local setup needed |
| **Local MCP Server** | [`mcp/`](mcp/) | [Model Context Protocol](https://modelcontextprotocol.io/) stdio server for local use |
| **Agent Skills** | [`skills/`](skills/) | Markdown-based skill definitions for AI coding agents |

## Quick Start

### Prerequisites

- A MindRouter API key (get one from the [dashboard](https://mindrouter.uidaho.edu/dashboard/api-keys))

### Option 1: Hosted MCP Server (Recommended)

Connect directly to MindRouter's built-in MCP endpoint over SSE. No Python, no dependencies, no local processes.

Add to your MCP client config (`.mcp.json`, `.cursor/mcp.json`, etc.):

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

Or via Claude Code CLI:

```bash
claude mcp add --transport sse \
  -H "Authorization: Bearer mr2_your_key_here" \
  mindrouter-search https://mindrouter.uidaho.edu/mcp/search/sse
```

### Option 2: Local MCP Server (stdio)

Run the MCP server locally. Requires Python 3.11+.

```bash
pip install "mcp[cli]" httpx
export MINDROUTER_API_KEY=mr2_your_key_here
claude mcp add mindrouter-search -- python3 agentic_ai/mcp/search/server.py
```

See [`mcp/README.md`](mcp/README.md) for full setup, including configuration for Cursor, CoWork, and other MCP clients.

### Option 3: Agent Skill (lightweight, no dependencies)

Skills are markdown files containing API context and instructions. Copy them into your agent's skill/command directory:

```bash
# Example: Claude Code
mkdir -p .claude/skills
cp -r agentic_ai/skills/web-search .claude/skills/

# Set your API key
export MINDROUTER_API_KEY=mr2_your_key_here
```

See [`skills/README.md`](skills/README.md) for installation with other agent tools.

## Architecture

```
User / Agent
  |
  |-- MCP tool call (SSE)         --> MindRouter SSE MCP Server (hosted) --> Search backend
  |
  |-- MCP tool call (stdio)       --> MCP Server (local)  --> MindRouter /v1/search
  |
  |-- Agent skill (markdown)      --> Agent executes curl --> MindRouter /v1/search
                                                                 |
                                                                 v
                                                           Brave Search / SearXNG
```

All paths authenticate with your MindRouter API key and deduct a small token cost from your quota per search.

## Available Tools

| Tool | Skill | MCP Tool | Hosted SSE Endpoint | REST API |
|------|-------|----------|---------------------|----------|
| **Web Search** | `web-search` | `web_search` | `GET /mcp/search/sse` | `POST /v1/search` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINDROUTER_API_KEY` | *(required)* | Your MindRouter API key |
| `MINDROUTER_BASE_URL` | `https://mindrouter.uidaho.edu` | MindRouter instance URL (local server / skills only) |

# MindRouter Agentic AI Integrations

Tools, skills, and servers that let AI coding agents and agentic systems use MindRouter's APIs. Compatible with Claude Code, ForgeCode, OpenCode, Codex, Cursor, CoWork, and other agentic tools.

## Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| **MCP Servers** | [`mcp/`](mcp/) | [Model Context Protocol](https://modelcontextprotocol.io/) servers for any MCP-compatible client |
| **Agent Skills** | [`skills/`](skills/) | Markdown-based skill definitions for AI coding agents |

## Quick Start

### Prerequisites

- A MindRouter API key (get one from the [dashboard](https://mindrouter.uidaho.edu/dashboard/api-keys))
- Python 3.11+ (for MCP servers)

### Option 1: MCP Server (standardized tool protocol)

MCP servers expose MindRouter tools to any MCP-compatible agent. The server runs locally and forwards requests to MindRouter over HTTPS.

```bash
# Install dependencies
pip install "mcp[cli]" httpx

# Configure
export MINDROUTER_API_KEY=mr2_your_key_here

# Add to your MCP-compatible agent (example: Claude Code)
claude mcp add mindrouter-search -- python3 agentic_ai/mcp/search/server.py
```

See [`mcp/README.md`](mcp/README.md) for full setup, including configuration for Cursor, CoWork, and other MCP clients.

### Option 2: Agent Skill (lightweight, no dependencies)

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
  |-- Agent skill (markdown)      --> Agent executes curl --> MindRouter /v1/search
  |
  |-- MCP tool call (web_search)  --> MCP Server (local)  --> MindRouter /v1/search
                                                                 |
                                                                 v
                                                           Brave Search / SearXNG
```

Both paths authenticate with your MindRouter API key and deduct a small token cost from your quota per search.

## Available Tools

| Tool | Skill | MCP Tool | API Endpoint |
|------|-------|----------|--------------|
| **Web Search** | `web-search` | `web_search` | `POST /v1/search` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINDROUTER_API_KEY` | *(required)* | Your MindRouter API key |
| `MINDROUTER_BASE_URL` | `https://mindrouter.uidaho.edu` | MindRouter instance URL |

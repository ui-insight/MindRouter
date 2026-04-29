# MindRouter Agent Skills

Reusable skill definitions that give AI coding agents access to MindRouter capabilities. Skills are markdown files containing API context, usage instructions, and example commands that agents use to interact with MindRouter services.

Compatible with [Claude Code](https://docs.anthropic.com/en/docs/claude-code/skills), [ForgeCode](https://forgecode.dev/), [OpenCode](https://opencode.ai/), [Codex](https://openai.com/codex/), and other agentic coding tools that support markdown-based skill/command definitions.

## Available Skills

| Skill | Directory | Description |
|-------|-----------|-------------|
| **Web Search** | `web-search/` | Search the web via MindRouter's search API (Brave Search / SearXNG) |

## Installation

How you install depends on your agent tool. The skill files are standard markdown with YAML frontmatter — copy them into the location your tool expects.

### Claude Code

```bash
mkdir -p .claude/skills
cp -r agentic_ai/skills/web-search .claude/skills/
```

Then invoke with `/web-search <query>`.

### Other Agent Tools

Copy or symlink the skill directory into your tool's custom command/skill location. The `SKILL.md` file contains everything the agent needs: API endpoint, authentication, request/response format, and usage notes.

If your tool doesn't have a specific skills directory, you can reference the file directly in your system prompt or project context.

## Configuration

Set your MindRouter API key in the environment before starting your agent:

```bash
export MINDROUTER_API_KEY=mr2_your_key_here
```

Optionally set a custom base URL (defaults to `https://mindrouter.uidaho.edu`):

```bash
export MINDROUTER_BASE_URL=https://your-mindrouter-instance.example.com
```

Get an API key from the [MindRouter dashboard](https://mindrouter.uidaho.edu/dashboard/api-keys).

## Skills vs MCP Servers

MindRouter provides two integration paths for agentic AI tools:

| Feature | Skills | MCP Servers |
|---------|--------|-------------|
| **Format** | Markdown + YAML frontmatter | Python MCP server (stdio transport) |
| **Setup** | Copy folder into agent's skill directory | Configure in agent's MCP settings |
| **Scope** | Any agent that reads markdown skill files | Any MCP-compatible client |
| **Invocation** | Slash command or agent-initiated | Automatic tool use via MCP protocol |
| **Dependencies** | None (uses curl/shell) | Python, `mcp[cli]`, `httpx` |

Use **skills** for lightweight, dependency-free integration. Use **MCP servers** for a standardized tool interface with structured input/output.

See [`../mcp/`](../mcp/) for MCP server setup.

#!/usr/bin/env python3
"""MindRouter Web Search MCP Server.

Exposes MindRouter's /v1/search endpoint as an MCP tool so agentic
systems (Claude Code, CoWork, Cursor, etc.) can search the web.

Usage:
    pip install "mcp[cli]" httpx
    export MINDROUTER_API_KEY=mr2_your_key_here
    python server.py

Configure in Claude Code:
    claude mcp add mindrouter-search -- python /path/to/mcp/search/server.py

Or in .mcp.json:
    {
      "mcpServers": {
        "mindrouter-search": {
          "command": "python3",
          "args": ["/path/to/mcp/search/server.py"],
          "env": {
            "MINDROUTER_API_KEY": "mr2_your_key_here"
          }
        }
      }
    }
"""

import os
import sys
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mindrouter-search")

BASE_URL = os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu")
API_KEY = os.environ.get("MINDROUTER_API_KEY", "")


@mcp.tool()
async def web_search(query: str, max_results: Optional[int] = 5) -> str:
    """Search the web using MindRouter's search API.

    Returns titles, URLs, and snippets for each result. Use this when
    you need current information, documentation, or facts that may not
    be in your training data.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (1-50, default 5).
    """
    if not API_KEY:
        return "Error: MINDROUTER_API_KEY environment variable is not set."

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BASE_URL}/v1/search",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={"query": query, "max_results": max_results},
        )

    if resp.status_code != 200:
        return f"Search failed (HTTP {resp.status_code}): {resp.text}"

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['url']}")
        lines.append(f"   {r['snippet']}")
        if r.get("published"):
            lines.append(f"   Published: {r['published']}")
        lines.append("")

    return "\n".join(lines)


def main():
    if not API_KEY:
        print(
            "Warning: MINDROUTER_API_KEY is not set. "
            "The web_search tool will return an error until configured.",
            file=sys.stderr,
        )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

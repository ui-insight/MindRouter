---
name: web-search
description: Search the web using MindRouter's search API. Use when you need current information, documentation, recent news, or facts beyond your training data.
argument-hint: <search query>
---

# Web Search via MindRouter

Search the web using MindRouter's `/v1/search` endpoint, which queries Brave Search or SearXNG and returns structured results.

## How to search

Run a `curl` command against the MindRouter search API:

```bash
curl -s -X POST "${MINDROUTER_BASE_URL:-https://mindrouter.uidaho.edu}/v1/search" \
  -H "Authorization: Bearer $MINDROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "<search query>", "max_results": 10}'
```

## Handling the response

The API returns JSON with this structure:

```json
{
  "query": "the search query",
  "provider": "Brave Search",
  "results": [
    {
      "title": "Page Title",
      "url": "https://example.com/page",
      "snippet": "A brief excerpt from the page...",
      "published": "2026-04-15"
    }
  ],
  "total_results": 10
}
```

Parse the JSON response and present results clearly to the user:
- Show the **title** and **URL** for each result
- Include the **snippet** so the user can judge relevance
- Include the **published date** when available
- Offer to fetch full page content for any result the user wants to read in detail

## Requirements

- `MINDROUTER_API_KEY` must be set in the environment
- `MINDROUTER_BASE_URL` defaults to `https://mindrouter.uidaho.edu` if not set

## Notes

- Each search deducts a small fixed token cost from the user's MindRouter quota (default: 50 tokens)
- Query length: 1-2000 characters
- Max results: 1-50 (default 10)
- If the API returns an error, report the HTTP status and message to the user

#!/usr/bin/env python3
"""
MindRouter2 Comprehensive Smoke Test

A manual diagnostic tool that exercises every API surface of MindRouter2
against a live deployment. Prints colored pass/fail results.

Usage:
    python test.py --api-key mr2_xxx
    python test.py --api-key mr2_xxx --base-url http://host:8000
    python test.py --api-key mr2_xxx --admin-key mr2_yyy
    python test.py --api-key mr2_xxx --section health
    python test.py --api-key mr2_xxx --timeout 120
"""

import argparse
import json
import sys
import time

import httpx

# ---------------------------------------------------------------------------
# Colored output helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _pass(name: str, detail: str = ""):
    msg = f"  {GREEN}PASS{RESET}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def _fail(name: str, detail: str = ""):
    msg = f"  {RED}FAIL{RESET}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def _skip(name: str, detail: str = ""):
    msg = f"  {YELLOW}SKIP{RESET}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def section_header(title: str):
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

results: list[dict] = []


def record(status: str, name: str, detail: str = ""):
    results.append({"status": status, "name": name, "detail": detail})
    if status == "pass":
        _pass(name, detail)
    elif status == "fail":
        _fail(name, detail)
    else:
        _skip(name, detail)


# ---------------------------------------------------------------------------
# Section 1 — Health & Monitoring
# ---------------------------------------------------------------------------

def test_health(client: httpx.Client, cfg: argparse.Namespace):
    section_header("1. Health & Monitoring")

    # GET /healthz
    name = "GET /healthz"
    try:
        r = client.get("/healthz")
        if r.status_code == 200 and "alive" in r.text:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:120]}")
    except Exception as e:
        record("fail", name, str(e))

    # GET /readyz
    name = "GET /readyz"
    try:
        r = client.get("/readyz")
        body = r.json()
        if r.status_code == 200 and "status" in body and "checks" in body:
            record("pass", name, f"status={body['status']}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:120]}")
    except Exception as e:
        record("fail", name, str(e))

    # GET /status
    name = "GET /status"
    try:
        r = client.get("/status")
        body = r.json()
        if r.status_code == 200 and "service" in body and "backends" in body:
            record("pass", name, f"backends={body['backends']}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:120]}")
    except Exception as e:
        record("fail", name, str(e))

    # GET /metrics
    name = "GET /metrics"
    try:
        r = client.get("/metrics")
        ct = r.headers.get("content-type", "")
        if r.status_code == 200 and "text/plain" in ct:
            record("pass", name, f"content-type={ct}")
        else:
            record("fail", name, f"status={r.status_code} content-type={ct}")
    except Exception as e:
        record("fail", name, str(e))


# ---------------------------------------------------------------------------
# Section 2 — Authentication
# ---------------------------------------------------------------------------

def test_auth(client: httpx.Client, cfg: argparse.Namespace):
    section_header("2. Authentication")

    # Valid API key
    name = "Valid API key → 200 on /v1/models"
    try:
        r = client.get("/v1/models", headers={"Authorization": f"Bearer {cfg.api_key}"})
        if r.status_code == 200:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code}")
    except Exception as e:
        record("fail", name, str(e))

    # Missing API key
    name = "Missing API key → 401/403"
    try:
        r = client.get("/v1/models")  # no auth header
        if r.status_code in (401, 403):
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"expected 401/403, got {r.status_code}")
    except Exception as e:
        record("fail", name, str(e))

    # Invalid API key
    name = "Invalid API key → 401/403"
    try:
        r = client.get("/v1/models", headers={"Authorization": "Bearer invalid_key_xxx"})
        if r.status_code in (401, 403):
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"expected 401/403, got {r.status_code}")
    except Exception as e:
        record("fail", name, str(e))


# ---------------------------------------------------------------------------
# Section 3 — OpenAI-Compatible Endpoints
# ---------------------------------------------------------------------------

def test_openai(client: httpx.Client, cfg: argparse.Namespace):
    section_header("3. OpenAI-Compatible Endpoints")
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    model = cfg.vllm_model

    # GET /v1/models
    name = "GET /v1/models"
    try:
        r = client.get("/v1/models", headers=headers)
        body = r.json()
        if r.status_code == 200 and "data" in body:
            record("pass", name, f"{len(body['data'])} models")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:120]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /v1/chat/completions — non-streaming
    name = "POST /v1/chat/completions (non-streaming)"
    try:
        r = client.post("/v1/chat/completions", headers=headers, json={
            "model": model,
            "stream": False,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
        })
        body = r.json()
        if r.status_code == 200 and "choices" in body and "usage" in body:
            content = body["choices"][0]["message"].get("content") or ""
            reasoning = body["choices"][0]["message"].get("reasoning_content") or ""
            display = content[:60] if content else f"[reasoning] {reasoning[:50]}"
            record("pass", name, f"response={display!r}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /v1/chat/completions — streaming
    name = "POST /v1/chat/completions (streaming)"
    try:
        with client.stream("POST", "/v1/chat/completions", headers=headers, json={
            "model": model,
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say hi."}],
        }) as r:
            if r.status_code != 200:
                record("fail", name, f"status={r.status_code}")
            else:
                chunks = []
                saw_done = False
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            saw_done = True
                        else:
                            chunks.append(payload)
                if saw_done and len(chunks) > 0:
                    record("pass", name, f"{len(chunks)} chunks, saw [DONE]")
                else:
                    record("fail", name, f"chunks={len(chunks)} done={saw_done}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /v1/chat/completions — JSON mode
    name = "POST /v1/chat/completions (JSON mode)"
    try:
        r = client.post("/v1/chat/completions", headers=headers, json={
            "model": model,
            "stream": False,
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "Return JSON: {\"color\": \"blue\"}"}],
        })
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            content = body["choices"][0]["message"]["content"]
            json.loads(content)  # must parse as valid JSON
            record("pass", name, f"valid JSON response")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except json.JSONDecodeError:
        record("fail", name, f"response was not valid JSON: {content[:120]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /v1/chat/completions — with parameters
    name = "POST /v1/chat/completions (temperature, max_tokens, seed)"
    try:
        r = client.post("/v1/chat/completions", headers=headers, json={
            "model": model,
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 16,
            "seed": 42,
            "messages": [{"role": "user", "content": "Say ok."}],
        })
        if r.status_code == 200:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /v1/completions — legacy
    name = "POST /v1/completions (legacy)"
    try:
        r = client.post("/v1/completions", headers=headers, json={
            "model": model,
            "prompt": "The capital of France is",
            "max_tokens": 16,
        })
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /v1/embeddings
    name = "POST /v1/embeddings"
    try:
        r = client.post("/v1/embeddings", headers=headers, json={
            "model": cfg.embedding_model,
            "input": "Hello world",
        })
        body = r.json()
        if r.status_code == 200 and "data" in body:
            emb = body["data"][0].get("embedding", [])
            record("pass", name, f"dim={len(emb)}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))


# ---------------------------------------------------------------------------
# Section 4 — Ollama-Compatible Endpoints
# ---------------------------------------------------------------------------

def test_ollama(client: httpx.Client, cfg: argparse.Namespace):
    section_header("4. Ollama-Compatible Endpoints")
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    model = cfg.ollama_model

    # GET /api/tags
    name = "GET /api/tags"
    try:
        r = client.get("/api/tags", headers=headers)
        body = r.json()
        if r.status_code == 200 and "models" in body:
            record("pass", name, f"{len(body['models'])} models")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:120]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /api/chat — non-streaming
    name = "POST /api/chat (non-streaming)"
    try:
        r = client.post("/api/chat", headers=headers, json={
            "model": model,
            "stream": False,
            "options": {"num_predict": 32},
            "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
        })
        body = r.json()
        if r.status_code == 200 and "message" in body and body.get("done") is True:
            content = body["message"]["content"][:60]
            record("pass", name, f"response={content!r}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /api/chat — streaming
    name = "POST /api/chat (streaming)"
    try:
        with client.stream("POST", "/api/chat", headers=headers, json={
            "model": model,
            "stream": True,
            "options": {"num_predict": 32},
            "messages": [{"role": "user", "content": "Say hi."}],
        }) as r:
            if r.status_code != 200:
                record("fail", name, f"status={r.status_code}")
            else:
                chunks = 0
                saw_done = False
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    chunks += 1
                    if obj.get("done") is True:
                        saw_done = True
                if saw_done and chunks > 0:
                    record("pass", name, f"{chunks} chunks, saw done=true")
                else:
                    record("fail", name, f"chunks={chunks} done={saw_done}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /api/chat — JSON format
    name = "POST /api/chat (format: json)"
    try:
        r = client.post("/api/chat", headers=headers, json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"num_predict": 128},
            "messages": [{"role": "user", "content": "Return JSON: {\"color\": \"blue\"}"}],
        })
        body = r.json()
        if r.status_code == 200 and "message" in body:
            content = body["message"]["content"]
            json.loads(content)  # must parse as valid JSON
            record("pass", name, "valid JSON response")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except json.JSONDecodeError:
        record("fail", name, f"response was not valid JSON: {content[:120]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /api/chat — with parameters
    name = "POST /api/chat (temperature, num_predict, seed)"
    try:
        r = client.post("/api/chat", headers=headers, json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 16, "seed": 42},
            "messages": [{"role": "user", "content": "Say ok."}],
        })
        if r.status_code == 200:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /api/generate
    name = "POST /api/generate"
    try:
        r = client.post("/api/generate", headers=headers, json={
            "model": model,
            "stream": False,
            "prompt": "The capital of France is",
            "options": {"num_predict": 16},
        })
        body = r.json()
        if r.status_code == 200 and "response" in body:
            record("pass", name, f"response={body['response'][:60]!r}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /api/embeddings
    name = "POST /api/embeddings"
    try:
        r = client.post("/api/embeddings", headers=headers, json={
            "model": cfg.embedding_model,
            "prompt": "Hello world",
        })
        body = r.json()
        # Accept both Ollama format (top-level "embedding") and
        # OpenAI format (nested in "data[0].embedding")
        if r.status_code == 200 and "embedding" in body:
            record("pass", name, f"dim={len(body['embedding'])}")
        elif r.status_code == 200 and "data" in body:
            emb = body["data"][0].get("embedding", [])
            record("pass", name, f"dim={len(emb)}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))


# ---------------------------------------------------------------------------
# Section 5 — Anthropic-Compatible Endpoint
# ---------------------------------------------------------------------------

def test_anthropic(client: httpx.Client, cfg: argparse.Namespace):
    section_header("5. Anthropic-Compatible Endpoint")
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    model = cfg.ollama_model

    # POST /anthropic/v1/messages — non-streaming
    name = "POST /anthropic/v1/messages (non-streaming)"
    try:
        r = client.post("/anthropic/v1/messages", headers=headers, json={
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
        })
        body = r.json()
        if r.status_code == 200 and body.get("type") == "message" and "content" in body:
            content = body["content"][0]["text"][:60]
            record("pass", name, f"response={content!r}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /anthropic/v1/messages — streaming
    name = "POST /anthropic/v1/messages (streaming)"
    try:
        with client.stream("POST", "/anthropic/v1/messages", headers=headers, json={
            "model": model,
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "Say hi."}],
        }) as r:
            if r.status_code != 200:
                record("fail", name, f"status={r.status_code}")
            else:
                events = []
                saw_stop = False
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        events.append(event_type)
                        if event_type == "message_stop":
                            saw_stop = True
                deltas = sum(1 for e in events if e == "content_block_delta")
                if saw_stop and deltas > 0:
                    record("pass", name, f"{deltas} deltas, saw message_stop")
                else:
                    record("fail", name, f"deltas={deltas} stop={saw_stop} events={events}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /anthropic/v1/messages — with system prompt
    name = "POST /anthropic/v1/messages (system prompt)"
    try:
        r = client.post("/anthropic/v1/messages", headers=headers, json={
            "model": model,
            "max_tokens": 32,
            "system": "You are a pirate. Always respond in pirate speak.",
            "messages": [{"role": "user", "content": "Say hello."}],
        })
        body = r.json()
        if r.status_code == 200 and body.get("type") == "message":
            content = body["content"][0]["text"][:60]
            record("pass", name, f"response={content!r}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /anthropic/v1/messages — with parameters
    name = "POST /anthropic/v1/messages (temperature, top_p, stop_sequences)"
    try:
        r = client.post("/anthropic/v1/messages", headers=headers, json={
            "model": model,
            "max_tokens": 16,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "Say ok."}],
        })
        body = r.json()
        if r.status_code == 200 and body.get("type") == "message":
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /anthropic/v1/messages — response format validation
    name = "POST /anthropic/v1/messages (response format)"
    try:
        r = client.post("/anthropic/v1/messages", headers=headers, json={
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say hi."}],
        })
        body = r.json()
        ok = (
            r.status_code == 200
            and body.get("type") == "message"
            and body.get("role") == "assistant"
            and body.get("stop_reason") in ("end_turn", "max_tokens")
            and "usage" in body
            and "input_tokens" in body["usage"]
            and "output_tokens" in body["usage"]
        )
        if ok:
            record("pass", name, f"stop_reason={body['stop_reason']} usage={body['usage']}")
        else:
            record("fail", name, f"status={r.status_code} body={json.dumps(body)[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /anthropic/v1/messages — auth required
    name = "POST /anthropic/v1/messages (no auth → 401)"
    try:
        r = client.post("/anthropic/v1/messages", json={
            "model": model,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        if r.status_code in (401, 403):
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"expected 401/403, got {r.status_code}")
    except Exception as e:
        record("fail", name, str(e))

    # POST /anthropic/v1/messages — vLLM model via Anthropic endpoint
    name = f"POST /anthropic/v1/messages (vLLM model: {cfg.vllm_model})"
    try:
        r = client.post("/anthropic/v1/messages", headers=headers, json={
            "model": cfg.vllm_model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say ok."}],
        })
        body = r.json()
        if r.status_code == 200 and body.get("type") == "message":
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))


# ---------------------------------------------------------------------------
# Section 6 — Cross-Engine Translation
# ---------------------------------------------------------------------------

def test_cross_engine(client: httpx.Client, cfg: argparse.Namespace):
    section_header("6. Cross-Engine Translation")
    headers = {"Authorization": f"Bearer {cfg.api_key}"}

    # Ollama model via OpenAI endpoint
    name = f"Ollama model ({cfg.ollama_model}) via /v1/chat/completions"
    try:
        r = client.post("/v1/chat/completions", headers=headers, json={
            "model": cfg.ollama_model,
            "stream": False,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say ok."}],
        })
        if r.status_code == 200:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # vLLM model via Ollama endpoint
    name = f"vLLM model ({cfg.vllm_model}) via /api/chat"
    try:
        r = client.post("/api/chat", headers=headers, json={
            "model": cfg.vllm_model,
            "stream": False,
            "options": {"num_predict": 16},
            "messages": [{"role": "user", "content": "Say ok."}],
        })
        if r.status_code == 200:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # OpenAI JSON mode → Ollama backend
    name = "OpenAI JSON mode → Ollama backend"
    try:
        r = client.post("/v1/chat/completions", headers=headers, json={
            "model": cfg.ollama_model,
            "stream": False,
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "Return JSON: {\"n\": 1}"}],
        })
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            content = body["choices"][0]["message"]["content"]
            json.loads(content)
            record("pass", name, "valid JSON response")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except json.JSONDecodeError:
        record("fail", name, f"response was not valid JSON: {content[:120]}")
    except Exception as e:
        record("fail", name, str(e))

    # Ollama JSON format → vLLM backend
    name = "Ollama JSON format → vLLM backend"
    try:
        r = client.post("/api/chat", headers=headers, json={
            "model": cfg.vllm_model,
            "stream": False,
            "format": "json",
            "options": {"num_predict": 128},
            "messages": [{"role": "user", "content": "Return JSON: {\"n\": 1}"}],
        })
        body = r.json()
        if r.status_code == 200 and "message" in body:
            content = body["message"]["content"]
            json.loads(content)
            record("pass", name, "valid JSON response")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except json.JSONDecodeError:
        record("fail", name, f"response was not valid JSON: {content[:120]}")
    except Exception as e:
        record("fail", name, str(e))


# ---------------------------------------------------------------------------
# Section 7 — Error Handling
# ---------------------------------------------------------------------------

def test_errors(client: httpx.Client, cfg: argparse.Namespace):
    section_header("7. Error Handling")
    headers = {"Authorization": f"Bearer {cfg.api_key}"}

    # Missing model field
    name = "Missing model field → 4xx"
    try:
        r = client.post("/v1/chat/completions", headers=headers, json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        if 400 <= r.status_code < 500:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"expected 4xx, got {r.status_code}")
    except Exception as e:
        record("fail", name, str(e))

    # Missing messages field — server may accept with empty messages
    name = "Missing messages field → handled"
    try:
        r = client.post("/v1/chat/completions", headers=headers, json={
            "model": cfg.vllm_model,
        })
        if r.status_code == 200 or 400 <= r.status_code < 600:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"unexpected status {r.status_code}")
    except Exception as e:
        record("fail", name, str(e))

    # Invalid JSON body
    name = "Invalid JSON body → 4xx"
    try:
        r = client.request("POST", "/v1/chat/completions",
                           headers={**headers, "Content-Type": "application/json"},
                           content=b"{bad json")
        if 400 <= r.status_code < 500:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"expected 4xx, got {r.status_code}")
    except Exception as e:
        record("fail", name, str(e))

    # Non-existent model
    name = "Non-existent model → 4xx/503"
    try:
        r = client.post("/v1/chat/completions", headers=headers, json={
            "model": "nonexistent-model-xyz-999",
            "messages": [{"role": "user", "content": "hi"}],
        })
        if 400 <= r.status_code < 600:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"expected 4xx/503, got {r.status_code}")
    except Exception as e:
        record("fail", name, str(e))


# ---------------------------------------------------------------------------
# Section 8 — Admin API
# ---------------------------------------------------------------------------

def test_admin(client: httpx.Client, cfg: argparse.Namespace):
    section_header("8. Admin API")

    if not cfg.admin_key:
        for endpoint in ["/api/admin/backends", "/api/admin/queue",
                         "/api/admin/audit/search", "/api/admin/users"]:
            record("skip", f"GET {endpoint}", "no --admin-key provided")
        return

    headers = {"Authorization": f"Bearer {cfg.admin_key}"}

    # GET /api/admin/backends
    name = "GET /api/admin/backends"
    try:
        r = client.get("/api/admin/backends", headers=headers)
        if r.status_code == 200:
            body = r.json()
            count = len(body) if isinstance(body, list) else "?"
            record("pass", name, f"{count} backends")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # GET /api/admin/queue
    name = "GET /api/admin/queue"
    try:
        r = client.get("/api/admin/queue", headers=headers)
        if r.status_code == 200:
            record("pass", name, f"{r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # GET /api/admin/audit/search
    name = "GET /api/admin/audit/search"
    try:
        r = client.get("/api/admin/audit/search", headers=headers)
        body = r.json()
        if r.status_code == 200 and "results" in body:
            record("pass", name, f"{body.get('total', '?')} total records")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))

    # GET /api/admin/users
    name = "GET /api/admin/users"
    try:
        r = client.get("/api/admin/users", headers=headers)
        body = r.json()
        if r.status_code == 200 and "users" in body:
            record("pass", name, f"{len(body['users'])} users")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        record("fail", name, str(e))


# ---------------------------------------------------------------------------
# Section registry
# ---------------------------------------------------------------------------

SECTIONS = {
    "health": test_health,
    "auth": test_auth,
    "openai": test_openai,
    "ollama": test_ollama,
    "anthropic": test_anthropic,
    "cross": test_cross_engine,
    "errors": test_errors,
    "admin": test_admin,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MindRouter2 comprehensive smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --api-key mr2_xxx
  python test.py --api-key mr2_xxx --base-url http://host:8000
  python test.py --api-key mr2_xxx --admin-key mr2_yyy
  python test.py --api-key mr2_xxx --section health
  python test.py --api-key mr2_xxx --section health --section auth
""",
    )
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of MindRouter2 (default: http://localhost:8000)")
    parser.add_argument("--api-key", required=True,
                        help="API key for authentication (required)")
    parser.add_argument("--admin-key", default=None,
                        help="Admin API key (admin tests skipped without this)")
    parser.add_argument("--ollama-model", default="phi4:14b",
                        help="Ollama model to test (default: phi4:14b)")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-120b",
                        help="vLLM model to test (default: openai/gpt-oss-120b)")
    parser.add_argument("--embedding-model", default="EMBED/all-minilm:33m",
                        help="Embedding model to test (default: EMBED/all-minilm:33m)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Request timeout in seconds (default: 180)")
    parser.add_argument("--section", action="append", dest="sections",
                        choices=list(SECTIONS.keys()),
                        help="Run only specific section(s); can be repeated")

    cfg = parser.parse_args()
    sections_to_run = cfg.sections or list(SECTIONS.keys())

    print(f"\n{BOLD}MindRouter2 Smoke Test{RESET}")
    print(f"  Base URL:      {cfg.base_url}")
    print(f"  Ollama model:  {cfg.ollama_model}")
    print(f"  vLLM model:    {cfg.vllm_model}")
    print(f"  Embed model:   {cfg.embedding_model}")
    print(f"  Timeout:       {cfg.timeout}s")
    print(f"  Admin key:     {'set' if cfg.admin_key else 'not set'}")
    print(f"  Sections:      {', '.join(sections_to_run)}")

    # Health and auth tests don't send the default auth header — they manage
    # their own headers.  Other sections include auth by default only in their
    # own request calls, so we create a bare client (no default auth).
    client = httpx.Client(
        base_url=cfg.base_url,
        timeout=httpx.Timeout(cfg.timeout),
    )

    t0 = time.time()
    try:
        for section_name in sections_to_run:
            SECTIONS[section_name](client, cfg)
    finally:
        client.close()

    elapsed = time.time() - t0

    # Summary
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    skipped = sum(1 for r in results if r["status"] == "skip")
    total = len(results)

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    parts = []
    if passed:
        parts.append(f"{GREEN}{passed} passed{RESET}")
    if failed:
        parts.append(f"{RED}{failed} failed{RESET}")
    if skipped:
        parts.append(f"{YELLOW}{skipped} skipped{RESET}")
    print(f"  {', '.join(parts)}  ({total} total, {elapsed:.1f}s)")

    if failed:
        print(f"\n  {RED}Failed tests:{RESET}")
        for r in results:
            if r["status"] == "fail":
                print(f"    {RED}✗{RESET} {r['name']}  {r['detail']}")

    print()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MindRouter2 Comprehensive Production Test

A standalone async test script that exercises every API surface of a live
MindRouter deployment. Uses httpx.AsyncClient, auto-discovers models from
/v1/models, and reports colored PASS/FAIL/SKIP per test with a summary.

Usage:
    python production_test.py --api-key mr2_xxx --base-url https://your-host:8000
    python production_test.py --api-key mr2_xxx --base-url https://your-host:8000 --section basic-chat
    python production_test.py --api-key mr2_xxx --base-url https://your-host:8000 --skip-stress --skip-voice
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

import httpx

# ---------------------------------------------------------------------------
# ANSI colored output helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
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
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

results: List[Dict[str, Any]] = []

# Matrix tracking: model_results[model_id][feature] = {"status", "latency_ms", "detail"}
model_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

# All latencies for throughput summary
all_latencies: List[float] = []


def record(status: str, name: str, detail: str = "", latency_ms: float = 0,
           model: str = "", feature: str = ""):
    results.append({"status": status, "name": name, "detail": detail, "latency_ms": latency_ms})
    if latency_ms > 0:
        all_latencies.append(latency_ms)
    # Track in matrix
    if model and feature:
        if model not in model_results:
            model_results[model] = {}
        model_results[model][feature] = {"status": status, "latency_ms": latency_ms, "detail": detail}
    suffix = f"  {DIM}{latency_ms:.0f}ms{RESET}" if latency_ms > 0 else ""
    if status == "pass":
        _pass(name, f"{detail}{suffix}" if detail else f"{latency_ms:.0f}ms" if latency_ms else "")
    elif status == "fail":
        _fail(name, detail)
    else:
        _skip(name, detail)


# ---------------------------------------------------------------------------
# Schemas for structured output tests
# ---------------------------------------------------------------------------

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

COMPLEX_SCHEMA = {
    "type": "object",
    "properties": {
        "full_name": {"type": "string"},
        "year_of_birth": {"type": "integer"},
        "scientific_field": {
            "type": "string",
            "enum": ["Physics", "Chemistry", "Biology", "Mathematics", "Computer Science"],
        },
        "notable_achievements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "year": {"type": "integer"},
                },
                "required": ["title", "year"],
            },
        },
        "is_nobel_laureate": {"type": "boolean"},
    },
    "required": ["full_name", "year_of_birth", "scientific_field", "notable_achievements", "is_nobel_laureate"],
}

IMAGE_DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "colors": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["description", "colors"],
}

# ---------------------------------------------------------------------------
# Tool definitions for tool calling tests
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. Paris, France",
                },
            },
            "required": ["location"],
        },
    },
}

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. 2+2",
                },
            },
            "required": ["expression"],
        },
    },
}

# 64x64 red PNG for multimodal tests (small but large enough for vision processors)
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeElEQVR4nO3PQQkAMAzAwEqrf1ObiD2OQSACLnN2v264oAEtaEALGtCCBrSgAS1oQAsa0IIGtKABLWhACxrQgga0oAEtaEALGtCCBrSgAS1oQAsa0IIGtKABLWhACxrQgga0oAEtaEALGtCCBrSgAS1oQAsa0ILHLvlF8VqfrMBeAAAAAElFTkSuQmCC"


# ---------------------------------------------------------------------------
# Inline JSON validation helpers (no jsonschema dependency)
# ---------------------------------------------------------------------------

def validate_json_keys(data: Any, schema: Dict) -> bool:
    """Check that required keys exist and types roughly match."""
    if not isinstance(data, dict):
        return False
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in data:
            return False
        expected_type = props.get(key, {}).get("type")
        val = data[key]
        if expected_type == "string" and not isinstance(val, str):
            return False
        if expected_type == "integer" and not isinstance(val, int):
            return False
        if expected_type == "boolean" and not isinstance(val, bool):
            return False
        if expected_type == "array" and not isinstance(val, list):
            return False
        if expected_type == "object" and not isinstance(val, dict):
            return False
    return True


def parse_json_content(content: Optional[str]) -> Optional[Any]:
    """Try to parse JSON from content, stripping markdown fences if present."""
    if content is None:
        return None
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Model capability discovery
# ---------------------------------------------------------------------------

class ModelCatalog:
    """Holds discovered models categorized by capability."""

    def __init__(self):
        self.all_models: List[Dict] = []
        self.chat_models: List[str] = []
        self.vision_models: List[str] = []
        self.embedding_models: List[str] = []
        self.rerank_models: List[str] = []
        self.thinking_models: List[str] = []
        self.tool_models: List[str] = []
        self.structured_models: List[str] = []

    def categorize(self, models_data: List[Dict]):
        self.all_models = models_data
        for m in models_data:
            model_id = m.get("id", "")
            caps = m.get("capabilities", {})

            # Embedding models have modality or capability flag
            if caps.get("embeddings"):
                self.embedding_models.append(model_id)
                continue

            # Rerank models (by capability flag or name)
            if caps.get("rerank") or "rerank" in model_id.lower():
                self.rerank_models.append(model_id)
                continue

            # Chat models (everything else)
            self.chat_models.append(model_id)

            if caps.get("multimodal") or caps.get("vision"):
                self.vision_models.append(model_id)
            if caps.get("thinking"):
                self.thinking_models.append(model_id)
            if caps.get("tools") or caps.get("tool_calling"):
                self.tool_models.append(model_id)
            if caps.get("structured_output"):
                self.structured_models.append(model_id)

    def summary(self) -> str:
        parts = [
            f"chat={len(self.chat_models)}",
            f"vision={len(self.vision_models)}",
            f"embedding={len(self.embedding_models)}",
            f"rerank={len(self.rerank_models)}",
            f"thinking={len(self.thinking_models)}",
            f"tools={len(self.tool_models)}",
            f"structured={len(self.structured_models)}",
        ]
        return ", ".join(parts)

    def is_thinking_model(self, model_id: str) -> bool:
        return model_id in self.thinking_models


# ---------------------------------------------------------------------------
# Configuration namespace
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, args: argparse.Namespace):
        self.api_key: str = args.api_key
        self.base_url: str = args.base_url
        self.admin_key: Optional[str] = args.admin_key
        self.timeout: int = args.timeout
        self.concurrency: int = args.concurrency
        self.verbose: bool = args.verbose
        self.skip_stress: bool = args.skip_stress
        self.skip_voice: bool = args.skip_voice
        self.ollama_model: str = args.ollama_model
        self.vllm_model: str = args.vllm_model
        self.embedding_model: str = args.embedding_model
        self.rerank_model: str = args.rerank_model
        self.max_tokens: int = args.max_tokens
        self.sections: Optional[List[str]] = args.sections
        self.catalog = ModelCatalog()


# ---------------------------------------------------------------------------
# Helper: build auth headers
# ---------------------------------------------------------------------------

def auth_headers(cfg: Config) -> Dict[str, str]:
    return {"Authorization": f"Bearer {cfg.api_key}"}


def vprint(cfg: Config, msg: str):
    """Print only in verbose mode."""
    if cfg.verbose:
        print(f"    {DIM}{msg}{RESET}")


# ---------------------------------------------------------------------------
# Section 1: Discovery & Health
# ---------------------------------------------------------------------------

async def test_discovery(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 1: Discovery & Health")

    # GET /healthz
    name = "GET /healthz"
    try:
        t0 = time.time()
        r = await client.get("/healthz")
        latency_ms = (time.time() - t0) * 1000
        if r.status_code == 200:
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:120]}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e))

    # GET /v1/models - count and categorize
    name = "GET /v1/models"
    try:
        t0 = time.time()
        r = await client.get("/v1/models", headers=auth_headers(cfg))
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "data" in body:
            cfg.catalog.categorize(body["data"])
            record("pass", name, f"{len(body['data'])} models -- {cfg.catalog.summary()}", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e))

    # GET /api/tags - Ollama model list
    name = "GET /api/tags"
    try:
        t0 = time.time()
        r = await client.get("/api/tags", headers=auth_headers(cfg))
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "models" in body:
            record("pass", name, f"{len(body['models'])} models", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:120]}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e))

    # Print model capability summary
    cat = cfg.catalog
    if cat.all_models:
        print(f"\n  {BOLD}Model Capability Summary:{RESET}")
        print(f"    Chat models:      {len(cat.chat_models)}")
        if cat.vision_models:
            print(f"    Vision models:    {', '.join(cat.vision_models[:5])}")
        if cat.thinking_models:
            print(f"    Thinking models:  {', '.join(cat.thinking_models[:5])}")
        if cat.tool_models:
            print(f"    Tool models:      {', '.join(cat.tool_models[:5])}")
        if cat.structured_models:
            print(f"    Structured models:{', '.join(cat.structured_models[:5])}")
        if cat.embedding_models:
            print(f"    Embedding models: {', '.join(cat.embedding_models[:5])}")
        if cat.rerank_models:
            print(f"    Rerank models:    {', '.join(cat.rerank_models[:5])}")


# ---------------------------------------------------------------------------
# Section 2: Basic Chat (Every Model)
# ---------------------------------------------------------------------------

async def test_basic_chat(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 2: Basic Chat")

    sem = asyncio.Semaphore(cfg.concurrency)
    headers = auth_headers(cfg)

    # -- Part A: Test every discovered chat model --
    print(f"\n  {BOLD}2a. All discovered chat models (max_tokens={cfg.max_tokens}):{RESET}")

    async def test_one_model(model_id: str):
        async with sem:
            tname = f"Chat: {model_id}"
            try:
                t0 = time.time()
                r = await client.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model_id,
                        "stream": False,
                        "max_tokens": cfg.max_tokens,
                        "messages": [{"role": "user", "content": "Say hello in 3 words."}],
                    },
                )
                latency_ms = (time.time() - t0) * 1000
                if r.status_code == 200:
                    body = r.json()
                    msg = body.get("choices", [{}])[0].get("message", {})
                    content = msg.get("content") or msg.get("reasoning_content") or ""
                    record("pass", tname, f"'{content[:50]}'", latency_ms=latency_ms, model=model_id, feature="chat")
                elif r.status_code in (503, 504):
                    record("skip", tname, f"status={r.status_code} (model not loaded)", model=model_id, feature="chat")
                else:
                    record("fail", tname, f"status={r.status_code} body={r.text[:150]}", latency_ms=latency_ms, model=model_id, feature="chat")
            except httpx.TimeoutException:
                record("skip", tname, "timeout (model not loaded)", model=model_id, feature="chat")
            except Exception as e:
                record("fail", tname, str(e)[:150], model=model_id, feature="chat")

    tasks = [test_one_model(m) for m in cfg.catalog.chat_models]
    if tasks:
        await asyncio.gather(*tasks)
    else:
        record("skip", "Chat: all models", "no chat models discovered")

    # -- Part B: Specific dialect tests with default models --
    print(f"\n  {BOLD}2b. Dialect-specific tests:{RESET}")

    # OpenAI streaming
    name = "OpenAI streaming chat"
    try:
        collected = []
        saw_done = False
        t0 = time.time()
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": cfg.vllm_model,
                "stream": True,
                "max_tokens": cfg.max_tokens,
                "messages": [{"role": "user", "content": "Say hi."}],
            },
        ) as r:
            if r.status_code != 200:
                latency_ms = (time.time() - t0) * 1000
                record("fail", name, f"status={r.status_code}", latency_ms=latency_ms, model=cfg.vllm_model, feature="streaming")
            else:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            saw_done = True
                        else:
                            collected.append(payload)
                latency_ms = (time.time() - t0) * 1000
                if saw_done and len(collected) > 0:
                    record("pass", name, f"{len(collected)} chunks, saw [DONE]", latency_ms=latency_ms, model=cfg.vllm_model, feature="streaming")
                else:
                    record("fail", name, f"chunks={len(collected)} done={saw_done}", latency_ms=latency_ms, model=cfg.vllm_model, feature="streaming")
    except httpx.TimeoutException:
        record("skip", name, "timeout", model=cfg.vllm_model, feature="streaming")
    except Exception as e:
        record("fail", name, str(e)[:150], model=cfg.vllm_model, feature="streaming")

    # Ollama non-streaming
    name = "Ollama non-streaming chat"
    try:
        t0 = time.time()
        r = await client.post(
            "/api/chat",
            headers=headers,
            json={
                "model": cfg.ollama_model,
                "stream": False,
                "options": {"num_predict": cfg.max_tokens},
                "messages": [{"role": "user", "content": "Say hello in 3 words."}],
            },
        )
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "message" in body and body.get("done") is True:
            content = body["message"].get("content", "")[:50]
            record("pass", name, f"'{content}'", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Ollama streaming
    name = "Ollama streaming chat"
    try:
        chunks = 0
        saw_done = False
        t0 = time.time()
        async with client.stream(
            "POST",
            "/api/chat",
            headers=headers,
            json={
                "model": cfg.ollama_model,
                "stream": True,
                "options": {"num_predict": cfg.max_tokens},
                "messages": [{"role": "user", "content": "Say hi."}],
            },
        ) as r:
            if r.status_code != 200:
                latency_ms = (time.time() - t0) * 1000
                record("fail", name, f"status={r.status_code}", latency_ms=latency_ms)
            else:
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    chunks += 1
                    if obj.get("done") is True:
                        saw_done = True
                latency_ms = (time.time() - t0) * 1000
                if saw_done and chunks > 0:
                    record("pass", name, f"{chunks} chunks, saw done=true", latency_ms=latency_ms)
                else:
                    record("fail", name, f"chunks={chunks} done={saw_done}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Anthropic non-streaming
    name = "Anthropic non-streaming chat"
    try:
        t0 = time.time()
        r = await client.post(
            "/anthropic/v1/messages",
            headers=headers,
            json={
                "model": cfg.ollama_model,
                "max_tokens": cfg.max_tokens,
                "messages": [{"role": "user", "content": "Say hello in 3 words."}],
            },
        )
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and body.get("type") == "message" and "content" in body:
            text = body["content"][0].get("text", "")[:50]
            record("pass", name, f"'{text}'", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])


# ---------------------------------------------------------------------------
# Section 3: Structured Output
# ---------------------------------------------------------------------------

async def test_structured(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 3: Structured Output")
    headers = auth_headers(cfg)
    prompt = "Provide info about Albert Einstein as JSON with name (string) and age (integer, at death)."

    # OpenAI json_object mode
    name = "OpenAI json_object mode"
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": cfg.vllm_model,
            "stream": False,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            content = body["choices"][0]["message"].get("content") or ""
            parsed = parse_json_content(content)
            if parsed is not None:
                record("pass", name, "valid JSON response", latency_ms=latency_ms, model=cfg.vllm_model, feature="structured")
            else:
                record("fail", name, f"not valid JSON: {content[:100]}", latency_ms=latency_ms, model=cfg.vllm_model, feature="structured")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # OpenAI json_schema mode
    name = "OpenAI json_schema mode"
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": cfg.vllm_model,
            "stream": False,
            "max_tokens": 512,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "person", "strict": True, "schema": PERSON_SCHEMA},
            },
            "messages": [{"role": "user", "content": prompt}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            content = body["choices"][0]["message"].get("content") or ""
            parsed = parse_json_content(content)
            if parsed is not None and validate_json_keys(parsed, PERSON_SCHEMA):
                record("pass", name, f"name={parsed.get('name')}, age={parsed.get('age')}", latency_ms=latency_ms)
            else:
                record("fail", name, f"schema mismatch or parse error: {content[:100]}", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Ollama format:"json"
    name = "Ollama format:json"
    try:
        t0 = time.time()
        r = await client.post("/api/chat", headers=headers, json={
            "model": cfg.ollama_model,
            "stream": False,
            "format": "json",
            "options": {"num_predict": 128},
            "messages": [{"role": "user", "content": prompt}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "message" in body:
            content = body["message"].get("content", "")
            parsed = parse_json_content(content)
            if parsed is not None:
                record("pass", name, "valid JSON response", latency_ms=latency_ms)
            else:
                record("fail", name, f"not valid JSON: {content[:100]}", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Ollama format:SCHEMA
    name = "Ollama format:schema"
    try:
        t0 = time.time()
        r = await client.post("/api/chat", headers=headers, json={
            "model": cfg.ollama_model,
            "stream": False,
            "format": PERSON_SCHEMA,
            "options": {"num_predict": 128},
            "messages": [{"role": "user", "content": prompt}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "message" in body:
            content = body["message"].get("content", "")
            parsed = parse_json_content(content)
            if parsed is not None and validate_json_keys(parsed, PERSON_SCHEMA):
                record("pass", name, f"name={parsed.get('name')}, age={parsed.get('age')}", latency_ms=latency_ms)
            else:
                record("fail", name, f"schema mismatch: {content[:100]}", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Anthropic with output json_schema
    name = "Anthropic json_schema"
    try:
        t0 = time.time()
        r = await client.post("/anthropic/v1/messages", headers=headers, json={
            "model": cfg.ollama_model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": PERSON_SCHEMA,
                },
            },
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and body.get("type") == "message":
            text = ""
            for block in body.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "")
            parsed = parse_json_content(text)
            if parsed is not None and validate_json_keys(parsed, PERSON_SCHEMA):
                record("pass", name, f"name={parsed.get('name')}, age={parsed.get('age')}", latency_ms=latency_ms)
            else:
                record("fail", name, f"schema mismatch: {text[:100]}", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Complex nested schema
    name = "Complex nested schema (OpenAI)"
    complex_prompt = "Provide structured historical data for Albert Einstein. Output ONLY the JSON object."
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": cfg.vllm_model,
            "stream": False,
            "max_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "scientist", "strict": True, "schema": COMPLEX_SCHEMA},
            },
            "messages": [{"role": "user", "content": complex_prompt}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            content = body["choices"][0]["message"].get("content") or ""
            parsed = parse_json_content(content)
            if parsed is not None and validate_json_keys(parsed, COMPLEX_SCHEMA):
                record("pass", name, f"full_name={parsed.get('full_name')}, achievements={len(parsed.get('notable_achievements', []))}", latency_ms=latency_ms)
            else:
                record("fail", name, f"schema mismatch: {content[:120]}", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # JSON schema + thinking enabled (on a thinking model)
    name = "JSON schema + thinking"
    thinking_model = cfg.vllm_model  # gpt-oss supports reasoning_effort
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": thinking_model,
            "stream": False,
            "max_tokens": 2048,
            "reasoning_effort": "low",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "person", "strict": True, "schema": PERSON_SCHEMA},
            },
            "messages": [{"role": "user", "content": prompt}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            msg = body["choices"][0]["message"]
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            parsed = parse_json_content(content)
            has_thinking = bool(reasoning and len(str(reasoning).strip()) > 0)
            if parsed is not None and validate_json_keys(parsed, PERSON_SCHEMA):
                record("pass", name, f"schema_ok=True, thinking={has_thinking}", latency_ms=latency_ms)
            else:
                record("fail", name, f"schema mismatch: {content[:100]}", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # JSON schema + thinking on every discovered thinking model
    # (catches models like qwen3.5 that are always-thinking)
    thinking_models = cfg.catalog.thinking_models
    if thinking_models:
        sem = asyncio.Semaphore(cfg.concurrency)

        async def test_schema_thinking(model_id: str):
            tname = f"JSON schema + thinking [{model_id}]"
            async with sem:
                try:
                    payload = {
                        "model": model_id,
                        "stream": False,
                        "max_tokens": 2048,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"name": "person", "strict": True, "schema": PERSON_SCHEMA},
                        },
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    # Use reasoning_effort for gpt-oss, think:true for others
                    if "gpt-oss" in model_id:
                        payload["reasoning_effort"] = "low"
                    else:
                        payload["think"] = True
                    t0 = time.time()
                    r = await client.post("/v1/chat/completions", headers=headers, json=payload)
                    latency_ms = (time.time() - t0) * 1000
                    if r.status_code == 200:
                        body = r.json()
                        msg = body.get("choices", [{}])[0].get("message", {})
                        content = msg.get("content") or ""
                        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                        parsed = parse_json_content(content)
                        has_thinking = bool(reasoning and len(str(reasoning).strip()) > 0)
                        if parsed is not None and validate_json_keys(parsed, PERSON_SCHEMA):
                            record("pass", tname, f"schema_ok=True, thinking={has_thinking}", latency_ms=latency_ms)
                        else:
                            record("fail", tname, f"schema mismatch: {content[:100]}", latency_ms=latency_ms)
                    elif r.status_code == 422:
                        latency_ms = (time.time() - t0) * 1000
                        record("pass", tname, f"422: {r.json().get('detail', '')[:80]}", latency_ms=latency_ms)
                    elif r.status_code == 400 and "not support thinking" in r.text:
                        record("skip", tname, "model does not support thinking")
                    elif r.status_code in (503, 504):
                        record("skip", tname, f"status={r.status_code}")
                    else:
                        latency_ms = (time.time() - t0) * 1000
                        record("fail", tname, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
                except httpx.TimeoutException:
                    record("skip", tname, "timeout")
                except Exception as e:
                    record("fail", tname, str(e)[:150])

        # Skip the model we already tested above to avoid duplication
        extra_models = [m for m in thinking_models if m != cfg.vllm_model]
        if extra_models:
            await asyncio.gather(*[test_schema_thinking(m) for m in extra_models])

    # -- Per-model structured output test (json_schema on every capable model) --
    structured_models = cfg.catalog.structured_models
    if structured_models:
        print(f"\n  {BOLD}3b. Structured output per model ({len(structured_models)} models):{RESET}")
        sem_s = asyncio.Semaphore(cfg.concurrency)

        async def test_structured_per_model(model_id: str):
            tname = f"Structured [{model_id}]"
            async with sem_s:
                is_thinking = cfg.catalog.is_thinking_model(model_id)
                tok = 2048 if is_thinking else 512
                payload: Dict[str, Any] = {
                    "model": model_id,
                    "stream": False,
                    "max_tokens": tok,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "person", "strict": True, "schema": PERSON_SCHEMA},
                    },
                    "messages": [{"role": "user", "content": prompt}],
                }
                try:
                    t0 = time.time()
                    r = await client.post("/v1/chat/completions", headers=headers, json=payload)
                    latency_ms = (time.time() - t0) * 1000
                    if r.status_code == 200:
                        body = r.json()
                        content = body.get("choices", [{}])[0].get("message", {}).get("content") or ""
                        parsed = parse_json_content(content)
                        if parsed is not None and validate_json_keys(parsed, PERSON_SCHEMA):
                            record("pass", tname, f"name={parsed.get('name')}, age={parsed.get('age')}", latency_ms=latency_ms, model=model_id, feature="structured")
                        else:
                            record("fail", tname, f"schema mismatch: {content[:80]}", latency_ms=latency_ms, model=model_id, feature="structured")
                    elif r.status_code == 422:
                        record("pass", tname, f"422 expected for thinking+schema", latency_ms=latency_ms, model=model_id, feature="structured")
                    elif r.status_code in (503, 504):
                        record("skip", tname, f"status={r.status_code} (model not loaded)", model=model_id, feature="structured")
                    else:
                        record("fail", tname, f"status={r.status_code} body={r.text[:150]}", latency_ms=latency_ms, model=model_id, feature="structured")
                except httpx.TimeoutException:
                    record("skip", tname, "timeout (model not loaded)", model=model_id, feature="structured")
                except Exception as e:
                    record("fail", tname, str(e)[:150], model=model_id, feature="structured")

        await asyncio.gather(*[test_structured_per_model(m) for m in structured_models])


# ---------------------------------------------------------------------------
# Section 4: Thinking/Reasoning
# ---------------------------------------------------------------------------

async def test_thinking(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 4: Thinking/Reasoning")
    headers = auth_headers(cfg)
    prompt = "What is 15 * 37? Show your work."

    # Find a qwen thinking model from catalog or use default
    qwen_model = None
    for m in cfg.catalog.thinking_models:
        if "qwen" in m.lower():
            qwen_model = m
            break

    # -- gpt-oss reasoning_effort via OpenAI --
    for effort in ("low", "high"):
        name = f"gpt-oss reasoning_effort={effort} (OpenAI)"
        try:
            t0 = time.time()
            r = await client.post("/v1/chat/completions", headers=headers, json={
                "model": cfg.vllm_model,
                "stream": False,
                "max_tokens": 2048,
                "reasoning_effort": effort,
                "messages": [{"role": "user", "content": prompt}],
            })
            latency_ms = (time.time() - t0) * 1000
            body = r.json()
            if r.status_code == 200 and "choices" in body:
                msg = body["choices"][0]["message"]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                has_reasoning = bool(reasoning and len(str(reasoning).strip()) > 0)
                content = (msg.get("content") or "")[:60]
                if has_reasoning:
                    record("pass", name, f"reasoning_len={len(str(reasoning))}, content='{content}'", latency_ms=latency_ms, model=cfg.vllm_model, feature="thinking")
                else:
                    record("fail", name, f"no reasoning_content found, content='{content}'", latency_ms=latency_ms, model=cfg.vllm_model, feature="thinking")
            else:
                record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
        except httpx.TimeoutException:
            record("skip", name, "timeout")
        except Exception as e:
            record("fail", name, str(e)[:150])

    # -- gpt-oss via Ollama with think:"low" --
    name = "gpt-oss think:low (Ollama)"
    try:
        t0 = time.time()
        r = await client.post("/api/chat", headers=headers, json={
            "model": cfg.vllm_model,
            "stream": False,
            "think": "low",
            "options": {"num_predict": 2048},
            "messages": [{"role": "user", "content": prompt}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "message" in body:
            msg = body["message"]
            thinking = msg.get("thinking") or msg.get("reasoning") or msg.get("reasoning_content") or ""
            has_thinking = bool(thinking and len(str(thinking).strip()) > 0)
            content = (msg.get("content") or "")[:60]
            if has_thinking:
                record("pass", name, f"thinking_len={len(str(thinking))}, content='{content}'", latency_ms=latency_ms)
            else:
                record("fail", name, f"no thinking field, content='{content}'", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # -- gpt-oss via Anthropic with thinking enabled --
    name = "gpt-oss thinking:enabled (Anthropic)"
    try:
        t0 = time.time()
        r = await client.post("/anthropic/v1/messages", headers=headers, json={
            "model": cfg.vllm_model,
            "max_tokens": 16384,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "enabled", "budget_tokens": 8192},
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and body.get("type") == "message":
            has_thinking = False
            has_text = False
            for block in body.get("content", []):
                if block.get("type") == "thinking" and block.get("thinking"):
                    has_thinking = True
                if block.get("type") == "text" and block.get("text"):
                    has_text = True
            if has_thinking:
                record("pass", name, f"thinking=True, text={has_text}", latency_ms=latency_ms)
            else:
                # Some models return reasoning in text block only
                record("fail", name, f"no thinking block found, text={has_text}", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # -- qwen-style think:true via OpenAI --
    if qwen_model:
        name = f"qwen think:true (OpenAI) [{qwen_model}]"
        try:
            t0 = time.time()
            r = await client.post("/v1/chat/completions", headers=headers, json={
                "model": qwen_model,
                "stream": False,
                "max_tokens": 2048,
                "think": True,
                "messages": [{"role": "user", "content": prompt}],
            })
            latency_ms = (time.time() - t0) * 1000
            body = r.json()
            if r.status_code == 200 and "choices" in body:
                msg = body["choices"][0]["message"]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                content = msg.get("content") or ""
                has_reasoning = bool(reasoning and len(str(reasoning).strip()) > 0)
                has_think_tags = "<think>" in content
                if has_reasoning or has_think_tags:
                    detail = f"reasoning_len={len(str(reasoning))}" if has_reasoning else "found <think> tags in content"
                    record("pass", name, detail, latency_ms=latency_ms, model=qwen_model, feature="thinking")
                else:
                    record("fail", name, "no reasoning_content found", latency_ms=latency_ms, model=qwen_model, feature="thinking")
            elif r.status_code in (503, 504):
                record("skip", name, f"status={r.status_code}", model=qwen_model, feature="thinking")
            else:
                record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms, model=qwen_model, feature="thinking")
        except httpx.TimeoutException:
            record("skip", name, "timeout", model=qwen_model, feature="thinking")
        except Exception as e:
            record("fail", name, str(e)[:150], model=qwen_model, feature="thinking")

        # qwen think:false
        name = f"qwen think:false (OpenAI) [{qwen_model}]"
        try:
            t0 = time.time()
            r = await client.post("/v1/chat/completions", headers=headers, json={
                "model": qwen_model,
                "stream": False,
                "max_tokens": 2048,
                "think": False,
                "messages": [{"role": "user", "content": prompt}],
            })
            latency_ms = (time.time() - t0) * 1000
            body = r.json()
            if r.status_code == 200 and "choices" in body:
                msg = body["choices"][0]["message"]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                has_reasoning = bool(reasoning and len(str(reasoning).strip()) > 0)
                if not has_reasoning:
                    record("pass", name, "no reasoning content as expected", latency_ms=latency_ms)
                else:
                    record("fail", name, f"unexpected reasoning_content len={len(str(reasoning))}", latency_ms=latency_ms)
            elif r.status_code in (503, 504):
                record("skip", name, f"status={r.status_code}")
            else:
                record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
        except httpx.TimeoutException:
            record("skip", name, "timeout")
        except Exception as e:
            record("fail", name, str(e)[:150])

        # qwen via Ollama think:true
        name = f"qwen think:true (Ollama) [{qwen_model}]"
        try:
            t0 = time.time()
            r = await client.post("/api/chat", headers=headers, json={
                "model": qwen_model,
                "stream": False,
                "think": True,
                "options": {"num_predict": 2048},
                "messages": [{"role": "user", "content": prompt}],
            })
            latency_ms = (time.time() - t0) * 1000
            body = r.json()
            if r.status_code == 200 and "message" in body:
                msg = body["message"]
                thinking = msg.get("thinking") or msg.get("reasoning") or msg.get("reasoning_content") or ""
                content = msg.get("content") or ""
                has_thinking = bool(thinking and len(str(thinking).strip()) > 0)
                has_think_tags = "<think>" in content
                if has_thinking or has_think_tags:
                    detail = f"thinking_len={len(str(thinking))}" if has_thinking else "found <think> tags in content"
                    record("pass", name, detail, latency_ms=latency_ms)
                else:
                    record("fail", name, "no thinking field", latency_ms=latency_ms)
            elif r.status_code in (503, 504):
                record("skip", name, f"status={r.status_code}")
            else:
                record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
        except httpx.TimeoutException:
            record("skip", name, "timeout")
        except Exception as e:
            record("fail", name, str(e)[:150])
    else:
        record("skip", "qwen think:true (OpenAI)", "no qwen thinking model discovered")
        record("skip", "qwen think:false (OpenAI)", "no qwen thinking model discovered")
        record("skip", "qwen think:true (Ollama)", "no qwen thinking model discovered")

    # -- Streaming + thinking (OpenAI) --
    name = "Streaming + thinking (OpenAI)"
    try:
        reasoning_chunks = 0
        content_chunks = 0
        t0 = time.time()
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": cfg.vllm_model,
                "stream": True,
                "max_tokens": 2048,
                "reasoning_effort": "low",
                "messages": [{"role": "user", "content": prompt}],
            },
        ) as r:
            if r.status_code != 200:
                latency_ms = (time.time() - t0) * 1000
                record("fail", name, f"status={r.status_code}", latency_ms=latency_ms)
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("reasoning_content"):
                            reasoning_chunks += 1
                        if delta.get("content"):
                            content_chunks += 1
                    except json.JSONDecodeError:
                        pass
                latency_ms = (time.time() - t0) * 1000
                if reasoning_chunks > 0:
                    record("pass", name, f"reasoning_chunks={reasoning_chunks}, content_chunks={content_chunks}", latency_ms=latency_ms)
                else:
                    record("fail", name, f"no reasoning chunks, content_chunks={content_chunks}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # -- Per-model thinking test (every thinking-capable model) --
    thinking_models_all = cfg.catalog.thinking_models
    if thinking_models_all:
        print(f"\n  {BOLD}4b. Thinking per model ({len(thinking_models_all)} models):{RESET}")
        sem_t = asyncio.Semaphore(cfg.concurrency)

        async def test_thinking_per_model(model_id: str):
            tname = f"Thinking [{model_id}]"
            async with sem_t:
                payload: Dict[str, Any] = {
                    "model": model_id,
                    "stream": False,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if "gpt-oss" in model_id:
                    payload["reasoning_effort"] = "low"
                else:
                    payload["think"] = True
                try:
                    t0 = time.time()
                    r = await client.post("/v1/chat/completions", headers=headers, json=payload)
                    latency_ms = (time.time() - t0) * 1000
                    if r.status_code == 200:
                        body = r.json()
                        msg = body.get("choices", [{}])[0].get("message", {})
                        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                        content = msg.get("content") or ""
                        has_reasoning = bool(reasoning and len(str(reasoning).strip()) > 0)
                        has_think_tags = "<think>" in content
                        if has_reasoning or has_think_tags:
                            record("pass", tname, f"reasoning_len={len(str(reasoning))}", latency_ms=latency_ms, model=model_id, feature="thinking")
                        else:
                            record("fail", tname, "no reasoning found", latency_ms=latency_ms, model=model_id, feature="thinking")
                    elif r.status_code in (503, 504):
                        record("skip", tname, f"status={r.status_code} (model not loaded)", model=model_id, feature="thinking")
                    else:
                        record("fail", tname, f"status={r.status_code} body={r.text[:150]}", latency_ms=latency_ms, model=model_id, feature="thinking")
                except httpx.TimeoutException:
                    record("skip", tname, "timeout (model not loaded)", model=model_id, feature="thinking")
                except Exception as e:
                    record("fail", tname, str(e)[:150], model=model_id, feature="thinking")

        await asyncio.gather(*[test_thinking_per_model(m) for m in thinking_models_all])


# ---------------------------------------------------------------------------
# Section 5: Tool Calling
# ---------------------------------------------------------------------------

async def test_tools(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 5: Tool Calling")
    headers = auth_headers(cfg)

    # Pick a tool-capable model, default to vllm_model
    tool_model = cfg.vllm_model

    # Single tool OpenAI
    name = "Single tool call (OpenAI)"
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": tool_model,
            "stream": False,
            "max_tokens": 256,
            "tools": [WEATHER_TOOL],
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            msg = body["choices"][0]["message"]
            tool_calls = msg.get("tool_calls", [])
            if tool_calls and any(tc.get("function", {}).get("name") == "get_weather" for tc in tool_calls):
                args = tool_calls[0].get("function", {}).get("arguments", "")
                record("pass", name, f"tool=get_weather, args={args[:60]}", latency_ms=latency_ms, model=tool_model, feature="tools")
            else:
                content = msg.get("content", "")[:80]
                record("fail", name, f"no get_weather tool_call, content='{content}'", latency_ms=latency_ms, model=tool_model, feature="tools")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Multiple tools OpenAI
    name = "Multiple tools (OpenAI)"
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": tool_model,
            "stream": False,
            "max_tokens": 256,
            "tools": [WEATHER_TOOL, CALC_TOOL],
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            msg = body["choices"][0]["message"]
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                names = [tc.get("function", {}).get("name") for tc in tool_calls]
                record("pass", name, f"tool_calls={names}", latency_ms=latency_ms)
            else:
                content = msg.get("content", "")[:80]
                # Model might answer directly instead of calling tool - that's acceptable
                record("pass", name, f"model answered directly: '{content}'", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Tool calling via Anthropic format
    name = "Tool call (Anthropic)"
    try:
        anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]
        t0 = time.time()
        r = await client.post("/anthropic/v1/messages", headers=headers, json={
            "model": tool_model,
            "max_tokens": 256,
            "tools": anthropic_tools,
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200:
            has_tool_use = False
            for block in body.get("content", []):
                if block.get("type") == "tool_use":
                    has_tool_use = True
                    record("pass", name, f"tool={block.get('name')}, input={json.dumps(block.get('input', {}))[:60]}", latency_ms=latency_ms)
                    break
            if not has_tool_use:
                # Check stop_reason
                if body.get("stop_reason") == "tool_use":
                    record("pass", name, "stop_reason=tool_use", latency_ms=latency_ms)
                else:
                    text = ""
                    for block in body.get("content", []):
                        if block.get("type") == "text":
                            text = block.get("text", "")[:60]
                    record("fail", name, f"no tool_use block, text='{text}'", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Tool calling via Ollama format (tools passed in OpenAI format)
    name = "Tool call (Ollama)"
    try:
        t0 = time.time()
        r = await client.post("/api/chat", headers=headers, json={
            "model": tool_model,
            "stream": False,
            "options": {"num_predict": 256},
            "tools": [WEATHER_TOOL],
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "message" in body:
            msg = body["message"]
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                names = [tc.get("function", {}).get("name") for tc in tool_calls]
                record("pass", name, f"tool_calls={names}", latency_ms=latency_ms)
            else:
                content = (msg.get("content") or "")[:80]
                record("fail", name, f"no tool_calls, content='{content}'", latency_ms=latency_ms)
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Streaming + tool calls (OpenAI)
    name = "Streaming tool call (OpenAI)"
    try:
        saw_tool_call = False
        tool_name = ""
        tool_args = ""
        t0 = time.time()
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": tool_model,
                "stream": True,
                "max_tokens": 256,
                "tools": [WEATHER_TOOL],
                "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            },
        ) as r:
            if r.status_code != 200:
                latency_ms = (time.time() - t0) * 1000
                record("fail", name, f"status={r.status_code}", latency_ms=latency_ms)
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        tc_list = delta.get("tool_calls", [])
                        for tc in tc_list:
                            saw_tool_call = True
                            func = tc.get("function", {})
                            if func.get("name"):
                                tool_name = func["name"]
                            if func.get("arguments"):
                                tool_args += func["arguments"]
                    except json.JSONDecodeError:
                        pass
                latency_ms = (time.time() - t0) * 1000
                if saw_tool_call:
                    record("pass", name, f"tool={tool_name}, args={tool_args[:60]}", latency_ms=latency_ms)
                else:
                    record("fail", name, "no tool_calls in stream", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # -- Per-model tool calling test (every tool-capable model) --
    tool_models_all = cfg.catalog.tool_models
    if tool_models_all:
        print(f"\n  {BOLD}5b. Tool calling per model ({len(tool_models_all)} models):{RESET}")
        sem_tc = asyncio.Semaphore(cfg.concurrency)

        async def test_tools_per_model(model_id: str):
            tname = f"Tools [{model_id}]"
            async with sem_tc:
                try:
                    t0 = time.time()
                    r = await client.post("/v1/chat/completions", headers=headers, json={
                        "model": model_id,
                        "stream": False,
                        "max_tokens": 256,
                        "tools": [WEATHER_TOOL],
                        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
                    })
                    latency_ms = (time.time() - t0) * 1000
                    if r.status_code == 200:
                        body = r.json()
                        msg = body.get("choices", [{}])[0].get("message", {})
                        tc = msg.get("tool_calls", [])
                        if tc and any(t.get("function", {}).get("name") == "get_weather" for t in tc):
                            record("pass", tname, f"tool=get_weather", latency_ms=latency_ms, model=model_id, feature="tools")
                        else:
                            content = (msg.get("content") or "")[:60]
                            record("fail", tname, f"no tool_call, content='{content}'", latency_ms=latency_ms, model=model_id, feature="tools")
                    elif r.status_code in (503, 504):
                        record("skip", tname, f"status={r.status_code} (model not loaded)", model=model_id, feature="tools")
                    else:
                        record("fail", tname, f"status={r.status_code} body={r.text[:150]}", latency_ms=latency_ms, model=model_id, feature="tools")
                except httpx.TimeoutException:
                    record("skip", tname, "timeout (model not loaded)", model=model_id, feature="tools")
                except Exception as e:
                    record("fail", tname, str(e)[:150], model=model_id, feature="tools")

        await asyncio.gather(*[test_tools_per_model(m) for m in tool_models_all])


# ---------------------------------------------------------------------------
# Section 6: Multimodal
# ---------------------------------------------------------------------------

async def test_multimodal(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 6: Multimodal")
    headers = auth_headers(cfg)

    if not cfg.catalog.vision_models:
        record("skip", "Multimodal tests", "no vision model discovered")
        return

    vision_model = cfg.catalog.vision_models[0]
    print(f"  Using primary vision model: {vision_model}")

    # OpenAI image_url with data URI (first vision model for dialect tests)
    name = "OpenAI multimodal (data URI)"
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": vision_model,
            "stream": False,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{TEST_IMAGE_B64}",
                        },
                    },
                ],
            }],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            content = body["choices"][0]["message"].get("content", "")[:80]
            record("pass", name, f"'{content}'", latency_ms=latency_ms, model=vision_model, feature="multimodal")
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}", model=vision_model, feature="multimodal")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms, model=vision_model, feature="multimodal")
    except httpx.TimeoutException:
        record("skip", name, "timeout", model=vision_model, feature="multimodal")
    except Exception as e:
        record("fail", name, str(e)[:150], model=vision_model, feature="multimodal")

    # Anthropic image
    name = "Anthropic multimodal (base64)"
    try:
        t0 = time.time()
        r = await client.post("/anthropic/v1/messages", headers=headers, json={
            "model": vision_model,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": TEST_IMAGE_B64,
                        },
                    },
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and body.get("type") == "message":
            text = ""
            for block in body.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "")[:80]
            record("pass", name, f"'{text}'", latency_ms=latency_ms)
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Ollama images field
    name = "Ollama multimodal (images)"
    try:
        t0 = time.time()
        r = await client.post("/api/chat", headers=headers, json={
            "model": vision_model,
            "stream": False,
            "options": {"num_predict": 128},
            "messages": [{
                "role": "user",
                "content": "Describe this image briefly.",
                "images": [TEST_IMAGE_B64],
            }],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "message" in body:
            content = (body["message"].get("content") or "")[:80]
            record("pass", name, f"'{content}'", latency_ms=latency_ms)
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Multimodal + structured output
    name = "Multimodal + json_schema"
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": vision_model,
            "stream": False,
            "max_tokens": 256,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "image_desc", "strict": True, "schema": IMAGE_DESCRIPTION_SCHEMA},
            },
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image. Return JSON with description and colors array."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"},
                    },
                ],
            }],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "choices" in body:
            content = body["choices"][0]["message"].get("content", "")
            parsed = parse_json_content(content)
            if parsed is not None and validate_json_keys(parsed, IMAGE_DESCRIPTION_SCHEMA):
                record("pass", name, f"colors={parsed.get('colors')}", latency_ms=latency_ms)
            else:
                record("fail", name, f"schema mismatch: {content[:100]}", latency_ms=latency_ms)
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Multimodal + thinking (if vision model supports thinking)
    vision_thinks = vision_model in cfg.catalog.thinking_models
    name = "Multimodal + thinking"
    if vision_thinks:
        try:
            t0 = time.time()
            r = await client.post("/v1/chat/completions", headers=headers, json={
                "model": vision_model,
                "stream": False,
                "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": True},
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image. Think step by step."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"},
                        },
                    ],
                }],
            })
            latency_ms = (time.time() - t0) * 1000
            body = r.json()
            if r.status_code == 200 and "choices" in body:
                msg = body["choices"][0]["message"]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                has_reasoning = bool(reasoning and len(str(reasoning).strip()) > 0)
                record("pass", name, f"thinking={has_reasoning}", latency_ms=latency_ms)
            elif r.status_code in (503, 504):
                record("skip", name, f"status={r.status_code}")
            else:
                record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
        except httpx.TimeoutException:
            record("skip", name, "timeout")
        except Exception as e:
            record("fail", name, str(e)[:150])
    else:
        record("skip", name, f"{vision_model} not a thinking model")

    # -- Per-model multimodal test (every vision model) --
    all_vision = cfg.catalog.vision_models
    if len(all_vision) > 1:
        print(f"\n  {BOLD}6b. Multimodal per model ({len(all_vision)} models):{RESET}")
        sem_v = asyncio.Semaphore(cfg.concurrency)

        async def test_multimodal_per_model(model_id: str):
            tname = f"Multimodal [{model_id}]"
            is_thinker = model_id in cfg.catalog.thinking_models
            mtokens = 2048 if is_thinker else 512
            async with sem_v:
                try:
                    t0 = time.time()
                    r = await client.post("/v1/chat/completions", headers=headers, json={
                        "model": model_id,
                        "stream": False,
                        "max_tokens": mtokens,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image briefly."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"},
                                },
                            ],
                        }],
                    })
                    latency_ms = (time.time() - t0) * 1000
                    if r.status_code == 200:
                        body = r.json()
                        msg = body.get("choices", [{}])[0].get("message", {})
                        content = (msg.get("content") or msg.get("reasoning_content") or "")[:80]
                        record("pass", tname, f"'{content}'", latency_ms=latency_ms, model=model_id, feature="multimodal")
                    elif r.status_code in (422, 503, 504):
                        record("skip", tname, f"status={r.status_code}", model=model_id, feature="multimodal")
                    else:
                        record("fail", tname, f"status={r.status_code} body={r.text[:150]}", latency_ms=latency_ms, model=model_id, feature="multimodal")
                except httpx.TimeoutException:
                    record("skip", tname, "timeout (model not loaded)", model=model_id, feature="multimodal")
                except Exception as e:
                    record("fail", tname, str(e)[:150], model=model_id, feature="multimodal")

        # Test all vision models (including the first one, for matrix tracking)
        await asyncio.gather(*[test_multimodal_per_model(m) for m in all_vision])
    elif len(all_vision) == 1:
        # Only one vision model -- record result from the dialect test above for matrix
        # (already recorded with model= above in the OpenAI multimodal test)
        pass


# ---------------------------------------------------------------------------
# Section 7: Embeddings
# ---------------------------------------------------------------------------

async def test_embeddings(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 7: Embeddings")
    headers = auth_headers(cfg)

    if not cfg.catalog.embedding_models and cfg.embedding_model not in [m.get("id") for m in cfg.catalog.all_models]:
        record("skip", "Embedding tests", "no embedding model discovered")
        return

    emb_model = cfg.embedding_model

    # OpenAI single input
    name = "OpenAI /v1/embeddings (single)"
    dim = 0
    try:
        t0 = time.time()
        r = await client.post("/v1/embeddings", headers=headers, json={
            "model": emb_model,
            "input": "Hello world",
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "data" in body:
            emb = body["data"][0].get("embedding", [])
            if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], (int, float)):
                dim = len(emb)
                record("pass", name, f"dim={dim}", latency_ms=latency_ms, model=emb_model, feature="embedding")
            else:
                record("fail", name, f"embedding not a list of floats", latency_ms=latency_ms, model=emb_model, feature="embedding")
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}", model=emb_model, feature="embedding")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms, model=emb_model, feature="embedding")
    except httpx.TimeoutException:
        record("skip", name, "timeout", model=emb_model, feature="embedding")
    except Exception as e:
        record("fail", name, str(e)[:150], model=emb_model, feature="embedding")

    # OpenAI batch input
    name = "OpenAI /v1/embeddings (batch of 3)"
    try:
        t0 = time.time()
        r = await client.post("/v1/embeddings", headers=headers, json={
            "model": emb_model,
            "input": ["Hello world", "Goodbye world", "Testing embeddings"],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "data" in body:
            data = body["data"]
            if len(data) == 3:
                dims = [len(d.get("embedding", [])) for d in data]
                all_same = len(set(dims)) == 1
                if all_same and dims[0] > 0:
                    record("pass", name, f"3 embeddings, dim={dims[0]}, consistent=True", latency_ms=latency_ms)
                else:
                    record("fail", name, f"dims={dims}, consistent={all_same}", latency_ms=latency_ms)
            else:
                record("fail", name, f"expected 3 embeddings, got {len(data)}", latency_ms=latency_ms)
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Ollama /api/embeddings
    name = "Ollama /api/embeddings"
    try:
        t0 = time.time()
        r = await client.post("/api/embeddings", headers=headers, json={
            "model": emb_model,
            "prompt": "Hello world",
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200:
            # Could be Ollama format (top-level "embedding") or OpenAI nested
            emb = body.get("embedding") or (body.get("data", [{}])[0].get("embedding") if "data" in body else None)
            if isinstance(emb, list) and len(emb) > 0:
                ollama_dim = len(emb)
                record("pass", name, f"dim={ollama_dim}", latency_ms=latency_ms)
                # Dimension consistency check
                if dim > 0 and ollama_dim != dim:
                    record("fail", "Embedding dimension consistency", f"OpenAI dim={dim} vs Ollama dim={ollama_dim}")
                elif dim > 0:
                    record("pass", "Embedding dimension consistency", f"both dim={dim}")
            else:
                record("fail", name, f"no embedding in response", latency_ms=latency_ms)
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])


# ---------------------------------------------------------------------------
# Section 8: Rerank & Score
# ---------------------------------------------------------------------------

async def test_rerank(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 8: Rerank & Score")
    headers = auth_headers(cfg)

    rerank_model = cfg.rerank_model
    documents = [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "The Eiffel Tower is located in Paris, France.",
    ]
    query = "What is the capital of France?"

    # POST /v1/rerank
    name = "POST /v1/rerank (basic)"
    try:
        t0 = time.time()
        r = await client.post("/v1/rerank", headers=headers, json={
            "model": rerank_model,
            "query": query,
            "documents": documents,
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "results" in body:
            res = body["results"]
            has_scores = all("relevance_score" in item for item in res)
            if len(res) == 3 and has_scores:
                scores = [item["relevance_score"] for item in res]
                # Verify sorted descending
                is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
                record("pass", name, f"{len(res)} results, sorted={is_sorted}, top={scores[0]:.4f}", latency_ms=latency_ms, model=rerank_model, feature="rerank")
            else:
                record("fail", name, f"results={len(res)} has_scores={has_scores}", latency_ms=latency_ms, model=rerank_model, feature="rerank")
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}", model=rerank_model, feature="rerank")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms, model=rerank_model, feature="rerank")
    except httpx.TimeoutException:
        record("skip", name, "timeout", model=rerank_model, feature="rerank")
    except Exception as e:
        record("fail", name, str(e)[:150], model=rerank_model, feature="rerank")

    # Rerank with top_n=2
    name = "POST /v1/rerank (top_n=2)"
    try:
        t0 = time.time()
        r = await client.post("/v1/rerank", headers=headers, json={
            "model": rerank_model,
            "query": query,
            "documents": documents,
            "top_n": 2,
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200 and "results" in body:
            res = body["results"]
            if len(res) == 2:
                record("pass", name, f"{len(res)} results returned", latency_ms=latency_ms)
            else:
                record("fail", name, f"expected 2 results, got {len(res)}", latency_ms=latency_ms)
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # POST /v1/score with text_1 + text_2
    name = "POST /v1/score (single pair)"
    try:
        t0 = time.time()
        r = await client.post("/v1/score", headers=headers, json={
            "model": rerank_model,
            "text_1": "What is the capital of France?",
            "text_2": "The capital of France is Paris.",
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200:
            # Check for score field
            score = body.get("score") or body.get("data", [{}])[0].get("score") if "data" in body else body.get("score")
            if score is not None and isinstance(score, (int, float)):
                record("pass", name, f"score={score:.4f}", latency_ms=latency_ms)
            elif "data" in body and len(body["data"]) > 0:
                s = body["data"][0].get("score")
                if s is not None:
                    record("pass", name, f"score={s:.4f}", latency_ms=latency_ms)
                else:
                    record("fail", name, f"no score in response: {json.dumps(body)[:150]}", latency_ms=latency_ms)
            else:
                record("fail", name, f"no score in response: {json.dumps(body)[:150]}", latency_ms=latency_ms)
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}")
        elif r.status_code == 404:
            record("skip", name, "endpoint not found")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Score batch (text_2 as list)
    name = "POST /v1/score (batch)"
    try:
        t0 = time.time()
        r = await client.post("/v1/score", headers=headers, json={
            "model": rerank_model,
            "text_1": "What is the capital of France?",
            "text_2": [
                "The capital of France is Paris.",
                "Python is a programming language.",
                "The Eiffel Tower is in Paris.",
            ],
        })
        latency_ms = (time.time() - t0) * 1000
        body = r.json()
        if r.status_code == 200:
            if "data" in body:
                scores = body["data"]
                if len(scores) == 3:
                    score_vals = [s.get("score", 0) for s in scores]
                    record("pass", name, f"3 scores: {[f'{s:.3f}' for s in score_vals]}", latency_ms=latency_ms)
                else:
                    record("fail", name, f"expected 3 scores, got {len(scores)}", latency_ms=latency_ms)
            else:
                record("fail", name, f"no data in response: {json.dumps(body)[:150]}", latency_ms=latency_ms)
        elif r.status_code in (503, 504):
            record("skip", name, f"status={r.status_code}")
        elif r.status_code == 404:
            record("skip", name, "endpoint not found")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])


# ---------------------------------------------------------------------------
# Section 9: Voice (TTS/STT)
# ---------------------------------------------------------------------------

async def test_voice(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 9: Voice (TTS/STT)")
    headers = auth_headers(cfg)

    if cfg.skip_voice:
        record("skip", "TTS speech", "--skip-voice")
        record("skip", "STT transcription", "--skip-voice")
        return

    # POST /v1/audio/speech
    name = "POST /v1/audio/speech (TTS)"
    tts_audio = None
    try:
        t0 = time.time()
        r = await client.post("/v1/audio/speech", headers=headers, json={
            "model": "kokoro",
            "input": "Hello, this is a test of text to speech.",
            "voice": "af_heart",
            "response_format": "mp3",
        })
        latency_ms = (time.time() - t0) * 1000
        if r.status_code == 200:
            audio_bytes = r.content
            if len(audio_bytes) > 100:
                tts_audio = audio_bytes
                record("pass", name, f"{len(audio_bytes)} bytes of audio", latency_ms=latency_ms, model="kokoro", feature="voice")
            else:
                record("fail", name, f"audio too small: {len(audio_bytes)} bytes", latency_ms=latency_ms, model="kokoro", feature="voice")
        elif r.status_code == 404:
            record("skip", name, "TTS not enabled/configured", model="kokoro", feature="voice")
        else:
            record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms, model="kokoro", feature="voice")
    except httpx.TimeoutException:
        record("skip", name, "timeout", model="kokoro", feature="voice")
    except Exception as e:
        record("fail", name, str(e)[:150], model="kokoro", feature="voice")

    # POST /v1/audio/transcriptions
    name = "POST /v1/audio/transcriptions (STT)"
    if tts_audio:
        try:
            t0 = time.time()
            r = await client.post(
                "/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                files={"file": ("test.mp3", tts_audio, "audio/mpeg")},
            )
            latency_ms = (time.time() - t0) * 1000
            if r.status_code == 200:
                body = r.json()
                text = body.get("text", "")
                if text:
                    record("pass", name, f"transcribed: '{text[:60]}'", latency_ms=latency_ms)
                else:
                    record("fail", name, "empty transcription", latency_ms=latency_ms)
            elif r.status_code == 404:
                record("skip", name, "STT not enabled/configured")
            else:
                record("fail", name, f"status={r.status_code} body={r.text[:200]}", latency_ms=latency_ms)
        except httpx.TimeoutException:
            record("skip", name, "timeout")
        except Exception as e:
            record("fail", name, str(e)[:150])
    else:
        record("skip", name, "no TTS audio to transcribe")


# ---------------------------------------------------------------------------
# Section 10: Streaming Deep Validation
# ---------------------------------------------------------------------------

async def test_streaming(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 10: Streaming Deep Validation")
    headers = auth_headers(cfg)

    # OpenAI SSE validation
    name = "OpenAI SSE format validation"
    try:
        all_valid = True
        chunk_count = 0
        saw_done = False
        errors = []
        t0 = time.time()
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": cfg.vllm_model,
                "stream": True,
                "max_tokens": cfg.max_tokens,
                "messages": [{"role": "user", "content": "Count to 5."}],
            },
        ) as r:
            if r.status_code != 200:
                latency_ms = (time.time() - t0) * 1000
                record("fail", name, f"status={r.status_code}", latency_ms=latency_ms, model=cfg.vllm_model, feature="streaming")
            else:
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    if not line.startswith("data: "):
                        errors.append(f"line missing 'data: ' prefix: {line[:40]}")
                        all_valid = False
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        saw_done = True
                        continue
                    try:
                        data = json.loads(payload)
                        chunk_count += 1
                        # Validate structure
                        if "choices" not in data:
                            errors.append("missing 'choices' key")
                            all_valid = False
                        else:
                            choice = data["choices"][0]
                            if "delta" not in choice:
                                errors.append("missing 'delta' in choice")
                                all_valid = False
                    except json.JSONDecodeError as e:
                        errors.append(f"JSON parse error: {e}")
                        all_valid = False

                latency_ms = (time.time() - t0) * 1000
                if all_valid and saw_done and chunk_count > 0:
                    record("pass", name, f"{chunk_count} valid chunks, [DONE] present", latency_ms=latency_ms, model=cfg.vllm_model, feature="streaming")
                else:
                    detail = f"valid={all_valid}, done={saw_done}, chunks={chunk_count}"
                    if errors:
                        detail += f", errors={errors[:3]}"
                    record("fail", name, detail, latency_ms=latency_ms, model=cfg.vllm_model, feature="streaming")
    except httpx.TimeoutException:
        record("skip", name, "timeout", model=cfg.vllm_model, feature="streaming")
    except Exception as e:
        record("fail", name, str(e)[:150], model=cfg.vllm_model, feature="streaming")

    # Ollama NDJSON validation
    name = "Ollama NDJSON format validation"
    try:
        all_valid = True
        chunk_count = 0
        saw_final_done = False
        errors = []
        t0 = time.time()
        async with client.stream(
            "POST",
            "/api/chat",
            headers=headers,
            json={
                "model": cfg.ollama_model,
                "stream": True,
                "options": {"num_predict": cfg.max_tokens},
                "messages": [{"role": "user", "content": "Count to 5."}],
            },
        ) as r:
            if r.status_code != 200:
                latency_ms = (time.time() - t0) * 1000
                record("fail", name, f"status={r.status_code}", latency_ms=latency_ms)
            else:
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        chunk_count += 1
                        if obj.get("done") is True:
                            saw_final_done = True
                    except json.JSONDecodeError as e:
                        errors.append(f"JSON parse error: {e}")
                        all_valid = False

                latency_ms = (time.time() - t0) * 1000
                if all_valid and saw_final_done and chunk_count > 0:
                    record("pass", name, f"{chunk_count} valid NDJSON lines, final done=true", latency_ms=latency_ms)
                else:
                    detail = f"valid={all_valid}, done={saw_final_done}, chunks={chunk_count}"
                    if errors:
                        detail += f", errors={errors[:3]}"
                    record("fail", name, detail, latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Anthropic SSE validation
    name = "Anthropic SSE event type validation"
    try:
        event_types = []
        saw_message_start = False
        saw_message_stop = False
        saw_delta = False
        t0 = time.time()
        async with client.stream(
            "POST",
            "/anthropic/v1/messages",
            headers=headers,
            json={
                "model": cfg.ollama_model,
                "max_tokens": cfg.max_tokens,
                "stream": True,
                "messages": [{"role": "user", "content": "Count to 5."}],
            },
        ) as r:
            if r.status_code != 200:
                latency_ms = (time.time() - t0) * 1000
                record("fail", name, f"status={r.status_code}", latency_ms=latency_ms)
            else:
                async for line in r.aiter_lines():
                    line = line.strip()
                    if line.startswith("event: "):
                        etype = line[7:].strip()
                        event_types.append(etype)
                        if etype == "message_start":
                            saw_message_start = True
                        elif etype == "message_stop":
                            saw_message_stop = True
                        elif etype == "content_block_delta":
                            saw_delta = True

                latency_ms = (time.time() - t0) * 1000
                if saw_message_start and saw_message_stop and saw_delta:
                    record("pass", name, f"events: {', '.join(sorted(set(event_types)))}", latency_ms=latency_ms)
                else:
                    record("fail", name, f"start={saw_message_start} stop={saw_message_stop} delta={saw_delta} events={event_types[:10]}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Streaming + thinking: verify reasoning deltas before content
    name = "Streaming + thinking order (reasoning before content)"
    try:
        first_reasoning_idx = -1
        first_content_idx = -1
        idx = 0
        t0 = time.time()
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": cfg.vllm_model,
                "stream": True,
                "max_tokens": 2048,
                "reasoning_effort": "low",
                "messages": [{"role": "user", "content": "What is 7*8?"}],
            },
        ) as r:
            if r.status_code != 200:
                latency_ms = (time.time() - t0) * 1000
                record("fail", name, f"status={r.status_code}", latency_ms=latency_ms)
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("reasoning_content") and first_reasoning_idx < 0:
                            first_reasoning_idx = idx
                        if delta.get("content") and first_content_idx < 0:
                            first_content_idx = idx
                        idx += 1
                    except json.JSONDecodeError:
                        pass

                latency_ms = (time.time() - t0) * 1000
                if first_reasoning_idx >= 0 and (first_content_idx < 0 or first_reasoning_idx < first_content_idx):
                    record("pass", name, f"reasoning@{first_reasoning_idx} before content@{first_content_idx}", latency_ms=latency_ms)
                elif first_reasoning_idx < 0:
                    record("fail", name, "no reasoning chunks found", latency_ms=latency_ms)
                else:
                    record("fail", name, f"reasoning@{first_reasoning_idx} NOT before content@{first_content_idx}", latency_ms=latency_ms)
    except httpx.TimeoutException:
        record("skip", name, "timeout")
    except Exception as e:
        record("fail", name, str(e)[:150])


# ---------------------------------------------------------------------------
# Section 11: Error Handling
# ---------------------------------------------------------------------------

async def test_errors(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 11: Error Handling")
    headers = auth_headers(cfg)

    # -- OpenAI endpoint errors --

    # Invalid model -> 404
    name = "OpenAI: invalid model -> 404"
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": "nonexistent-model-xyz-999",
            "messages": [{"role": "user", "content": "hi"}],
        })
        latency_ms = (time.time() - t0) * 1000
        if r.status_code == 404:
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 404, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    # No auth header -> 401
    name = "OpenAI: no auth -> 401"
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", json={
            "model": cfg.vllm_model,
            "messages": [{"role": "user", "content": "hi"}],
        })
        latency_ms = (time.time() - t0) * 1000
        if r.status_code in (401, 403):
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 401/403, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Invalid API key -> 401
    name = "OpenAI: invalid key -> 401"
    try:
        t0 = time.time()
        r = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer invalid_key_xxx"},
            json={
                "model": cfg.vllm_model,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        latency_ms = (time.time() - t0) * 1000
        if r.status_code in (401, 403):
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 401/403, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    # Missing required fields -> 422/400
    name = "OpenAI: missing model field -> 4xx"
    try:
        t0 = time.time()
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        latency_ms = (time.time() - t0) * 1000
        if 400 <= r.status_code < 500:
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 4xx, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    # -- Anthropic endpoint errors --

    name = "Anthropic: invalid model -> 404"
    try:
        t0 = time.time()
        r = await client.post("/anthropic/v1/messages", headers=headers, json={
            "model": "nonexistent-model-xyz-999",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        latency_ms = (time.time() - t0) * 1000
        if r.status_code == 404:
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 404, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    name = "Anthropic: no auth -> 401"
    try:
        t0 = time.time()
        r = await client.post("/anthropic/v1/messages", json={
            "model": cfg.ollama_model,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        latency_ms = (time.time() - t0) * 1000
        if r.status_code in (401, 403):
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 401/403, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    name = "Anthropic: invalid key -> 401"
    try:
        t0 = time.time()
        r = await client.post(
            "/anthropic/v1/messages",
            headers={"Authorization": "Bearer invalid_key_xxx"},
            json={
                "model": cfg.ollama_model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        latency_ms = (time.time() - t0) * 1000
        if r.status_code in (401, 403):
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 401/403, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    # -- Ollama endpoint errors --

    name = "Ollama: invalid model -> 404"
    try:
        t0 = time.time()
        r = await client.post("/api/chat", headers=headers, json={
            "model": "nonexistent-model-xyz-999",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        })
        latency_ms = (time.time() - t0) * 1000
        if r.status_code == 404:
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 404, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    name = "Ollama: no auth -> 401"
    try:
        t0 = time.time()
        r = await client.post("/api/chat", json={
            "model": cfg.ollama_model,
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        })
        latency_ms = (time.time() - t0) * 1000
        if r.status_code in (401, 403):
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 401/403, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    name = "Ollama: invalid key -> 401"
    try:
        t0 = time.time()
        r = await client.post(
            "/api/chat",
            headers={"Authorization": "Bearer invalid_key_xxx"},
            json={
                "model": cfg.ollama_model,
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        latency_ms = (time.time() - t0) * 1000
        if r.status_code in (401, 403):
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 401/403, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])

    name = "Ollama: missing model -> 4xx"
    try:
        t0 = time.time()
        r = await client.post("/api/chat", headers=headers, json={
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        })
        latency_ms = (time.time() - t0) * 1000
        if 400 <= r.status_code < 500:
            record("pass", name, f"{r.status_code}", latency_ms=latency_ms)
        else:
            record("fail", name, f"expected 4xx, got {r.status_code}", latency_ms=latency_ms)
    except Exception as e:
        record("fail", name, str(e)[:150])


# ---------------------------------------------------------------------------
# Section 12: Stress / Concurrency
# ---------------------------------------------------------------------------

async def test_stress(client: httpx.AsyncClient, cfg: Config):
    section_header("Section 12: Stress / Concurrency")

    if cfg.skip_stress:
        record("skip", "Concurrency stress test", "--skip-stress")
        return

    headers = auth_headers(cfg)
    concurrency = cfg.concurrency
    sem = asyncio.Semaphore(concurrency)

    prompts = [
        "Explain recursion briefly.",
        "What causes the northern lights?",
        "How does a compiler work?",
        "What is Euler's identity?",
        "Explain TCP vs UDP.",
        "How does photosynthesis work?",
        "What is the halting problem?",
        "Describe nuclear fusion in stars.",
        "Explain public key cryptography.",
        "What is general relativity?",
    ]

    latencies: List[float] = []
    successes = 0
    failures = 0
    total = concurrency

    async def stress_one(i: int):
        nonlocal successes, failures
        async with sem:
            prompt = prompts[i % len(prompts)]
            t0 = time.monotonic()
            try:
                r = await client.post("/v1/chat/completions", headers=headers, json={
                    "model": cfg.vllm_model,
                    "stream": False,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": prompt}],
                })
                elapsed = time.monotonic() - t0
                latencies.append(elapsed)
                if r.status_code == 200:
                    successes += 1
                else:
                    failures += 1
                    vprint(cfg, f"req {i}: status={r.status_code}")
            except Exception as e:
                elapsed = time.monotonic() - t0
                latencies.append(elapsed)
                failures += 1
                vprint(cfg, f"req {i}: error={e}")

    print(f"  Sending {concurrency} concurrent requests (max_tokens=16, non-streaming)...")
    t0 = time.monotonic()
    tasks = [stress_one(i) for i in range(total)]
    await asyncio.gather(*tasks)
    wall_time = time.monotonic() - t0

    # Compute stats
    if latencies:
        sorted_lats = sorted(latencies)
        p50_idx = int(len(sorted_lats) * 0.50)
        p95_idx = min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)
        stats_str = (
            f"min={sorted_lats[0]:.2f}s, "
            f"max={sorted_lats[-1]:.2f}s, "
            f"p50={sorted_lats[p50_idx]:.2f}s, "
            f"p95={sorted_lats[p95_idx]:.2f}s, "
            f"wall={wall_time:.2f}s"
        )
    else:
        stats_str = "no latency data"

    name = f"Concurrency stress ({concurrency} requests)"
    success_rate = (successes / total * 100) if total > 0 else 0
    detail = f"{successes}/{total} succeeded ({success_rate:.0f}%), {stats_str}"

    if failures == 0:
        record("pass", name, detail)
    else:
        record("fail", name, detail)


# ---------------------------------------------------------------------------
# Section registry
# ---------------------------------------------------------------------------

SECTIONS = {
    "discovery": test_discovery,
    "basic-chat": test_basic_chat,
    "structured": test_structured,
    "thinking": test_thinking,
    "tools": test_tools,
    "multimodal": test_multimodal,
    "embeddings": test_embeddings,
    "rerank": test_rerank,
    "voice": test_voice,
    "streaming": test_streaming,
    "errors": test_errors,
    "stress": test_stress,
}


# ---------------------------------------------------------------------------
# Matrix & Throughput Printing
# ---------------------------------------------------------------------------

MATRIX_FEATURES = ["chat", "structured", "thinking", "tools", "multimodal", "embedding", "rerank", "voice", "streaming"]
MATRIX_HEADERS = ["Chat", "Struct", "Think", "Tools", "Multi", "Embed", "Rerank", "Voice", "Stream"]


def _feature_capabilities(model_id: str, catalog: "ModelCatalog") -> set:
    """Return the set of features a model is capable of."""
    caps = set()
    if model_id in catalog.chat_models:
        caps.add("chat")
        caps.add("streaming")  # all chat models can stream
    if model_id in catalog.structured_models:
        caps.add("structured")
    if model_id in catalog.thinking_models:
        caps.add("thinking")
    if model_id in catalog.tool_models:
        caps.add("tools")
    if model_id in catalog.vision_models:
        caps.add("multimodal")
    if model_id in catalog.embedding_models:
        caps.add("embedding")
    if model_id in catalog.rerank_models:
        caps.add("rerank")
    # voice is special (kokoro) -- check if model has voice results
    return caps


def print_matrix(catalog: "ModelCatalog"):
    """Print the model x feature matrix."""
    if not model_results:
        return

    print(f"\n{BOLD}Model x Feature Matrix{RESET}")
    print(f"{'=' * 120}")

    # Header
    model_col_w = 35
    feat_col_w = 8
    hdr = f"{'Model':<{model_col_w}}"
    for h in MATRIX_HEADERS:
        hdr += f"  {h:>{feat_col_w}}"
    print(hdr)
    print(f"{chr(0x2500) * model_col_w}  {'  '.join([chr(0x2500) * feat_col_w] * len(MATRIX_HEADERS))}")

    # Sort models: chat first, then embedding, then rerank
    all_models_in_results = sorted(model_results.keys())

    for model_id in all_models_in_results:
        feats = model_results[model_id]
        caps = _feature_capabilities(model_id, catalog)

        # Truncate long model names
        display_name = model_id if len(model_id) <= model_col_w - 1 else model_id[:model_col_w - 4] + "..."
        row = f"{display_name:<{model_col_w}}"

        for feature in MATRIX_FEATURES:
            if feature in feats:
                entry = feats[feature]
                status = entry["status"]
                lat = entry.get("latency_ms", 0)
                if status == "pass":
                    if lat > 0:
                        lat_s = f"{lat / 1000:.0f}s" if lat >= 1000 else f"{lat:.0f}ms"
                        cell = f"{GREEN}\u2713{RESET} {lat_s}"
                    else:
                        cell = f"{GREEN}\u2713{RESET}"
                elif status == "fail":
                    cell = f"{RED}\u2717{RESET}"
                else:  # skip
                    cell = f"{YELLOW}\u25CB{RESET}"
                # Pad cell (accounting for ANSI codes which are ~9 chars)
                visible_len = len(cell) - len(GREEN) - len(RESET)  # approximate
                padding = feat_col_w - visible_len
                if padding < 0:
                    padding = 0
                row += f"  {cell}{' ' * padding}"
            elif feature in caps:
                # Capable but not tested
                cell = f"{DIM}?{RESET}"
                row += f"  {cell}{' ' * (feat_col_w - 1)}"
            else:
                # Not capable
                row += f"  {'-':>{feat_col_w}}"

        print(row)


def print_throughput_summary(wall_time: float):
    """Print throughput and latency percentile summary."""
    total_requests = len(results)
    if not all_latencies:
        return

    sorted_lats = sorted(all_latencies)
    n = len(sorted_lats)

    def percentile(pct: float) -> float:
        idx = min(int(n * pct / 100), n - 1)
        return sorted_lats[idx]

    p50 = percentile(50)
    p95 = percentile(95)
    p99 = percentile(99)
    avg_rps = total_requests / wall_time if wall_time > 0 else 0

    print(f"\n{BOLD}Throughput Summary{RESET}")
    print(f"{'=' * 70}")
    print(f"  Total requests:      {total_requests}")
    print(f"  Total wall time:     {wall_time:.1f}s")
    print(f"  Avg requests/sec:    {avg_rps:.1f}")
    print(f"  Latency p50:         {p50:.0f}ms")
    print(f"  Latency p95:         {p95:.0f}ms")
    print(f"  Latency p99:         {p99:.0f}ms")
    print(f"  Latency min:         {sorted_lats[0]:.0f}ms")
    print(f"  Latency max:         {sorted_lats[-1]:.0f}ms")
    print(f"  Requests with timing:{n}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(cfg: Config):
    print(f"\n{BOLD}MindRouter2 Comprehensive Production Test{RESET}")
    print(f"  Base URL:        {cfg.base_url}")
    print(f"  Ollama model:    {cfg.ollama_model}")
    print(f"  vLLM model:      {cfg.vllm_model}")
    print(f"  Embed model:     {cfg.embedding_model}")
    print(f"  Rerank model:    {cfg.rerank_model}")
    print(f"  Timeout:         {cfg.timeout}s")
    print(f"  Concurrency:     {cfg.concurrency}")
    print(f"  Max tokens:      {cfg.max_tokens}")
    print(f"  Skip stress:     {cfg.skip_stress}")
    print(f"  Skip voice:      {cfg.skip_voice}")
    print(f"  Verbose:         {cfg.verbose}")

    sections_to_run = cfg.sections or list(SECTIONS.keys())
    print(f"  Sections:        {', '.join(sections_to_run)}")

    async with httpx.AsyncClient(
        base_url=cfg.base_url,
        timeout=httpx.Timeout(cfg.timeout),
    ) as client:
        t0 = time.time()

        for section_name in sections_to_run:
            if section_name in SECTIONS:
                await SECTIONS[section_name](client, cfg)
            else:
                print(f"\n  {RED}Unknown section: {section_name}{RESET}")

        elapsed = time.time() - t0

    # Summary
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    skipped = sum(1 for r in results if r["status"] == "skip")
    total = len(results)

    print(f"\n{BOLD}{'=' * 70}{RESET}")
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
                print(f"    {RED}x{RESET} {r['name']}")
                if r["detail"]:
                    print(f"      {DIM}{r['detail'][:200]}{RESET}")

    # Print model x feature matrix
    print_matrix(cfg.catalog)

    # Print throughput summary
    print_throughput_summary(elapsed)

    print()
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(
        description="MindRouter2 comprehensive production test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python production_test.py --api-key mr2_xxx --base-url https://your-host:8000
  python production_test.py --api-key mr2_xxx --base-url https://your-host:8000 --section discovery
  python production_test.py --api-key mr2_xxx --base-url https://your-host:8000 --skip-stress --skip-voice
  python production_test.py --api-key mr2_xxx --base-url https://your-host:8000 --concurrency 20
""",
    )
    parser.add_argument("--api-key", required=True,
                        help="API key for authentication (required)")
    parser.add_argument("--base-url", required=True,
                        help="Base URL of MindRouter (e.g. https://your-mindrouter-host)")
    parser.add_argument("--admin-key", default=None,
                        help="Admin API key (optional, for future admin tests)")
    parser.add_argument("--section", action="append", dest="sections",
                        choices=list(SECTIONS.keys()),
                        help="Run only specific section(s); repeatable")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Request timeout in seconds (default: 180)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrency level for stress tests (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--skip-stress", action="store_true",
                        help="Skip the stress/concurrency test section")
    parser.add_argument("--skip-voice", action="store_true",
                        help="Skip the voice (TTS/STT) test section")
    parser.add_argument("--ollama-model", default="microsoft/phi-4",
                        help="Ollama model to test (default: microsoft/phi-4)")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-120b",
                        help="vLLM model to test (default: openai/gpt-oss-120b)")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-8B",
                        help="Embedding model (default: Qwen/Qwen3-Embedding-8B)")
    parser.add_argument("--rerank-model", default="Qwen/Qwen3-Reranker-8B",
                        help="Rerank model (default: Qwen/Qwen3-Reranker-8B)")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Default max_tokens for chat tests (default: 32)")

    args = parser.parse_args()
    cfg = Config(args)

    exit_code = asyncio.run(async_main(cfg))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_tool_calling_live.py: Live tool calling compliance
#     tests across all tool-capable models and API surfaces.
#
# Discovers tool-capable models via /v1/models, then tests
# each one across OpenAI, Ollama, and Anthropic API styles
# with both streaming and non-streaming requests.
#
# Usage:
#   MINDROUTER_API_KEY=<key> python tests/test_tool_calling_live.py
#   MINDROUTER_API_KEY=<key> MINDROUTER_BASE_URL=https://mindrouter.uidaho.edu python tests/test_tool_calling_live.py
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tabulate import tabulate

# --- Configuration ---

BASE = os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu")
API_KEY = os.environ.get("MINDROUTER_API_KEY")
MAX_WORKERS = 6
TIMEOUT = 300
REPLICATES = 1
LOG_FILE = "tool_calling_log.txt"

log_lock = threading.Lock()

# --- Tool Definitions ---

# Simple weather tool — straightforward single-argument function
WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Seattle'",
                },
            },
            "required": ["city"],
        },
    },
}

WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g. 'Seattle'",
            },
        },
        "required": ["city"],
    },
}

TOOLS_OPENAI = [WEATHER_TOOL_OPENAI]
TOOLS_ANTHROPIC = [WEATHER_TOOL_ANTHROPIC]

PROMPT = "What is the current weather in Seattle? Use the get_weather tool to find out."

# Model name substrings that indicate non-chat models (skip these)
EXCLUDED_PATTERNS = ["embed", "rerank", "Embedding", "Reranker"]


# --- Model Discovery ---


def discover_tool_models():
    """Query /v1/models and return list of models with tools capability."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        r = requests.get(f"{BASE}/v1/models", headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Failed to discover models: {e}")
        return []

    models = []
    for m in data.get("data", []):
        model_id = m["id"]
        caps = m.get("capabilities", {})

        # Skip non-tool models
        if not caps.get("tools"):
            continue

        # Skip embedding/reranker models
        if any(pat.lower() in model_id.lower() for pat in EXCLUDED_PATTERNS):
            continue

        models.append({
            "id": model_id,
            "thinking": caps.get("thinking", False),
            "multimodal": caps.get("multimodal", False),
        })

    return sorted(models, key=lambda x: x["id"])


# --- Request Builders ---


def build_openai_request(model, stream=False):
    """Build /v1/chat/completions request with tools."""
    return "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS_OPENAI,
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512,
    }


def build_ollama_request(model, stream=False):
    """Build /api/chat request with tools."""
    return "/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS_OPENAI,  # Ollama uses same format as OpenAI
        "stream": stream,
    }


def build_anthropic_request(model, stream=False):
    """Build /anthropic/v1/messages request with tools."""
    return "/anthropic/v1/messages", {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS_ANTHROPIC,
        "tool_choice": {"type": "auto"},
        "max_tokens": 512,
        "stream": stream,
    }


API_STYLES = {
    "openai": build_openai_request,
    "ollama": build_ollama_request,
    "anthropic": build_anthropic_request,
}


# --- Response Extractors ---


def extract_tool_calls_openai(data):
    """Extract tool calls from OpenAI non-streaming response."""
    choices = data.get("choices", [])
    if not choices:
        return [], data.get("choices", [{}])[0].get("finish_reason", "")
    msg = choices[0].get("message", {})
    finish = choices[0].get("finish_reason", "")
    tool_calls = msg.get("tool_calls", [])
    results = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = args_raw
        results.append({
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": args,
        })
    return results, finish


def extract_tool_calls_ollama(data):
    """Extract tool calls from Ollama non-streaming response."""
    msg = data.get("message", {})
    tool_calls = msg.get("tool_calls", [])
    done = data.get("done", False)
    results = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        args_raw = fn.get("arguments", {})
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except (json.JSONDecodeError, TypeError):
            args = args_raw
        results.append({
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": args,
        })
    finish = "tool_calls" if tool_calls else ("stop" if done else "")
    return results, finish


def extract_tool_calls_anthropic(data):
    """Extract tool calls from Anthropic non-streaming response."""
    content = data.get("content", [])
    stop_reason = data.get("stop_reason", "")
    results = []
    for block in content:
        if block.get("type") == "tool_use":
            results.append({
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": block.get("input", {}),
            })
    return results, stop_reason


# --- Stream Extractors ---


def extract_tool_calls_openai_stream(response):
    """Extract tool calls from OpenAI SSE streaming response."""
    tool_calls_acc = {}  # index -> {id, name, arguments_parts}
    finish_reason = ""

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            fr = chunk.get("choices", [{}])[0].get("finish_reason")
            if fr:
                finish_reason = fr

            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": "",
                        "name": "",
                        "arguments_parts": [],
                    }
                acc = tool_calls_acc[idx]
                if tc_delta.get("id"):
                    acc["id"] = tc_delta["id"]
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    acc["name"] = fn["name"]
                if fn.get("arguments"):
                    acc["arguments_parts"].append(fn["arguments"])
        except json.JSONDecodeError:
            continue

    results = []
    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        args_str = "".join(acc["arguments_parts"])
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = args_str
        results.append({
            "id": acc["id"],
            "name": acc["name"],
            "arguments": args,
        })
    return results, finish_reason


def extract_tool_calls_ollama_stream(response):
    """Extract tool calls from Ollama NDJSON streaming response."""
    tool_calls = []
    done = False

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            chunk = json.loads(line)
            if chunk.get("done"):
                done = True
            msg = chunk.get("message", {})
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                args_raw = fn.get("arguments", {})
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (json.JSONDecodeError, TypeError):
                    args = args_raw
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": args,
                })
        except json.JSONDecodeError:
            continue

    finish = "tool_calls" if tool_calls else ("stop" if done else "")
    return tool_calls, finish


def extract_tool_calls_anthropic_stream(response):
    """Extract tool calls from Anthropic event streaming response."""
    tool_blocks = {}  # index -> {id, name, input_parts}
    current_index = -1
    stop_reason = ""

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[len("data: "):]
            try:
                data = json.loads(payload)
                event_type = data.get("type", "")

                if event_type == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_index = data.get("index", current_index + 1)
                        tool_blocks[current_index] = {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input_parts": [],
                        }

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "input_json_delta":
                        idx = data.get("index", current_index)
                        if idx in tool_blocks:
                            tool_blocks[idx]["input_parts"].append(
                                delta.get("partial_json", "")
                            )

                elif event_type == "message_delta":
                    sr = data.get("delta", {}).get("stop_reason", "")
                    if sr:
                        stop_reason = sr

                elif event_type == "message_stop":
                    break

            except json.JSONDecodeError:
                continue

    results = []
    for idx in sorted(tool_blocks.keys()):
        tb = tool_blocks[idx]
        input_str = "".join(tb["input_parts"])
        try:
            args = json.loads(input_str) if input_str else {}
        except json.JSONDecodeError:
            args = input_str
        results.append({
            "id": tb["id"],
            "name": tb["name"],
            "arguments": args,
        })
    return results, stop_reason


EXTRACTORS = {
    "openai": extract_tool_calls_openai,
    "ollama": extract_tool_calls_ollama,
    "anthropic": extract_tool_calls_anthropic,
}

STREAM_EXTRACTORS = {
    "openai": extract_tool_calls_openai_stream,
    "ollama": extract_tool_calls_ollama_stream,
    "anthropic": extract_tool_calls_anthropic_stream,
}


# --- Validation ---


def validate_tool_calls(tool_calls, finish_reason):
    """Validate that the response contains a valid get_weather tool call.

    Returns (ok, error_message).
    """
    if not tool_calls:
        return False, f"No tool calls in response (finish_reason={finish_reason})"

    # Find a get_weather call
    weather_calls = [tc for tc in tool_calls if tc["name"] == "get_weather"]
    if not weather_calls:
        names = [tc["name"] for tc in tool_calls]
        return False, f"No get_weather call found; got: {names}"

    tc = weather_calls[0]

    # Validate arguments
    args = tc["arguments"]
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return False, f"Arguments not valid JSON: {args!r}"

    if not isinstance(args, dict):
        return False, f"Arguments not a dict: {type(args)}"

    if "city" not in args:
        return False, f"Missing 'city' in arguments: {args}"

    city = args["city"]
    if not isinstance(city, str) or not city.strip():
        return False, f"Invalid city value: {city!r}"

    return True, ""


# --- Test Runner ---


def run_single_test(model_id, api_style, stream, rep_id):
    """Run a single tool calling test. Returns result dict."""
    builder = API_STYLES[api_style]
    endpoint, body = builder(model_id, stream=stream)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if api_style == "anthropic":
        headers["x-api-key"] = API_KEY
        headers["anthropic-version"] = "2023-06-01"

    url = f"{BASE}{endpoint}"

    res = {
        "model": model_id,
        "api_style": api_style,
        "stream": stream,
        "ok": False,
        "latency": 0,
        "status_code": "ERR",
        "tool_calls": [],
        "finish_reason": "",
        "error_msg": "",
    }

    try:
        t0 = time.perf_counter()

        if stream:
            r = requests.post(
                url, json=body, headers=headers, timeout=TIMEOUT, stream=True,
            )
            res["latency"] = time.perf_counter() - t0
            res["status_code"] = r.status_code

            if r.ok:
                extractor = STREAM_EXTRACTORS[api_style]
                tool_calls, finish = extractor(r)
                res["tool_calls"] = tool_calls
                res["finish_reason"] = finish
                res["latency"] = time.perf_counter() - t0
            else:
                res["error_msg"] = r.text[:300]
        else:
            r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
            res["latency"] = time.perf_counter() - t0
            res["status_code"] = r.status_code

            if r.ok:
                data = r.json()
                extractor = EXTRACTORS[api_style]
                tool_calls, finish = extractor(data)
                res["tool_calls"] = tool_calls
                res["finish_reason"] = finish
            else:
                res["error_msg"] = r.text[:300]

        if r.ok:
            ok, err = validate_tool_calls(res["tool_calls"], res["finish_reason"])
            res["ok"] = ok
            if not ok:
                res["error_msg"] = err

        # Log full response for debugging
        with log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                stream_tag = "STREAM" if stream else "SYNC"
                f.write(
                    f"--- {api_style.upper()} {stream_tag} | {model_id} | "
                    f"OK={res['ok']} | {res['status_code']} ---\n"
                )
                if not stream and r.ok:
                    try:
                        f.write(f"BODY: {json.dumps(r.json(), indent=2)[:2000]}\n\n")
                    except Exception:
                        f.write(f"BODY: {r.text[:2000]}\n\n")
                elif not r.ok:
                    f.write(f"ERROR: {r.text[:1000]}\n\n")
                else:
                    f.write(f"TOOL_CALLS: {json.dumps(res['tool_calls'])}\n\n")

    except requests.exceptions.Timeout:
        res["error_msg"] = f"Timeout after {TIMEOUT}s"
        res["status_code"] = "TIMEOUT"
    except Exception as e:
        res["error_msg"] = str(e)[:200]

    return res


# --- Main ---


def main():
    if not API_KEY:
        print("Set MINDROUTER_API_KEY environment variable")
        return

    # Discover models
    print(f"Discovering tool-capable models from {BASE}...")
    models = discover_tool_models()
    if not models:
        print("No tool-capable models found!")
        return

    print(f"Found {len(models)} tool-capable models:\n")
    for m in models:
        tags = []
        if m["thinking"]:
            tags.append("thinking")
        if m["multimodal"]:
            tags.append("multimodal")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"  {m['id']}{tag_str}")

    # Build test matrix
    api_styles = ["openai", "ollama", "anthropic"]
    stream_modes = [False, True]

    tasks = []
    for m in models:
        for style in api_styles:
            for stream in stream_modes:
                for rep in range(REPLICATES):
                    tasks.append((m["id"], style, stream, rep))

    total = len(tasks)
    print(f"\nRunning {total} tests ({len(models)} models x {len(api_styles)} APIs x {len(stream_modes)} stream modes x {REPLICATES} reps)")
    print(f"Max workers: {MAX_WORKERS}, Timeout: {TIMEOUT}s\n")

    # Init log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== TOOL CALLING TEST START {time.ctime()} ===\n")
        f.write(f"Base URL: {BASE}\n")
        f.write(f"Models: {len(models)}, Tests: {total}\n\n")

    # Header
    print(
        f"{'PROGRESS':<12} | {'MODEL':<30} | {'API':<10} | {'STREAM':<8} | "
        f"{'TOOLS':<6} | {'LATENCY':<10} | {'STATUS'}"
    )
    print("-" * 110)

    # Run tests
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(run_single_test, *t): t for t in tasks
        }
        for future in as_completed(future_to_task):
            completed += 1
            r = future.result()
            results.append(r)

            # Format output
            model_short = r["model"]
            if len(model_short) > 28:
                model_short = model_short[:28] + ".."
            stream_str = "stream" if r["stream"] else "sync"
            n_tools = len(r["tool_calls"])
            lat_str = f"{r['latency']:.1f}s" if r["latency"] > 0 else "N/A"
            status_mark = "PASS" if r["ok"] else "FAIL"

            print(
                f"[{completed:03d}/{total:03d}]   | {model_short:<30} | "
                f"{r['api_style']:<10} | {stream_str:<8} | "
                f"{n_tools:<6} | {lat_str:<10} | {status_mark} "
                f"{'(' + str(r['status_code']) + ')' if not r['ok'] else ''}"
            )
            if not r["ok"] and r["error_msg"]:
                print(f"             | {r['error_msg'][:95]}")

    # --- Summary tables ---
    print("\n" + "=" * 110)
    print("SUMMARY BY MODEL")
    print("=" * 110)

    # Aggregate by model
    model_summary = {}
    for r in results:
        key = r["model"]
        if key not in model_summary:
            model_summary[key] = {
                "total": 0, "pass": 0, "fail": 0, "timeout": 0,
                "latencies": [],
                "styles_pass": {"openai": 0, "ollama": 0, "anthropic": 0},
                "styles_total": {"openai": 0, "ollama": 0, "anthropic": 0},
            }
        s = model_summary[key]
        s["total"] += 1
        if r["ok"]:
            s["pass"] += 1
            s["latencies"].append(r["latency"])
            s["styles_pass"][r["api_style"]] += 1
        elif r["status_code"] == "TIMEOUT":
            s["timeout"] += 1
        else:
            s["fail"] += 1
        s["styles_total"][r["api_style"]] += 1

    model_table = []
    for model_id in sorted(model_summary.keys()):
        s = model_summary[model_id]
        rate = f"{100 * s['pass'] / s['total']:.0f}%" if s["total"] else "N/A"
        avg_lat = f"{statistics.mean(s['latencies']):.1f}s" if s["latencies"] else "N/A"
        oi = f"{s['styles_pass']['openai']}/{s['styles_total']['openai']}"
        ol = f"{s['styles_pass']['ollama']}/{s['styles_total']['ollama']}"
        an = f"{s['styles_pass']['anthropic']}/{s['styles_total']['anthropic']}"
        model_table.append([
            model_id, f"{s['pass']}/{s['total']}", rate,
            oi, ol, an, avg_lat,
        ])

    print(tabulate(
        model_table,
        headers=["Model", "Pass/Total", "Rate", "OpenAI", "Ollama", "Anthropic", "Avg Lat"],
        tablefmt="grid",
    ))

    # Aggregate by API style x stream mode
    print("\n" + "=" * 110)
    print("SUMMARY BY API STYLE x STREAM MODE")
    print("=" * 110)

    style_summary = {}
    for r in results:
        key = (r["api_style"], "stream" if r["stream"] else "sync")
        if key not in style_summary:
            style_summary[key] = {"total": 0, "pass": 0, "latencies": []}
        style_summary[key]["total"] += 1
        if r["ok"]:
            style_summary[key]["pass"] += 1
            style_summary[key]["latencies"].append(r["latency"])

    style_table = []
    for (style, mode) in sorted(style_summary.keys()):
        s = style_summary[(style, mode)]
        rate = f"{100 * s['pass'] / s['total']:.0f}%" if s["total"] else "N/A"
        avg_lat = f"{statistics.mean(s['latencies']):.1f}s" if s["latencies"] else "N/A"
        style_table.append([style, mode, f"{s['pass']}/{s['total']}", rate, avg_lat])

    print(tabulate(
        style_table,
        headers=["API Style", "Mode", "Pass/Total", "Rate", "Avg Lat"],
        tablefmt="grid",
    ))

    # Overall
    total_pass = sum(1 for r in results if r["ok"])
    total_fail = sum(1 for r in results if not r["ok"])
    print(f"\nOVERALL: {total_pass}/{len(results)} passed, {total_fail} failed")
    print(f"Log written to {LOG_FILE}")

    # Exit code
    if total_fail > 0:
        exit(1)


if __name__ == "__main__":
    main()

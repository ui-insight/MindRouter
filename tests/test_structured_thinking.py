import os
import requests
import json
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
from jsonschema import validate, ValidationError

# --- Configuration ---
BASE = os.environ.get("MINDROUTER_BASE_URL", "https://localhost:8000")
REPLICATES = 3 
MAX_WORKERS = 10
TIMEOUT = 500
API_KEY = os.environ.get("MINDROUTER_API_KEY")
LOG_FILE = "log.txt"

log_lock = threading.Lock()

# Elaborate Schema with Enums and Constraints
SCHEMA = {
    "type": "object",
    "properties": {
        "full_name": {"type": "string"},
        "year_of_birth": {"type": "integer", "minimum": 1800},
        "scientific_field": {
            "type": "string", 
            "enum": ["Physics", "Chemistry", "Biology", "Mathematics", "Computer Science"]
        },
        "notable_achievements": {
            "type": "array",
            "items": {
                "type": "object", 
                "properties": {"title": {"type": "string"}, "year": {"type": "integer"}}, 
                "required": ["title", "year"]
            }
        },
        "is_nobel_laureate": {"type": "boolean"}
    },
    "required": ["full_name", "year_of_birth", "scientific_field", "notable_achievements", "is_nobel_laureate"],
    "additionalProperties": False
}

PROMPT = "Provide structured historical data for Albert Einstein. Output ONLY the JSON object."

# Matrix Definition
MODEL_VARIANTS = [
    ("openai/gpt-oss-120b", {"effort": "low"}),
    ("openai/gpt-oss-120b", {"effort": "medium"}),
    ("openai/gpt-oss-120b", {"effort": "high"}),
    ("gpt-oss-32k:120b", {"effort": "low"}),
    ("gpt-oss-32k:120b", {"effort": "medium"}),
    ("gpt-oss-32k:120b", {"effort": "high"}),
    ("qwen/qwen3.5-400b", {"thinking": False}),
    ("qwen/qwen3.5-400b", {"thinking": True}),
    ("qwen3-32k:32b", {"thinking": True}),
    ("qwen3-32k:32b", {"thinking": False}),
    ("qwen2.5-8k:7b", "Default"),
    ("phi4:14b", "Default"),
]

ENDPOINTS = ["/v1/chat/completions", "/api/chat", "/api/generate", "/anthropic/v1/messages"]

def build_payload(endpoint, model, reasoning):
    messages = [{"role": "user", "content": PROMPT}]
    
    # 1. OpenAI / vLLM Dialect
    if endpoint == "/v1/chat/completions":
        payload = {
            "model": model, "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {"name": "report", "strict": True, "schema": SCHEMA}}
        }
        if isinstance(reasoning, dict):
            # GPT-OSS style effort
            if "effort" in reasoning:
                payload["reasoning_effort"] = reasoning["effort"]
            # Qwen style hard-toggle
            if "thinking" in reasoning:
                payload["chat_template_kwargs"] = {"enable_thinking": reasoning["thinking"]}
        return payload

    # 2. Ollama Dialects
    elif endpoint in ["/api/chat", "/api/generate"]:
        think_val = False
        if isinstance(reasoning, dict):
            if "thinking" in reasoning:
                think_val = reasoning["thinking"]  # bool for qwen-style
            elif "effort" in reasoning:
                think_val = reasoning["effort"]  # "low"/"medium"/"high" for gpt-oss

        payload = {"model": model, "stream": False, "format": SCHEMA, "think": think_val}
        if endpoint == "/api/chat": payload["messages"] = messages
        else: payload["prompt"] = PROMPT
        return payload

    # 3. Anthropic 2026 Dialect
    elif endpoint == "/anthropic/v1/messages":
        payload = {
            "model": model, "messages": messages, "max_tokens": 16384, 
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}
        }
        if isinstance(reasoning, dict):
            if reasoning.get("thinking") or "effort" in reasoning:
                payload["thinking"] = {"type": "enabled", "budget_tokens": 8192}
            elif "thinking" in reasoning and not reasoning["thinking"]:
                payload["thinking"] = {"type": "disabled"}
        return payload

def extract_thinking_and_content(endpoint, response_json):
    thinking, content = None, None
    try:
        if endpoint == "/v1/chat/completions":
            msg = response_json["choices"][0]["message"]
            thinking = msg.get("reasoning") or msg.get("reasoning_content")
            content = msg.get("content")
        elif endpoint in ["/api/chat", "/api/generate"]:
            msg = response_json.get("message", response_json)
            # Check all possible keys: Ollama native "thinking", canonical "reasoning"/"reasoning_content"
            thinking = msg.get("thinking") or msg.get("reasoning") or msg.get("reasoning_content")
            content = msg.get("content") or response_json.get("response")
        elif endpoint == "/anthropic/v1/messages":
            for block in response_json.get("content", []):
                if block.get("type") == "thinking": thinking = block.get("thinking")
                if block.get("type") == "text": content = block.get("text")
    except: pass
    return thinking, content

def run_task(endpoint, model, reasoning, rep_id):
    url = f"{BASE}{endpoint}"
    payload = build_payload(endpoint, model, reasoning)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    target_lvl = "N/A"
    if isinstance(reasoning, dict):
        if "effort" in reasoning: target_lvl = reasoning["effort"]
        elif "thinking" in reasoning: target_lvl = "ON" if reasoning["thinking"] else "OFF"

    res = {
        "endpoint": endpoint, "model": model, "think_lvl": target_lvl, 
        "ok": False, "lat": 0, "status_code": "ERR", 
        "think_detected": False, "structured_ok": False, "think_match": False, "error_msg": ""
    }
    
    try:
        t0 = time.perf_counter()
        r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        res["lat"] = time.perf_counter() - t0
        res["status_code"] = r.status_code

        if r.ok:
            thinking, content = extract_thinking_and_content(endpoint, r.json())
            res["think_detected"] = bool(thinking and len(thinking.strip()) > 0)
            
            # Struct Check
            if content:
                try:
                    clean = content.strip().replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean)
                    validate(instance=data, schema=SCHEMA)
                    res["structured_ok"] = True
                except: res["structured_ok"] = False

            # Strict Thinking Match Logic
            if target_lvl == "N/A":
                res["think_match"] = True  # No preference — always OK
            elif target_lvl == "OFF":
                res["think_match"] = not res["think_detected"]  # Must NOT detect thinking
            else:
                res["think_match"] = res["think_detected"]  # MUST detect thinking
            
            if res["structured_ok"] and res["think_match"]:
                res["ok"] = True
            elif not res["think_match"]:
                res["status_code"] = "THINK_MISMATCH"
                res["error_msg"] = f"Detected:{res['think_detected']} while Target:{target_lvl}"
            else:
                res["status_code"] = "SCHEMA_ERR"

        with log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"--- REQ {endpoint} | {model} | Target: {target_lvl} | OK: {res['ok']} ---\n")
                f.write(f"BODY: {r.text}\n\n")
    except Exception as e: res["error_msg"] = str(e)
    return res

def main():
    if not API_KEY: return print("❌ Set MINDROUTER_API_KEY")
    with open(LOG_FILE, "w", encoding="utf-8") as f: f.write(f"=== TEST START {time.ctime()} ===\n")

    tasks = [(ep, m, r, i+1) for ep in ENDPOINTS for m, r in MODEL_VARIANTS for i in range(REPLICATES)]
    results, completed = [], 0
    
    print(f"{'PROGRESS':<10} | {'ENDPOINT':<22} | {'MODEL':<20} | {'THINKING':<10} | {'DETECTED':<10} | {'STRUCT':<8} | {'STATUS'}")
    print("-" * 125)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(run_task, *t): t for t in tasks}
        for future in as_completed(future_to_task):
            completed += 1
            r = future.result()
            results.append(r)
            
            struct_mark = "✅" if r['structured_ok'] else "❌"
            think_mark = "✅" if (r['think_match'] and r['think_detected']) else ("⚪" if (r['think_match'] and not r['think_detected']) else "🚨")

            print(f"[{completed:03d}/{len(tasks):03d}] | {r['endpoint']:<22} | {r['model'].split('/')[-1]:<20} | {r['think_lvl']:<10} | {think_mark:<10} | {struct_mark:<8} | {'✅ OK' if r['ok'] else '❌ ' + str(r['status_code'])}")
            if not r['ok'] and r['error_msg']:
                print(f"    └─ ❗ {r['error_msg'][:120]}")

    # Summary
    table = []
    summary = {}
    for r in results:
        key = (r['endpoint'], r['model'], r['think_lvl'])
        if key not in summary: summary[key] = {"ok": 0, "struct": 0, "think": 0, "lats": []}
        if r['ok']: summary[key]["ok"] += 1
        if r['structured_ok']: summary[key]["struct"] += 1
        if r['think_match']: summary[key]["think"] += 1
        summary[key]["lats"].append(r['lat'])
    
    for (e, m, t), s in summary.items():
        table.append([e, m, t, f"{s['struct']}/{REPLICATES}", f"{s['think']}/{REPLICATES}", f"{statistics.mean(s['lats']):.2f}s" if s['lats'] else "N/A"])
    
    print("\n" + tabulate(sorted(table), headers=["Endpoint", "Model", "Thinking", "Structured", "Think Match", "Lat"], tablefmt="grid"))

if __name__ == "__main__":
    main()

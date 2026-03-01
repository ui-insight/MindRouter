#!/usr/bin/env python3
"""
MindRouter2 Stress Test with Auto-Provisioned Multi-User Support

Validates the fair-share WDRR scheduler under load by simulating multiple
concurrent users with different roles (student, staff, faculty, admin).

Test users are auto-created via the admin API and cleaned up on exit.

Usage:
    python stress.py --api-key mr2_ADMIN_KEY
    python stress.py --api-key mr2_ADMIN_KEY --duration 300 --concurrency 20
    python stress.py --api-key mr2_ADMIN_KEY --users 8
"""

import argparse
import asyncio
import json
import math
import os
import random
import signal
import string
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR_SCREEN = "\033[2J\033[H"

# Prompts used for chat requests
PROMPTS = [
    "Explain the concept of recursion in one sentence.",
    "What is the capital of France?",
    "Describe photosynthesis briefly.",
    "Name three programming languages.",
    "What is 2 + 2?",
    "Summarize the theory of relativity in one sentence.",
    "What color is the sky?",
    "Define the word 'algorithm'.",
    "List two prime numbers.",
    "What is machine learning?",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TestUser:
    """A provisioned test user."""
    user_id: int
    username: str
    role: str
    api_key: str  # the full raw key


@dataclass
class RequestRecord:
    """A single recorded request result."""
    timestamp: float
    username: str
    role: str
    endpoint: str
    status_code: int
    latency_s: float
    tokens: int = 0
    error: str = ""
    streaming: bool = False


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------
class MetricsCollector:
    """Thread-safe metrics accumulator."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.records: List[RequestRecord] = []
        self.active_count = 0

    async def record(self, rec: RequestRecord):
        async with self._lock:
            self.records.append(rec)

    async def inc_active(self):
        async with self._lock:
            self.active_count += 1

    async def dec_active(self):
        async with self._lock:
            self.active_count = max(0, self.active_count - 1)

    async def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of current metrics."""
        async with self._lock:
            records = list(self.records)
            active = self.active_count

        if not records:
            return {
                "total": 0, "ok": 0, "err": 0, "active": active,
                "throughput": 0.0, "tokens": 0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0,
                "by_user": {}, "errors": {},
            }

        ok = [r for r in records if 200 <= r.status_code < 300]
        err = [r for r in records if r.status_code < 200 or r.status_code >= 300]

        latencies = sorted(r.latency_s for r in ok) if ok else [0.0]
        total_tokens = sum(r.tokens for r in records)

        elapsed = records[-1].timestamp - records[0].timestamp if len(records) > 1 else 1.0
        throughput = len(records) / max(elapsed, 0.001)

        # Per-user breakdown
        by_user: Dict[str, Dict[str, Any]] = {}
        for r in records:
            key = r.username
            if key not in by_user:
                by_user[key] = {"role": r.role, "count": 0, "latencies": [], "tokens": 0}
            by_user[key]["count"] += 1
            if 200 <= r.status_code < 300:
                by_user[key]["latencies"].append(r.latency_s)
            by_user[key]["tokens"] += r.tokens

        # Error breakdown by status code
        error_counts: Dict[int, int] = {}
        for r in err:
            error_counts[r.status_code] = error_counts.get(r.status_code, 0) + 1

        return {
            "total": len(records),
            "ok": len(ok),
            "err": len(err),
            "active": active,
            "throughput": throughput,
            "tokens": total_tokens,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "max": max(latencies) if latencies else 0.0,
            "by_user": by_user,
            "errors": error_counts,
        }


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


# ---------------------------------------------------------------------------
# UserProvisioner — async context manager
# ---------------------------------------------------------------------------
class UserProvisioner:
    """Creates and cleans up test users via the admin API."""

    def __init__(self, base_url: str, admin_key: str, user_specs: List[Dict[str, str]]):
        self.base_url = base_url
        self.admin_key = admin_key
        self.user_specs = user_specs
        self.users: List[TestUser] = []
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> List[TestUser]:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0),
        )
        headers = {"Authorization": f"Bearer {self.admin_key}"}

        # Fetch groups to map role names to group IDs
        groups_resp = await self._client.get("/api/admin/groups", headers=headers)
        if groups_resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch groups: {groups_resp.status_code} {groups_resp.text[:200]}")
        groups = groups_resp.json().get("groups", [])
        # Map: role name -> group_id (try exact match, then plural, then fallback)
        role_to_group: Dict[str, int] = {}
        for g in groups:
            role_to_group[g["name"].lower()] = g["id"]
        def _resolve_group_id(role: str) -> int:
            r = role.lower()
            if r in role_to_group:
                return role_to_group[r]
            # Try plural/singular variants
            if r + "s" in role_to_group:
                return role_to_group[r + "s"]
            if r.rstrip("s") in role_to_group:
                return role_to_group[r.rstrip("s")]
            # Fallback to first non-admin group, or first group
            for g in groups:
                if not g.get("is_admin"):
                    return g["id"]
            return groups[0]["id"]

        for spec in self.user_specs:
            # Create user
            password = "StressTest_" + "".join(random.choices(string.ascii_letters + string.digits, k=12))
            group_id = _resolve_group_id(spec["role"])
            resp = await self._client.post("/api/admin/users", headers=headers, json={
                "username": spec["username"],
                "email": spec["email"],
                "password": password,
                "role": spec["role"],
                "group_id": group_id,
                "full_name": f"Stress Test {spec['role'].title()}",
            })
            if resp.status_code == 409:
                # User already exists (leftover from a previous run) — look up and delete first
                print(f"  {YELLOW}Warning: user '{spec['username']}' already exists, cleaning up...{RESET}")
                # List users to find the ID
                list_resp = await self._client.get("/api/admin/users", headers=headers)
                if list_resp.status_code == 200:
                    for u in list_resp.json().get("users", []):
                        if u["username"] == spec["username"]:
                            await self._client.delete(f"/api/admin/users/{u['id']}", headers=headers)
                            break
                # Retry creation
                resp = await self._client.post("/api/admin/users", headers=headers, json={
                    "username": spec["username"],
                    "email": spec["email"],
                    "password": password,
                    "role": spec["role"],
                    "group_id": group_id,
                    "full_name": f"Stress Test {spec['role'].title()}",
                })

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to create user '{spec['username']}': "
                    f"{resp.status_code} {resp.text[:200]}"
                )
            user_data = resp.json()
            user_id = user_data["id"]

            # Create API key
            key_resp = await self._client.post(
                f"/api/admin/users/{user_id}/api-keys",
                headers=headers,
                json={"name": "stress-test"},
            )
            if key_resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to create API key for '{spec['username']}': "
                    f"{key_resp.status_code} {key_resp.text[:200]}"
                )
            key_data = key_resp.json()

            self.users.append(TestUser(
                user_id=user_id,
                username=spec["username"],
                role=spec["role"],
                api_key=key_data["full_key"],
            ))
            print(f"  {GREEN}Created{RESET} {spec['username']} ({spec['role']}) → id={user_id}")

        return self.users

    async def __aexit__(self, *exc):
        if not self._client:
            return
        headers = {"Authorization": f"Bearer {self.admin_key}"}
        for user in self.users:
            # Retry deletion a few times in case in-flight requests are still completing
            for attempt in range(3):
                try:
                    resp = await self._client.delete(
                        f"/api/admin/users/{user.user_id}",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        print(f"  {GREEN}Deleted{RESET} {user.username} (id={user.user_id})")
                        break
                    elif attempt < 2:
                        await asyncio.sleep(2)
                    else:
                        print(f"  {RED}Failed to delete{RESET} {user.username}: {resp.status_code}")
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2)
                    else:
                        print(f"  {RED}Error deleting{RESET} {user.username}: {e}")
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------
def _random_prompt() -> str:
    return random.choice(PROMPTS)


def build_openai_chat(model: str, max_tokens: int, stream: bool) -> Dict[str, Any]:
    return {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": _random_prompt()}],
    }


def build_ollama_chat(model: str, max_tokens: int, stream: bool) -> Dict[str, Any]:
    return {
        "model": model,
        "stream": stream,
        "options": {"num_predict": max_tokens},
        "messages": [{"role": "user", "content": _random_prompt()}],
    }


def build_ollama_generate(model: str, max_tokens: int) -> Dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "prompt": _random_prompt(),
        "options": {"num_predict": max_tokens},
    }


def build_embedding(model: str) -> Dict[str, Any]:
    return {
        "model": model,
        "input": _random_prompt(),
    }


# Workload definitions: (weight, endpoint, builder_fn, is_streaming)
def get_workload(cfg) -> List[tuple]:
    workload = [
        (40, "/v1/chat/completions", lambda: build_openai_chat(cfg.vllm_model, cfg.max_tokens, False), False),
        (20, "/v1/chat/completions", lambda: build_openai_chat(cfg.vllm_model, cfg.max_tokens, True), True),
        (20, "/api/chat", lambda: build_ollama_chat(cfg.ollama_model, cfg.max_tokens, False), False),
        (10, "/api/chat", lambda: build_ollama_chat(cfg.ollama_model, cfg.max_tokens, True), True),
    ]
    if not cfg.chat_only:
        workload.append((5, "/v1/embeddings", lambda: build_embedding(cfg.embedding_model), False))
        workload.append((5, "/api/generate", lambda: build_ollama_generate(cfg.ollama_model, cfg.max_tokens), False))
    return workload


def pick_workload(workload: List[tuple]) -> tuple:
    """Weighted random selection from workload mix."""
    total = sum(w for w, *_ in workload)
    r = random.randint(1, total)
    cumulative = 0
    for entry in workload:
        cumulative += entry[0]
        if r <= cumulative:
            return entry
    return workload[-1]


# ---------------------------------------------------------------------------
# StressWorker
# ---------------------------------------------------------------------------
async def stress_worker(
    worker_id: int,
    users: List[TestUser],
    workload: List[tuple],
    metrics: MetricsCollector,
    base_url: str,
    timeout: float,
    stop_event: asyncio.Event,
    verbose: bool = False,
):
    """A single async worker that loops sending requests until stopped."""
    req_seq = 0
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout),
    ) as client:
        while not stop_event.is_set():
            user = random.choice(users)
            weight, endpoint, builder_fn, is_streaming = pick_workload(workload)
            payload = builder_fn()
            headers = {"Authorization": f"Bearer {user.api_key}"}
            req_seq += 1
            tag = f"W{worker_id}#{req_seq}"
            mode = "stream" if is_streaming else "sync"

            if verbose:
                _ts = time.strftime("%H:%M:%S")
                print(
                    f"  {DIM}{_ts}{RESET} {CYAN}{tag}{RESET}  "
                    f"{GREEN}SEND{RESET}  {endpoint} ({mode})  "
                    f"user={user.username}",
                    flush=True,
                )

            await metrics.inc_active()
            t0 = time.monotonic()
            status_code = 0
            tokens = 0
            error = ""
            resp_body = None

            try:
                if is_streaming:
                    tokens = await _do_streaming_request(client, endpoint, headers, payload)
                    status_code = 200
                else:
                    resp = await client.post(endpoint, headers=headers, json=payload)
                    status_code = resp.status_code
                    if status_code == 200:
                        resp_body = resp.json()
                        tokens = _extract_tokens(resp_body, endpoint)
                    else:
                        # Capture error body for verbose output
                        try:
                            resp_body = resp.json()
                        except Exception:
                            resp_body = {"raw": resp.text[:200]}
            except httpx.TimeoutException:
                status_code = 408
                error = "timeout"
            except httpx.ConnectError:
                status_code = 502
                error = "connection_error"
            except Exception as e:
                status_code = 500
                error = str(e)[:200]
            finally:
                latency = time.monotonic() - t0
                await metrics.dec_active()

            if verbose:
                _ts = time.strftime("%H:%M:%S")
                ok = 200 <= status_code < 300
                status_color = GREEN if ok else RED
                line = (
                    f"  {DIM}{_ts}{RESET} {CYAN}{tag}{RESET}  "
                    f"{status_color}{status_code}{RESET}  "
                    f"{latency:.2f}s  {tokens} tok"
                )
                if error:
                    line += f"  {RED}{error}{RESET}"
                if not ok and resp_body:
                    # Show server error detail
                    detail = ""
                    if isinstance(resp_body, dict):
                        detail = resp_body.get("detail", resp_body.get("error", {}).get("message", ""))
                    if detail:
                        line += f"  {YELLOW}{str(detail)[:120]}{RESET}"
                print(line, flush=True)

            await metrics.record(RequestRecord(
                timestamp=time.time(),
                username=user.username,
                role=user.role,
                endpoint=endpoint,
                status_code=status_code,
                latency_s=latency,
                tokens=tokens,
                error=error,
                streaming=is_streaming,
            ))

            # Small jitter between requests
            await asyncio.sleep(random.uniform(0.01, 0.1))


async def _do_streaming_request(
    client: httpx.AsyncClient, endpoint: str,
    headers: dict, payload: dict,
) -> int:
    """Send a streaming request, consume all chunks, return token estimate."""
    tokens = 0
    async with client.stream("POST", endpoint, headers=headers, json=payload) as resp:
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Streaming error: {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        if endpoint.startswith("/v1/"):
            # SSE format
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        usage = chunk.get("usage")
                        if usage:
                            tokens = usage.get("total_tokens", 0)
                        else:
                            # Count chunks as rough token estimate
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                tokens += 1
                    except json.JSONDecodeError:
                        pass
        else:
            # Ollama NDJSON format
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("done"):
                        tokens = obj.get("eval_count", tokens)
                        break
                    if obj.get("message", {}).get("content"):
                        tokens += 1
                except json.JSONDecodeError:
                    pass
    return tokens


def _extract_tokens(body: dict, endpoint: str) -> int:
    """Extract token count from a response body."""
    if "usage" in body:
        return body["usage"].get("total_tokens", 0)
    # Ollama responses
    if "eval_count" in body:
        return body.get("prompt_eval_count", 0) + body.get("eval_count", 0)
    # Embeddings
    if "data" in body and isinstance(body["data"], list):
        return body.get("usage", {}).get("total_tokens", 0)
    return 0


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
async def dashboard_loop(
    metrics: MetricsCollector,
    base_url: str,
    admin_key: str,
    duration: int,
    start_time: float,
    stop_event: asyncio.Event,
):
    """Print live dashboard every 5 seconds."""
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(10.0),
    ) as client:
        headers = {"Authorization": f"Bearer {admin_key}"}

        while not stop_event.is_set():
            await asyncio.sleep(5)
            if stop_event.is_set():
                break

            snap = await metrics.snapshot()
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)

            # Poll /status for queue/backend info
            queue_pending = 0
            backends_healthy = 0
            try:
                resp = await client.get("/status")
                if resp.status_code == 200:
                    status_data = resp.json()
                    queue_info = status_data.get("queue", {})
                    queue_pending = queue_info.get("pending", queue_info.get("total", 0))
                    backends_healthy = status_data.get("backends", {}).get("healthy", 0)
            except Exception:
                pass

            # Format elapsed time
            e_min, e_sec = divmod(int(elapsed), 60)
            d_min, d_sec = divmod(duration, 60)

            lines = []
            lines.append(f"{BOLD}{CYAN}─── MindRouter2 Stress Test ── {e_min:02d}:{e_sec:02d} / {d_min:02d}:{d_sec:02d} {'─' * 30}{RESET}")
            lines.append(
                f"  Requests:  {snap['total']:,} total │ "
                f"{snap['ok']:,} ok │ {snap['err']:,} err │ "
                f"{snap['active']} active"
            )
            lines.append(
                f"  Throughput: {snap['throughput']:.1f} req/s  │ "
                f"Tokens: {snap['tokens']:,} total"
            )
            lines.append(
                f"  Latency:   p50={snap['p50']:.1f}s  "
                f"p95={snap['p95']:.1f}s  "
                f"p99={snap['p99']:.1f}s"
            )
            lines.append(
                f"  Queue:     {queue_pending} pending  │ "
                f"Backends: {backends_healthy} healthy"
            )

            if snap["errors"]:
                err_parts = [f"{code}:{cnt}" for code, cnt in sorted(snap["errors"].items())]
                lines.append(f"  Errors:    {', '.join(err_parts)}")

            lines.append("")
            lines.append("  Per-user breakdown:")
            for username, info in sorted(snap["by_user"].items()):
                user_lats = sorted(info["latencies"]) if info["latencies"] else [0.0]
                user_p50 = _percentile(user_lats, 50)
                lines.append(
                    f"    {username:<30s} ({info['role']:<8s}) "
                    f"{info['count']:>5d} req │ p50={user_p50:.1f}s │ "
                    f"{info['tokens']:>8,} tok"
                )

            lines.append(f"{BOLD}{CYAN}{'─' * 70}{RESET}")

            # Print with screen clear
            output = CLEAR_SCREEN + "\n".join(lines)
            print(output, flush=True)


# ---------------------------------------------------------------------------
# User spec generation
# ---------------------------------------------------------------------------
def generate_user_specs(num_users: int) -> List[Dict[str, str]]:
    """Generate user specs with role distribution: ~50% student, ~17% each for others."""
    if num_users < 4:
        # With fewer than 4, ensure at least one of each important role
        roles = ["student", "staff", "faculty", "admin"][:num_users]
    else:
        # Allocate: 1 admin, 1 faculty, 1 staff, rest students
        n_admin = max(1, round(num_users * 0.17))
        n_faculty = max(1, round(num_users * 0.17))
        n_staff = max(1, round(num_users * 0.17))
        n_student = num_users - n_admin - n_faculty - n_staff
        n_student = max(1, n_student)

        roles = (
            ["student"] * n_student
            + ["staff"] * n_staff
            + ["faculty"] * n_faculty
            + ["admin"] * n_admin
        )

    # Count per role for naming
    role_counters: Dict[str, int] = {}
    specs = []
    for role in roles:
        role_counters[role] = role_counters.get(role, 0) + 1
        n = role_counters[role]
        username = f"_stress_{role}_{n}"
        specs.append({
            "username": username,
            "email": f"{username}@test.local",
            "role": role,
        })
    return specs


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
async def print_summary(metrics: MetricsCollector, elapsed: float):
    """Print the final test summary."""
    snap = await metrics.snapshot()

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  MindRouter2 Stress Test — Final Summary{RESET}")
    print(f"{'=' * 70}")

    # Overall stats
    success_rate = (snap["ok"] / snap["total"] * 100) if snap["total"] else 0
    print(f"\n  Duration:     {elapsed:.1f}s")
    print(f"  Total:        {snap['total']:,} requests")
    print(f"  Success:      {snap['ok']:,} ({success_rate:.1f}%)")
    print(f"  Errors:       {snap['err']:,}")
    print(f"  Throughput:   {snap['throughput']:.2f} req/s")
    print(f"  Tokens:       {snap['tokens']:,}")

    # Error breakdown
    if snap["errors"]:
        print(f"\n  {RED}Error breakdown:{RESET}")
        for code, count in sorted(snap["errors"].items()):
            print(f"    HTTP {code}: {count}")

    # Latency percentiles
    print(f"\n  Latency percentiles:")
    print(f"    p50:  {snap['p50']:.2f}s")
    print(f"    p95:  {snap['p95']:.2f}s")
    print(f"    p99:  {snap['p99']:.2f}s")
    print(f"    max:  {snap['max']:.2f}s")

    # Per-user table
    print(f"\n  {'User':<30s} {'Role':<10s} {'Reqs':>6s} {'p50':>8s} {'Tokens':>10s}")
    print(f"  {'─' * 30} {'─' * 10} {'─' * 6} {'─' * 8} {'─' * 10}")
    for username, info in sorted(snap["by_user"].items()):
        user_lats = sorted(info["latencies"]) if info["latencies"] else [0.0]
        user_p50 = _percentile(user_lats, 50)
        print(
            f"  {username:<30s} {info['role']:<10s} "
            f"{info['count']:>6d} {user_p50:>7.2f}s {info['tokens']:>10,}"
        )

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main(cfg):
    user_specs = generate_user_specs(cfg.users)
    workload = get_workload(cfg)

    print(f"\n{BOLD}MindRouter2 Stress Test{RESET}")
    print(f"  Base URL:      {cfg.base_url}")
    print(f"  Duration:      {cfg.duration}s")
    print(f"  Concurrency:   {cfg.concurrency}")
    print(f"  Users:         {cfg.users}")
    print(f"  Ollama model:  {cfg.ollama_model}")
    print(f"  vLLM model:    {cfg.vllm_model}")
    if not cfg.chat_only:
        print(f"  Embed model:   {cfg.embedding_model}")
    print(f"  Max tokens:    {cfg.max_tokens}")
    print(f"  Timeout:       {cfg.timeout}s")
    if cfg.chat_only:
        print(f"  Mode:          chat-only (no embeddings/generate)")
    if cfg.verbose:
        print(f"  Verbose:       ON (per-request logging)")

    # SETUP
    print(f"\n{BOLD}{CYAN}── Setup: Creating test users ──{RESET}")

    provisioner = UserProvisioner(cfg.base_url, cfg.api_key, user_specs)

    stop_event = asyncio.Event()
    metrics = MetricsCollector()

    # Handle Ctrl+C gracefully
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    async with provisioner as users:
        if not users:
            print(f"{RED}No users created, aborting.{RESET}")
            return

        print(f"\n{BOLD}{CYAN}── Running stress test ({cfg.duration}s) ──{RESET}\n")

        start_time = time.time()

        # Schedule timer to stop after duration
        async def _timer():
            await asyncio.sleep(cfg.duration)
            stop_event.set()

        timer_task = asyncio.create_task(_timer())

        # Start dashboard (skip in verbose mode — log lines would get cleared)
        dash_task = None
        if not cfg.verbose:
            dash_task = asyncio.create_task(
                dashboard_loop(metrics, cfg.base_url, cfg.api_key, cfg.duration, start_time, stop_event)
            )

        # Start workers
        worker_tasks = [
            asyncio.create_task(
                stress_worker(i, users, workload, metrics, cfg.base_url, cfg.timeout, stop_event,
                              verbose=cfg.verbose)
            )
            for i in range(cfg.concurrency)
        ]

        # Wait for stop
        await stop_event.wait()

        # Cancel timer if it's still running
        timer_task.cancel()
        try:
            await timer_task
        except asyncio.CancelledError:
            pass

        # Cancel dashboard
        if dash_task:
            dash_task.cancel()
            try:
                await dash_task
            except asyncio.CancelledError:
                pass

        elapsed = time.time() - start_time

        # Clear dashboard output before summary (skip in verbose mode)
        if not cfg.verbose:
            print(CLEAR_SCREEN, end="", flush=True)

        # Wait for workers to finish their current in-flight requests
        # (stop_event is set so they won't start new ones)
        print(f"\n{DIM}  Waiting for in-flight requests to complete...{RESET}", flush=True)
        done, pending = await asyncio.wait(worker_tasks, timeout=cfg.timeout)
        if pending:
            print(f"  {YELLOW}{len(pending)} workers still active after timeout, cancelling...{RESET}")
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        # REPORT
        print(f"\n{BOLD}{CYAN}── Teardown: Cleaning up test users ──{RESET}")

    # Print summary after teardown (provisioner __aexit__ runs above)
    await print_summary(metrics, elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="MindRouter2 multi-user stress test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stress.py --api-key mr2_ADMIN_KEY
  python stress.py --api-key mr2_ADMIN_KEY --duration 300 --concurrency 20
  python stress.py --api-key mr2_ADMIN_KEY --users 8
""",
    )
    parser.add_argument("--api-key", required=True,
                        help="Admin API key for provisioning (required)")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="MindRouter2 URL (default: http://localhost:8000)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Test duration in seconds (default: 300)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of async workers (default: 10)")
    parser.add_argument("--users", type=int, default=6,
                        help="Number of simulated users (default: 6)")
    parser.add_argument("--ollama-model", default="phi4:14b",
                        help="Ollama chat model (default: phi4:14b)")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-120b",
                        help="vLLM chat model (default: openai/gpt-oss-120b)")
    parser.add_argument("--embedding-model", default="EMBED/all-minilm:33m",
                        help="Embedding model (default: EMBED/all-minilm:33m)")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max tokens per response (default: 32)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Per-request timeout in seconds (default: 180)")
    parser.add_argument("--chat-only", action="store_true", default=False,
                        help="Only send chat requests (skip embeddings and generate)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Print per-request lifecycle lines (disables dashboard)")

    cfg = parser.parse_args()
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()

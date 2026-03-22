#!/usr/bin/env python3
"""
MindRouter Concurrency Test

Sends a configurable number of concurrent requests to a specific model
and reports latency statistics. Useful for testing backend capacity,
timeout behavior, and queue dynamics under controlled load.

Usage:
    python scripts/concurrency_test.py --api-key mr2_xxx --model qwen/qwen3.5-122b --concurrent 24
    python scripts/concurrency_test.py --api-key mr2_xxx --model openai/gpt-oss-120b --concurrent 8 --duration 120
    python scripts/concurrency_test.py --api-key mr2_xxx --model qwen/qwen3.5-122b --concurrent 24 --max-tokens 256
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import List

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

PROMPTS = [
    "Explain the concept of recursion in computer science.",
    "What causes the northern lights?",
    "Describe how a compiler works.",
    "What is the significance of Euler's identity?",
    "Explain the difference between TCP and UDP.",
    "How does photosynthesis convert light to energy?",
    "What is the halting problem?",
    "Describe the process of nuclear fusion in stars.",
    "Explain how public key cryptography works.",
    "What is the theory of general relativity?",
]


@dataclass
class Stats:
    ok: int = 0
    err: int = 0
    timeouts: int = 0
    tokens: int = 0
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    active: int = 0
    peak_active: int = 0


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    stats: Stats,
    request_num: int,
    stream: bool,
    think: bool = False,
):
    prompt = PROMPTS[request_num % len(PROMPTS)]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if stream:
        body["stream"] = True
    if think:
        body["think"] = True

    headers = {"Authorization": f"Bearer {api_key}"}

    stats.active += 1
    if stats.active > stats.peak_active:
        stats.peak_active = stats.active

    t0 = time.monotonic()
    try:
        if stream:
            token_count = 0
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=body,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    stats.err += 1
                    stats.errors.append(f"HTTP {response.status_code}")
                    return
                async for chunk in response.aiter_bytes():
                    # Just consume the stream
                    if b'"completion_tokens"' in chunk:
                        pass
                    token_count += 1
            latency = time.monotonic() - t0
            stats.ok += 1
            stats.latencies.append(latency)
        else:
            r = await client.post(
                f"{base_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
            latency = time.monotonic() - t0
            if r.status_code == 200:
                data = r.json()
                toks = data.get("usage", {}).get("completion_tokens", 0)
                stats.ok += 1
                stats.tokens += toks
                stats.latencies.append(latency)
            else:
                stats.err += 1
                stats.errors.append(f"HTTP {r.status_code}: {r.text[:200]}")
    except httpx.TimeoutException:
        stats.timeouts += 1
        stats.errors.append(f"Timeout after {time.monotonic() - t0:.1f}s")
    except Exception as e:
        stats.err += 1
        stats.errors.append(str(e)[:200])
    finally:
        stats.active -= 1


async def _run_and_release(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    stats: Stats,
    request_num: int,
    stream: bool,
    think: bool = False,
):
    """Run a request then release the semaphore so the submit loop can send another."""
    try:
        await send_request(
            client, base_url, api_key, model, max_tokens, stats, request_num, stream,
            think,
        )
    finally:
        sem.release()


async def main():
    parser = argparse.ArgumentParser(description="MindRouter Concurrency Test")
    parser.add_argument("--api-key", required=True, help="MindRouter API key")
    parser.add_argument(
        "--base-url",
        default="https://mindrouter.uidaho.edu",
        help="MindRouter base URL (default: https://mindrouter.uidaho.edu)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
        help="Model to test (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=8,
        help="Number of concurrent requests (default: 8)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens per request (default: 64)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Use streaming requests",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning mode",
    )
    args = parser.parse_args()

    stats = Stats()
    sem = asyncio.Semaphore(args.concurrent)

    print(f"{BOLD}{CYAN}MindRouter Concurrency Test{RESET}")
    print(f"  Base URL:    {args.base_url}")
    print(f"  Model:       {args.model}")
    print(f"  Concurrent:  {args.concurrent}")
    print(f"  Duration:    {args.duration}s")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Streaming:   {args.stream}")
    print(f"  Thinking:    {args.think}")
    print(f"  Timeout:     {args.timeout}s")
    print()

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0, read=float(args.timeout), write=10.0, pool=10.0
        ),
    )

    t_start = time.monotonic()
    tasks = []
    request_num = 0
    stop_event = asyncio.Event()

    async def submit_loop():
        nonlocal request_num
        while not stop_event.is_set():
            # Wait for a free slot before submitting — this prevents
            # unbounded local queuing behind the semaphore.
            await sem.acquire()
            if stop_event.is_set():
                sem.release()
                break

            task = asyncio.create_task(
                _run_and_release(
                    sem,
                    client,
                    args.base_url,
                    args.api_key,
                    args.model,
                    args.max_tokens,
                    stats,
                    request_num,
                    args.stream,
                    args.think,
                )
            )
            tasks.append(task)
            request_num += 1

            # Progress update
            elapsed = time.monotonic() - t_start
            if request_num % 10 == 0:
                print(
                    f"  {DIM}[{elapsed:5.0f}s]{RESET} "
                    f"sent={request_num} ok={stats.ok} err={stats.err} "
                    f"timeout={stats.timeouts} active={stats.active}/{args.concurrent} "
                    f"peak={stats.peak_active}"
                )

    try:
        # Run the submit loop with a duration deadline
        try:
            await asyncio.wait_for(submit_loop(), timeout=args.duration)
        except asyncio.TimeoutError:
            stop_event.set()

        inflight = len([t for t in tasks if not t.done()])
        if inflight > 0:
            print(f"\n  Waiting for {inflight} in-flight requests...")
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await client.aclose()

    elapsed = time.monotonic() - t_start
    lats = sorted(stats.latencies)

    sep = "=" * 60
    print(f"\n{BOLD}{sep}{RESET}")
    print(f"{BOLD}  RESULTS — {args.model}{RESET}")
    print(f"{BOLD}{sep}{RESET}")
    print(f"  Duration:       {elapsed:.1f}s")
    print(f"  Concurrent:     {args.concurrent} (peak observed: {stats.peak_active})")
    print(f"  Requests OK:    {GREEN}{stats.ok}{RESET}")
    print(f"  Requests ERR:   {RED if stats.err else DIM}{stats.err}{RESET}")
    print(f"  Timeouts:       {RED if stats.timeouts else DIM}{stats.timeouts}{RESET}")
    print(f"  Total tokens:   {stats.tokens}")
    if elapsed > 0:
        print(
            f"  Throughput:     {stats.ok/elapsed:.1f} req/s"
            f", {stats.tokens/elapsed:.1f} tok/s"
        )
    if lats:
        n = len(lats)
        print(f"  Latency min:    {lats[0]:.2f}s")
        print(f"  Latency p25:    {lats[n // 4]:.2f}s")
        print(f"  Latency p50:    {lats[n // 2]:.2f}s")
        print(f"  Latency p75:    {lats[3 * n // 4]:.2f}s")
        print(f"  Latency p90:    {lats[int(n * 0.9)]:.2f}s")
        print(f"  Latency p99:    {lats[int(n * 0.99)]:.2f}s")
        print(f"  Latency max:    {lats[-1]:.2f}s")
    if stats.errors:
        print(f"\n  {RED}First 5 errors:{RESET}")
        for e in stats.errors[:5]:
            print(f"    - {e}")
    print()


if __name__ == "__main__":
    asyncio.run(main())

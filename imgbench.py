#!/usr/bin/env python3
"""
MindRouter image-generation capacity benchmark.

Closed-loop load model: N simulated users each loop
    generate -> think (random pause) -> generate ...
for --duration seconds. Every request is timed CLIENT-SIDE (wall clock from
send to full response), because the gateway's Request rows do not populate
started_at / queue_delay_ms / processing_time_ms for the image path, so the
DB cannot be used to reconstruct latency after the fact.

Works against either the MindRouter gateway (/v1/images/generations with an
API key) or a bare serve_klein.py worker (same endpoint shape, no key).

Typical use (measured on prod 2026-08-24, see docs/image-capacity.md):

    # SLO sweep at the current default step count
    python imgbench.py --api-key mr2_... --base-url https://mindrouter.uidaho.edu \
        --users 12 --inference-steps 20 --duration 180

    # solo latency of one worker, no think time
    python imgbench.py --base-url http://127.0.0.1:18400 --users 1 --pause 0 \
        --inference-steps 4 --duration 60

    # JSON summary for a sweep script
    python imgbench.py ... --json results/u12_s20.json

Report: throughput (img/min), latency p50/p90/p95/p99/max, error count, and
whether the run met --slo-p95 (default 30s).
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time

import httpx

DEFAULT_PROMPTS = [
    "a lighthouse on a rocky coast at dawn, soft fog, oil painting",
    "a red bicycle leaning against a brick wall, afternoon light, photograph",
    "an isometric illustration of a tiny bustling coffee shop",
    "a snow-covered pine forest under the northern lights",
    "a bowl of ramen with steam rising, top-down food photography",
    "a vintage typewriter on a wooden desk with scattered papers",
    "a golden retriever puppy wearing a raincoat in a puddle",
    "a futuristic tram gliding through a rain-soaked neon city",
    "a watercolor map of an imaginary island with tiny villages",
    "a close-up macro shot of a dew-covered spider web",
]


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def parse_pause(spec):
    """'2-6' -> (2.0, 6.0); '3' -> (3.0, 3.0); '0' -> no think time."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return float(lo), float(hi)
    v = float(spec)
    return v, v


async def user_loop(uid, client, args, deadline, samples, errors, progress):
    lo, hi = parse_pause(args.pause)
    rng = random.Random(1000 + uid)
    # Stagger the initial burst so N users don't all arrive in the same ms.
    await asyncio.sleep(rng.uniform(0, min(2.0, hi if hi > 0 else 0.5)))
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    while time.monotonic() < deadline:
        body = {
            "model": args.model,
            "prompt": rng.choice(DEFAULT_PROMPTS) if not args.prompt else args.prompt,
            "n": 1,
            "size": args.size,
            "response_format": "b64_json",
        }
        if args.inference_steps:
            body["num_inference_steps"] = args.inference_steps
        t0 = time.monotonic()
        try:
            r = await client.post(
                f"{args.base_url.rstrip('/')}/v1/images/generations",
                json=body,
                headers=headers,
            )
            dt = time.monotonic() - t0
            if r.status_code == 200 and r.json().get("data"):
                samples.append((t0, dt))
            else:
                errors.append((t0, r.status_code, r.text[:200]))
        except Exception as e:  # noqa: BLE001
            dt = time.monotonic() - t0
            errors.append((t0, "exc", f"{type(e).__name__}: {e}"[:200]))
        progress()
        if hi > 0:
            await asyncio.sleep(rng.uniform(lo, hi))


async def run(args):
    samples, errors = [], []
    t_start = time.monotonic()
    deadline = t_start + args.duration
    done = {"n": 0}
    last_print = {"t": t_start}

    def progress():
        done["n"] += 1
        now = time.monotonic()
        if now - last_print["t"] >= 10:
            last_print["t"] = now
            el = now - t_start
            lat = sorted(d for _, d in samples)
            print(
                f"  t={el:5.0f}s  ok={len(samples):4d}  err={len(errors):3d}  "
                f"{len(samples) / el * 60:6.1f} img/min  "
                f"p50={percentile(lat, 50):5.1f}s p95={percentile(lat, 95):5.1f}s",
                file=sys.stderr,
                flush=True,
            )

    limits = httpx.Limits(max_connections=args.users + 4, max_keepalive_connections=args.users + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), limits=limits, verify=not args.insecure) as client:
        tasks = [
            asyncio.create_task(user_loop(i, client, args, deadline, samples, errors, progress))
            for i in range(args.users)
        ]
        await asyncio.gather(*tasks)
    wall = time.monotonic() - t_start

    # Drop the warm-up window so a cold cache / first-request compile does not
    # distort steady-state numbers.
    steady = [(t, d) for t, d in samples if t - t_start >= args.warmup]
    steady_wall = max(wall - args.warmup, 1e-9)
    lat = sorted(d for _, d in steady)
    lat_all = sorted(d for _, d in samples)

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "users": args.users,
        "inference_steps": args.inference_steps,
        "size": args.size,
        "pause": args.pause,
        "duration_s": round(wall, 1),
        "warmup_s": args.warmup,
        "completed": len(samples),
        "completed_steady": len(steady),
        "errors": len(errors),
        "throughput_img_per_min": round(len(steady) / steady_wall * 60, 2),
        "throughput_all_img_per_min": round(len(samples) / wall * 60, 2),
        "latency_s": {
            "mean": round(statistics.fmean(lat), 2) if lat else None,
            "p50": round(percentile(lat, 50), 2),
            "p90": round(percentile(lat, 90), 2),
            "p95": round(percentile(lat, 95), 2),
            "p99": round(percentile(lat, 99), 2),
            "max": round(lat[-1], 2) if lat else None,
            "min": round(lat[0], 2) if lat else None,
        },
        "slo_p95_s": args.slo_p95,
        "slo_met": bool(lat) and percentile(lat, 95) <= args.slo_p95,
        "error_samples": errors[:5],
    }

    print()
    print(f"=== imgbench  {args.base_url}  users={args.users} steps={args.inference_steps} size={args.size} pause={args.pause}s")
    print(f"  duration    : {wall:.0f}s (steady-state after {args.warmup}s warm-up)")
    print(f"  completed   : {len(samples)}  (steady {len(steady)})   errors: {len(errors)}")
    print(f"  throughput  : {summary['throughput_img_per_min']:.1f} img/min steady  ({summary['throughput_all_img_per_min']:.1f} incl. warm-up)")
    ls = summary["latency_s"]
    print(f"  latency     : mean {ls['mean']}s  p50 {ls['p50']}s  p90 {ls['p90']}s  p95 {ls['p95']}s  p99 {ls['p99']}s  max {ls['max']}s")
    if lat_all and args.warmup:
        print(f"  (all samples p95 {percentile(lat_all, 95):.1f}s)")
    print(f"  SLO p95<={args.slo_p95}s : {'MET' if summary['slo_met'] else 'FAILED'}")
    for t, code, txt in errors[:5]:
        print(f"  error @ +{t - t_start:.0f}s: {code} {txt}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {args.json}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--api-key", default=None, help="MindRouter API key (omit for a bare worker)")
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-9B")
    p.add_argument("--users", type=int, default=1, help="concurrent closed-loop users")
    p.add_argument("--inference-steps", type=int, default=None, help="num_inference_steps (omit = server default)")
    p.add_argument("--size", default="800x800")
    p.add_argument("--duration", type=float, default=120, help="seconds")
    p.add_argument("--pause", default="2-6", help="think time seconds between a user's requests: 'lo-hi' or single value; 0 = none")
    p.add_argument("--warmup", type=float, default=15, help="seconds excluded from steady-state stats")
    p.add_argument("--prompt", default=None, help="fixed prompt (default: rotate a built-in set)")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--slo-p95", type=float, default=30.0)
    p.add_argument("--insecure", action="store_true", help="skip TLS verification")
    p.add_argument("--json", default=None, help="write summary JSON here")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

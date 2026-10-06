#!/usr/bin/env python3
############################################################
#
# mindrouter - live regression check: System One decisions on a
# vLLM chat model WHILE chat completions run on the same model.
#
# tests/decisions_bench.py measures an idle fleet, which is how the
# 2026-10-06 failure was missed: on replicas that use speculative
# decoding, vLLM answered HTTP 500 to the scoring request whenever
# another sequence was being decoded in the same step, so
# /v1/systemone on qwen/qwen3.8-27b failed for nearly every request
# under load and never when idle. (Cause and fix: the docstring of
# backend/app/services/decisions/vllm_logprobs.py.)
#
# Runs against a deployed MindRouter. Not a unit test.
#
#   MINDROUTER_API_KEY=mr2_... python tests/decisions_under_load.py \
#       --base-url https://mindrouter.uidaho.edu --model qwen/qwen3.8-27b
#
# For each kind of load (plain, json = structured output, thinking) it
# starts --loaders chat completions on --model, then sends --decisions
# yes/no decisions and as many 4-option choice decisions to the same
# model. One character per decision: "." answered, "X" failed.
# --control names a decision model that does not share those replicas
# (default "clef"; "" to skip) and is shown for comparison only.
#
# Exit status: 0 when every decision on --model was answered, 1 when any
# failed, 2 when the run proves nothing (no key, or a kind of load under
# which not one chat completion succeeded: the decisions then ran on an
# idle model, which is exactly what this check exists not to do).
#
# What it cannot see: the gateway retries a failed replica once on another
# replica, and the response does not say so. A failure masked that way shows
# only in the gateway log (decision_backend_retry_on_another_replica).
#
############################################################
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

STATE = "I want my money back."
QUESTIONS = {
    "noul": {"q": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    "choice": {"q": {"type": "choice", "instructions": "Which team should handle this message?",
                     "criteria": {"billing": "charges, refunds, invoices", "access": "passwords and logins",
                                  "shipping": "delivery and returns", "other": "anything else"}}},
}
LOADS = {
    "plain": {},
    "json": {"response_format": {"type": "json_schema", "json_schema": {"name": "trees", "schema": {
        "type": "object", "properties": {"trees": {"type": "array", "items": {"type": "string"}}},
        "required": ["trees"], "additionalProperties": False}}}},
    "thinking": {"think": True},
}


async def load(client: httpx.AsyncClient, args, extra: dict, stop: asyncio.Event, done: list) -> None:
    while not stop.is_set():
        try:
            r = await client.post(f"{args.base_url}/v1/chat/completions", headers=args.headers, timeout=180, json={
                "model": args.model, "max_tokens": 700, **extra,
                "messages": [{"role": "user", "content": "List twelve trees with one sentence about each."}]})
            status = r.status_code
        except httpx.HTTPError:
            status = 0
        done.append(status)
        if status != 200:
            # A refused request returns at once (rate limit, bad parameter):
            # without a pause this loop would hammer the gateway.
            await asyncio.sleep(2.0)


async def decide(client: httpx.AsyncClient, args, model: str, kind: str) -> tuple[bool, str]:
    try:
        r = await client.post(f"{args.base_url}/v1/systemone", headers=args.headers, timeout=60,
                              json={"model": model, "state": STATE, "questions": QUESTIONS[kind]})
    except httpx.HTTPError as e:
        return False, type(e).__name__
    if r.status_code != 200:
        return False, f"HTTP {r.status_code} {r.text[:120]}"
    try:
        body = r.json()
        answer = body["answers"]["q"]
    except (ValueError, KeyError, TypeError):
        return False, "HTTP 200 without an answer to the question"
    if (body.get("metadata") or {}).get("fallback"):
        # Answered, but by the fallback model: the model under test did not.
        return False, f"answered by fallback {body.get('model', '')}"
    if not isinstance(answer, dict) or answer.get("type") != kind:
        return False, "HTTP 200 with an answer of the wrong shape"
    return True, ""


async def phase(client: httpx.AsyncClient, args, name: str) -> tuple[int, int]:
    """(decisions on --model that failed, chat completions that succeeded meanwhile)."""
    stop, done = asyncio.Event(), []
    loaders = [asyncio.create_task(load(client, args, LOADS[name], stop, done)) for _ in range(args.loaders)]
    await asyncio.sleep(args.warmup)
    failures = 0
    models = [args.model] + ([args.control] if args.control else [])
    for model in models:
        for kind in QUESTIONS:
            marks, first_error = [], ""
            for _ in range(args.decisions):
                ok, why = await decide(client, args, model, kind)
                marks.append("." if ok else "X")
                first_error = first_error or why
                await asyncio.sleep(args.pause)
            failed = marks.count("X")
            if model == args.model:
                failures += failed
            note = f"   first failure: {first_error}" if failed else ""
            print(f"  load={name:<8} {model:<22} {kind:<6} {''.join(marks)}  {failed}/{len(marks)} failed{note}", flush=True)
    stop.set()
    await asyncio.gather(*loaders, return_exceptions=True)
    ok_load = sum(1 for s in done if s == 200)
    print(f"  load={name:<8} chat completions finished meanwhile: {ok_load} of {len(done)} ok", flush=True)
    return failures, ok_load


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="qwen/qwen3.8-27b", help="the vLLM chat model to load and to ask")
    p.add_argument("--control", default="clef", help='a decision model on other hardware, for comparison ("" = none)')
    p.add_argument("--loaders", type=int, default=6, help="concurrent chat completions on --model")
    p.add_argument("--decisions", type=int, default=24, help="decisions per kind, per model, per load")
    p.add_argument("--load", default="all", choices=["all", *LOADS], help="which kind of chat load")
    p.add_argument("--pause", type=float, default=0.5, help="seconds between decisions")
    p.add_argument("--warmup", type=float, default=4.0, help="seconds of load before the first decision")
    args = p.parse_args()
    if args.decisions < 1 or args.loaders < 1:
        p.error("--decisions and --loaders must be at least 1")
    key = os.environ.get("MINDROUTER_API_KEY") or os.environ.get("MR_KEY")
    if not key:
        print("set MINDROUTER_API_KEY", file=sys.stderr)
        return 2
    args.headers = {"Authorization": f"Bearer {key}"}
    args.base_url = args.base_url.rstrip("/")

    failures, unloaded = 0, []
    async with httpx.AsyncClient() as client:
        for name in (LOADS if args.load == "all" else [args.load]):
            failed, ok_load = await phase(client, args, name)
            failures += failed
            if not ok_load:
                unloaded.append(name)
    if failures:
        print(f"\nFAIL: {failures} decision(s) on {args.model} failed while chat completions were running on it")
        return 1
    if unloaded:
        print(f"\nINCONCLUSIVE: no chat completion succeeded under load {', '.join(unloaded)}, so those decisions "
              f"ran on an idle model. Check the key's rate limit and that {args.model} takes chat requests.")
        return 2
    print(f"\nPASS: every decision on {args.model} was answered while chat completions were running on it")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

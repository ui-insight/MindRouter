#!/usr/bin/env python3
############################################################
#
# mindrouter - live benchmark + behaviour probe for the EXPERIMENTAL
# System One API: POST /v1/systemone (services/decisions)
#
# Runs against a deployed MindRouter (default: local stack). Nothing here
# is a unit test; it needs the decisions API enabled and a model behind it.
# --model picks what answers: a vLLM model scored by letter likelihood
# (qwen3.8-27b) or an upstream decision server (laya), so the same script
# compares them.
#
#   MINDROUTER_API_KEY=mr2_... python tests/decisions_bench.py \
#       --base-url https://mindrouter.uidaho.edu --model qwen3.8-27b
#
# Sections (each can be selected with --only):
#   latency      single question / N questions sharing one large state
#   cache        prefix-cache reuse: cold vs warm (vLLM models only)
#   concurrency  moderate concurrent load (throughput, p50/p95)
#   stability    repeated runs: identical answers + probability drift
#   order        option-order sensitivity: as given vs reversed vs permutations=2
#   ambiguous    probabilities on deliberately ambiguous / underdetermined cases
#   calibration  accuracy + reliability on the LocalLLaMA/typed-decisions
#                benchmark (needs `pip install datasets`; --calib-cases N,
#                400 = the full test split = 2,000 decisions)
#
############################################################
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import httpx

STATE_SMALL = (
    "Ticket #8813 from a faculty member: 'Since yesterday's release the export "
    "button in the grant portal does nothing. I have a board meeting Monday and "
    "need the budget export.' Priority field left blank. Portal team on call."
)

# TypeSafe System One questions: a map of id -> {type, instructions, criteria}.
QUESTIONS = {
    "escalate": {"type": "noul", "instructions": "Should this ticket be escalated to an engineer today?"},
    "category": {"type": "choice", "instructions": "What kind of ticket is this?",
                 "criteria": {"bug": "Something that worked is broken", "feature request": "Asks for new behaviour",
                              "billing": "Charges, invoices, refunds", "question": "Asks how to do something"}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["Can wait", "This week", "Today", "Drop everything"]},
    "security": {"type": "noul", "instructions": "Does this ticket describe a security incident?"},
}

AMBIGUOUS = [
    ("The user wrote: 'It sort of works, mostly.'",
     {"broken": {"type": "noul", "instructions": "Is the feature broken?"}}),
    ("A coin was flipped once. The result was not recorded.",
     {"heads": {"type": "noul", "instructions": "Did the coin land heads?"}}),
    ("Message: 'Please handle this.' (no further detail)",
     {"dept": {"type": "choice", "instructions": "What department should handle this?",
               "criteria": {"billing": None, "engineering": None, "legal": None, "facilities": None}}}),
    ("Weather report: 21 C, light wind, partly cloudy.",
     {"umbrella": {"type": "noul", "instructions": "Will it rain within the hour?"}}),
]


def _big_state(n_chars: int) -> str:
    para = (
        "The research computing group operates a fleet of GPU nodes that serve "
        "language models to campus applications through a gateway. Requests carry "
        "an API key, are metered against a token budget, and are routed to a "
        "healthy backend that holds the requested model. Operators can enable or "
        "disable features per deployment. "
    )
    out = []
    while sum(len(p) for p in out) < n_chars:
        out.append(f"[section {len(out) + 1}] " + para)
    return "\n".join(out)[:n_chars]


def _repeat(questions: dict, times: int) -> dict:
    return {f"{qid}_{i}": q for i in range(times) for qid, q in questions.items()}


class Client:
    def __init__(self, base_url: str, key: str, model: str, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.model = model
        self.http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    async def ask(self, state, questions, permutations=1):
        body = {"model": self.model, "state": state, "questions": questions}
        if permutations != 1:
            body["permutations"] = permutations  # MindRouter extension; vLLM models only
        t0 = time.perf_counter()
        r = await self.http.post(f"{self.base}/v1/systemone", json=body)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json(), dt

    async def close(self):
        await self.http.aclose()


def _p(values, q):
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))
    return values[k]


def _label(answer: dict) -> str:
    """The single label an answer commits to, in the benchmark's vocabulary."""
    if answer["type"] == "noul":
        return "true" if answer["noul"] >= 0.5 else "false"
    if answer["type"] == "choice":
        return answer["choice"]
    probs = answer["probabilities"]
    return max(probs, key=probs.get)  # score: the most likely level index


def _top(answer: dict) -> float:
    """Probability of the label the answer commits to."""
    if answer["type"] == "noul":
        return max(answer["noul"], 1 - answer["noul"])
    return max(answer["probabilities"].values())


def _dist(answer: dict) -> str:
    if answer["type"] == "noul":
        return f"P(yes)={answer['noul']:.3f}"
    return json.dumps({k: round(v, 3) for k, v in answer["probabilities"].items()})


def _fmt(d):
    return {qid: (_label(a), round(_top(a), 3)) for qid, a in d["answers"].items()}


def _meta(d, key, default=None):
    return (d.get("metadata") or {}).get(key, default)


# ----------------------------------------------------------------- sections

async def sec_latency(c: Client, args):
    print("\n== latency: single question vs many questions sharing one state")
    one = {"escalate": QUESTIONS["escalate"]}
    for n_chars in (len(STATE_SMALL), 8_000, 24_000):
        state = STATE_SMALL if n_chars == len(STATE_SMALL) else _big_state(n_chars)
        for qs in (one, QUESTIONS, _repeat(QUESTIONS, 4)):
            times, d = [], None
            for _ in range(args.repeat):
                d, dt = await c.ask(state, qs)
                times.append(dt)
            print(f"  state={n_chars:>6} chars  questions={len(qs):>2}  "
                  f"p50={_p(times, .5)*1000:7.0f} ms  min={min(times)*1000:7.0f} ms  "
                  f"input_tokens={d['usage']['input_tokens']:>6}  prompt={_meta(d, 'prompt_tokens')}  "
                  f"cached={_meta(d, 'cached_tokens')}  calls={_meta(d, 'backend_calls')}")


async def sec_cache(c: Client, args):
    print("\n== prefix cache: same large state, cold then warm (vLLM models; cached = vLLM's own count)")
    state = _big_state(20_000) + f"\n[nonce {time.time()}]"  # unique -> cold on the first call
    for i in range(3):
        d, dt = await c.ask(state, QUESTIONS)
        print(f"  run {i}: {dt*1000:7.0f} ms  prompt_tokens={_meta(d, 'prompt_tokens')}  "
              f"cached_tokens={_meta(d, 'cached_tokens')}  billed input_tokens={d['usage']['input_tokens']}")
    print("  note: within ONE request the first question fills the cache and the rest reuse it;"
          " a warm run 0 means another replica already held the state.")


async def sec_concurrency(c: Client, args):
    print(f"\n== concurrency: {args.concurrency} parallel clients x {args.rounds} rounds, {len(QUESTIONS)} questions each")
    state = _big_state(4_000)
    lat = []

    async def worker(i):
        for r in range(args.rounds):
            s = state + f"\n[client {i} round {r}]"  # defeat cross-client cache sharing
            _, dt = await c.ask(s, QUESTIONS)
            lat.append(dt)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
    wall = time.perf_counter() - t0
    n = len(lat)
    print(f"  requests={n}  wall={wall:.1f} s  req/s={n/wall:.2f}  decisions/s={n*len(QUESTIONS)/wall:.1f}")
    print(f"  latency p50={_p(lat,.5)*1000:.0f} ms  p95={_p(lat,.95)*1000:.0f} ms  max={max(lat)*1000:.0f} ms")


async def sec_stability(c: Client, args):
    print(f"\n== stability: {args.repeat} repeated runs of the same request")
    runs = [(await c.ask(STATE_SMALL, QUESTIONS))[0] for _ in range(args.repeat)]
    for qid in QUESTIONS:
        answers = [run["answers"][qid] for run in runs]
        tops = [_top(a) for a in answers]
        masses = [(_meta(run, "questions") or {}).get(qid, {}).get("label_mass") for run in runs]
        mass = f"  label_mass={statistics.mean(masses):.3f}" if all(m is not None for m in masses) else ""
        print(f"  {qid:<10} labels={sorted({_label(a) for a in answers})}  top-probability mean={statistics.mean(tops):.4f} "
              f"spread={max(tops)-min(tops):.5f}{mass}")


async def sec_order(c: Client, args):
    print("\n== option order: as given / reversed / permutations=2 (vLLM models; an upstream ignores permutations)")
    for qid in ("category", "urgency"):
        q = QUESTIONS[qid]
        crit = q["criteria"]
        if isinstance(crit, dict):
            rq = {**q, "criteria": dict(reversed(list(crit.items())))}
        else:
            # A score's meaning is its order, so reversing levels changes the question; skip it.
            rq = None
        given, _ = await c.ask(STATE_SMALL, {qid: q})
        print(f"  {qid:<10} given={_label(given['answers'][qid]):<16} {_dist(given['answers'][qid])}")
        if rq:
            rev, _ = await c.ask(STATE_SMALL, {qid: rq})
            print(f"  {'':<10} rev  ={_label(rev['answers'][qid]):<16} {_dist(rev['answers'][qid])}")
        avg, _ = await c.ask(STATE_SMALL, {qid: q}, permutations=2)
        print(f"  {'':<10} perm2={_label(avg['answers'][qid]):<16} {_dist(avg['answers'][qid])}")


async def sec_ambiguous(c: Client, args):
    print("\n== ambiguous cases: a well-behaved reader should NOT be confident here")
    for state, qs in AMBIGUOUS:
        d, _ = await c.ask(state, qs)
        for qid, a in d["answers"].items():
            conf = f"  confidence={a['confidence']:.3f}" if "confidence" in a else ""
            print(f"  {qid:<9} label={_label(a):<12} {_dist(a)}{conf}")


async def sec_calibration(c: Client, args):
    print(f"\n== LocalLLaMA/typed-decisions, config 'all', test split ({args.calib_cases} cases x 5 questions)")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  pip install datasets   (skipped)")
        return
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    confs, hits, by_type = [], [], {}
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i, case):
        # The dataset is already in System One shape: state and questions are
        # JSON strings; gold[qid]["label"] is the reference label.
        state, questions, gold = json.loads(case["state"]), json.loads(case["questions"]), json.loads(case["gold"])
        async with sem:
            try:
                d, _ = await c.ask(state, questions)
            except Exception as e:
                print(f"  case {i}: {e}")
                return
        for qid, a in d["answers"].items():
            hit = _label(a) == str(gold[qid]["label"])
            confs.append(_top(a))
            hits.append(hit)
            by_type.setdefault(a["type"], []).append(hit)

    await asyncio.gather(*(one(i, case) for i, case in enumerate(ds) if i < args.calib_cases))
    if not hits:
        print("  no decisions scored")
        return
    print(f"  decisions={len(hits)}  accuracy={sum(hits)/len(hits):.3f}  "
          + "  ".join(f"{t}={sum(v)/len(v):.3f} (n={len(v)})" for t, v in sorted(by_type.items()))
          + f"  mean top-probability={statistics.mean(confs):.3f}")
    print("  published on this benchmark: Jev 1.13.0 0.727, teacher self-agreement ceiling 0.735")
    print("  reliability (top-probability bin -> accuracy):")
    bins = [[] for _ in range(10)]
    for cf, h in zip(confs, hits, strict=True):
        bins[min(9, int(cf * 10))].append((cf, h))
    ece = 0.0
    for b, xs in enumerate(bins):
        if xs:
            bin_acc = sum(h for _, h in xs) / len(xs)
            bin_conf = statistics.mean(cf for cf, _ in xs)
            ece += len(xs) / len(hits) * abs(bin_acc - bin_conf)
            print(f"    [{b/10:.1f},{(b+1)/10:.1f})  n={len(xs):>4}  acc={bin_acc:.3f}  conf={bin_conf:.3f}")
    print(f"  ECE={ece:.4f}")


SECTIONS = {
    "latency": sec_latency, "cache": sec_cache, "concurrency": sec_concurrency,
    "stability": sec_stability, "order": sec_order, "ambiguous": sec_ambiguous,
    "calibration": sec_calibration,
}


async def main():
    ap = argparse.ArgumentParser(description="Live benchmark for MindRouter's System One API")
    ap.add_argument("--base-url", default=os.environ.get("MINDROUTER_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("MINDROUTER_API_KEY"))
    ap.add_argument("--model", default="jev-latest",
                    help="jev-latest (the server's default), a vLLM model such as qwen3.8-27b, or an upstream such as laya")
    ap.add_argument("--only", default="latency,cache,concurrency,stability,order,ambiguous",
                    help="comma-separated sections (add 'calibration' explicitly)")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--calib-cases", type=int, default=100)
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("set MINDROUTER_API_KEY or pass --api-key")
    c = Client(args.base_url, args.api_key, args.model)
    try:
        d, dt = await c.ask(STATE_SMALL, QUESTIONS)
        print(f"smoke: {dt*1000:.0f} ms  answered by {d['model']} via {_meta(d, 'backend')}  {_fmt(d)}  "
              f"semantics={_meta(d, 'score_semantics')}")
        for name in args.only.split(","):
            await SECTIONS[name.strip()](c, args)
    finally:
        await c.close()


if __name__ == "__main__":
    asyncio.run(main())

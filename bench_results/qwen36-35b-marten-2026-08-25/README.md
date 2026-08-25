# qwen3.6-35b (A3B MoE FP8, MTP-3) — chat capacity on marten-gpu0, 2026-08-25

Same suite as `../qwen38-vs-qwen36-2026-08-22` (chat_bench.py from vandalchat@chat.uidaho.edu,
direct mode to https://marten.hpc.uidaho.edu:8001, backend 1 disabled in MindRouter for the run,
SLOs: TTFT p95 <= 2 s, tok/s p10 >= 15, err <= 1 %, stalls > 2 s <= 2 %, e2e p95 <= 139 s).
Pass 1 at the fleet setting (vLLM --max-num-seqs 96 / MindRouter 72), pass 2 at 256 (unit
`.seqs256`, backup `.bak-seqs96-20260825`).

**Warm-up caveat:** a freshly started instance JIT-compiles ~92 DeepGEMM kernels on first use
(engine freezes for seconds each) — pass-1 stages 16 and 32 were contaminated (stall gate) and are
superseded by pass-2's clean re-run after a discarded 256-stream warm-up stage.

## Saturation (users == streams)

| streams | cap 96: TTFT p95 / tps p10 / agg | cap 256: TTFT p95 / tps p10 / agg / KV |
|---|---|---|
| 16  | (JIT-contaminated) | 0.24 s / 88 / 1,659 / 8 % |
| 32  | (JIT-contaminated) | 0.26 s / 71 / 2,823 / 14 % |
| 48  | 0.25 s / 73 / 4,136 | — |
| 64  | 0.29 s / 64 / 4,668 | — |
| 96  | 0.34 s / 49 / 5,470 (KV 33 %) PASS | 0.35 s / 46 / 5,309 / 34 % |
| 112 | 2.73 s FAIL (16 queued) | — |
| 128 | 4.89 s FAIL (32 queued) | 0.40 s / 36 / 5,666 / 44 % |
| 160 | — | 0.37 s / 29 / 5,346 / 55 % |
| 192 | — | 0.35 s / 27 / 5,914 / 66 % |
| 224 | — | 0.44 s / 26 / 6,541 / 76 % |
| 256 | — | 0.67 s / 26 / 7,484 / 85 % PASS |

## Realistic (multi-turn users, think times)

| users | cap 96 | cap 256 |
|---|---|---|
| 128–320 | PASS, TTFT p95 0.17–0.23 s, up to 3,212 tok/s | — |
| 480 | PASS at the edge: TTFT p95 1.66 s, 72 running, 148 queued | PASS: TTFT p95 0.52 s, 112 running (peak 255), 69 queued, KV max 81 % |
| 600 | FAIL: TTFT p95 76 s, 363 queued | FAIL: TTFT p95 64 s, 294 queued, KV max 83 %, 20 % stalls |
| 720 | FAIL: 97 s | FAIL: 69 s |
| 960 | — | FAIL: 90 s, 533 queued |

## Findings
1. At the fleet cap (96) every SLO miss is the cap: KV 33 %, throughput still rising, tps p10 3x the gate.
2. Saturation with cap 256 passes all the way to 256 streams (7.5k tok/s, 2.7x the 27B models' plateau)
   with zero preemptions; KV reaches 85 % at 256 — the next step would preempt.
3. The realistic knee is **480 users per GPU with either cap** — beyond it KV (81–83 % max, long
   multi-turn histories, prefix-cache hit rate ~1 % on this hybrid model) caps admissions and the
   queue explodes. Raising the cap buys latency at the knee (TTFT p95 1.66 -> 0.52 s at 480) and
   burst/API headroom, not more concurrent chat users; per-stream speed at the knee drops 59 -> 30 tok/s.
4. Recommendation: vLLM 192 / MindRouter 144 (KV <= 66 % at saturation, tps p10 27, queue cliff
   removed up to 192 streams; 256/192 leaves no KV margin for long-context users).

## Change log — 2026-08-25 06:42–07:52 PDT: qwen3.6-35b fleet moved to 192/144
All three replicas (marten-gpu0 b1, lynx-gpu2 b8, aspen5-gpu3 b47) rolled one at a time:
MindRouter disable → drain → unit backup `.bak-seqs96-20260825` (marten also `.bak-seqs256-20260825`)
→ `--max-num-seqs 192` → restart (110–130 s to healthy) → 64-request warm-up burst → PATCH
`max_concurrent 144` → enable. Zero restarts, all healthy; 6/6 gateway checks OK. Fleet total 432 slots.

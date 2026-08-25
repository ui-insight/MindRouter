#!/bin/bash
cd ~/bench
V=~/bench/venv/bin/python
BASE=https://marten.hpc.uidaho.edu:8001; MODEL=qwen/qwen3.6-35b; L=q36-35b-marten-s256
rm -rf "$L"; mkdir -p "$L"
for i in $(seq 1 120); do curl -sf -m 5 $BASE/health >/dev/null && break; sleep 10; done
echo "healthy $(date -u)" > "$L/START"
# JIT warm-up: one 256-stream stage, results discarded
$V chat_bench.py --base-url "$BASE" --model "$MODEL" --metrics-url "$BASE/metrics" \
   --users 256 --no-adapt --no-think-time --stage-duration 120 --warmup 20 --min-turns 20 --cooldown 10 --seed 7 \
   --outdir "$L/warmup" --label "$L-warmup" > "$L/warmup.log" 2>&1
$V chat_bench.py --base-url "$BASE" --model "$MODEL" --metrics-url "$BASE/metrics" \
   --users 16,32,96,128,160,192,224,256 --no-adapt --no-think-time --stage-duration 120 --warmup 20 \
   --min-turns 20 --cooldown 10 --seed 42 \
   --outdir "$L/saturation" --label "$L-saturation" > "$L/saturation.log" 2>&1
$V chat_bench.py --base-url "$BASE" --model "$MODEL" --metrics-url "$BASE/metrics" \
   --users 480,600,720,960 --max-users 1536 --stage-duration 180 --warmup 30 \
   --min-turns 30 --cooldown 10 --seed 42 \
   --outdir "$L/realistic" --label "$L-realistic" > "$L/realistic.log" 2>&1
touch "$L/DONE"

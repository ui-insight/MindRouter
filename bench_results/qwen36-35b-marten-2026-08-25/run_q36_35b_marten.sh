#!/bin/bash
# qwen3.6-35b capacity on marten-gpu0 (backend 1 disabled in MindRouter during the run).
cd ~/bench
V=~/bench/venv/bin/python
BASE=https://marten.hpc.uidaho.edu:8001; MODEL=qwen/qwen3.6-35b; L=q36-35b-marten-s96
rm -rf "$L"; mkdir -p "$L"
# wait for live traffic to drain off the backend
for i in $(seq 1 60); do r=$(curl -s $BASE/metrics | grep -E "^vllm:num_requests_running\{" | awk "{print \$2}"); [ "${r%.*}" = "0" ] && break; sleep 5; done
echo "drained (running=$r) $(date -u)" > "$L/START"
$V chat_bench.py --base-url "$BASE" --model "$MODEL" --metrics-url "$BASE/metrics" \
   --users 16,32,48,64,80,96,112,128 --no-adapt --no-think-time --stage-duration 120 --warmup 20 \
   --min-turns 20 --cooldown 10 --seed 42 \
   --outdir "$L/saturation" --label "$L-saturation" > "$L/saturation.log" 2>&1
$V chat_bench.py --base-url "$BASE" --model "$MODEL" --metrics-url "$BASE/metrics" \
   --users 128,192,256,320 --max-users 768 --stage-duration 180 --warmup 30 \
   --min-turns 30 --cooldown 10 --seed 42 \
   --outdir "$L/realistic" --label "$L-realistic" > "$L/realistic.log" 2>&1
touch "$L/DONE"

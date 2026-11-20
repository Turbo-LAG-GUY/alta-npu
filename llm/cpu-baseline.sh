#!/usr/bin/env bash
# cpu-baseline.sh - LLM engine CPU baseline (2026-11-19, CPU only, no NPU): llama.cpp (b5dbca1, ARMv8.0 NEON,
# no dotprod) speed for the SmolLM family, the bar the NPU engine has to beat.
# pp128 = reading a 128-token prompt, tg64 = generating 64 tokens, tokens/s; 4 threads on the A73 cores (cpu 2-5),
# plus 6 threads (all cores) for the 135M model. SmolLM3-3B Q8_0 (3.1 GB) does not fit in free RAM: short run only.
# Usage: llm/cpu-baseline.sh > npu-llm/runs/cpu-baseline-<stamp>.log
set -u
B=/home/jordan/Documents/AI_Projects/llama.cpp/build/bin/llama-bench
G=/home/jordan/Documents/AI_Projects/npu-llm/models/gguf
free -m | sed -n 1,3p
for m in SmolLM2-135M-Instruct-Q8_0 SmolLM2-360M-Instruct-Q8_0 SmolLM2-1.7B-Instruct-Q8_0 SmolLM3-Q4_K_M; do
   echo "== $m"
   taskset -c 2-5 $B -m $G/$m.gguf -t 4 -p 128 -n 64 -r 3 2>&1 | grep -E '^\|'
done
echo "== SmolLM2-135M-Instruct-Q8_0, 6 threads (A53 + A73)"
$B -m $G/SmolLM2-135M-Instruct-Q8_0.gguf -t 6 -p 128 -n 64 -r 3 2>&1 | grep -E '^\|'
echo "== SmolLM3-Q8_0 (bigger than free RAM: pages from the SD card), short run"
free -m | sed -n 2p
taskset -c 2-5 timeout 900 $B -m $G/SmolLM3-Q8_0.gguf -t 4 -p 16 -n 8 -r 1 2>&1 | grep -E '^\|'
free -m | sed -n 2,3p

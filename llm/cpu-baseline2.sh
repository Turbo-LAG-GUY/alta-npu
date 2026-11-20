#!/usr/bin/env bash
# cpu-baseline2.sh - second CPU baseline pass (2026-11-19, CPU only): weights loaded into RAM before timing
# (-lm none; -mmp in older versions), because in pass 1 the 1.7B / 3B files were partly evicted between repetitions and pp128 measured SD-card
# reads (~47 MB/s). 5 repetitions. Usage: llm/cpu-baseline2.sh > npu-llm/runs/cpu-baseline2-<stamp>.log
set -u
B=/home/jordan/Documents/AI_Projects/llama.cpp/build/bin/llama-bench
G=/home/jordan/Documents/AI_Projects/npu-llm/models/gguf
for m in SmolLM2-135M-Instruct-Q8_0 SmolLM2-360M-Instruct-Q8_0 SmolLM2-1.7B-Instruct-Q8_0 SmolLM3-Q4_K_M; do
   echo "== $m   $(free -m | awk '/Mem:/ {print "available " $7 " MB"}')"
   taskset -c 2-5 $B -m $G/$m.gguf -lm none -t 4 -p 128 -n 64 -r 5 2>&1 | grep -E '^\|'
done

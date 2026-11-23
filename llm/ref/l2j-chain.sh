#!/usr/bin/env bash
# l2j-chain.sh (2026-11-22, CPU only): down input split quality, 135M (K 2,3) then 1.7B (K 2,4).
set -u
AI=/home/jordan/Documents/AI_Projects
P=/home/jordan/alta-npu/scripts/progress
for spec in "SmolLM2-135M-Instruct 2,3" "SmolLM2-1.7B-Instruct 2,4"; do
   set -- $spec
   $P llm-l2j STARTING "$1 K $2: llm/ref/l2j_down_split.py"
   L2J_K=$2 OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u /home/jordan/alta-npu/llm/ref/l2j_down_split.py \
      $AI/npu-llm/models/$1 256 > $AI/npu-llm/runs/l2j-$1.log 2>&1
   $P llm-l2j DONE "$1 rc=$?: $(tail -n 1 $AI/npu-llm/runs/l2j-$1.log)"
done

#!/usr/bin/env bash
# l2g-chain.sh (2026-11-20, CPU only): fixes for the NPU's 8-bit activation outliers (llm/ref/l2g_outliers.py: per-job
# output scales, SmoothQuant, rotation) on the SmolLM models, one after the other (one heavy job at a time):
# 135M, SmolLM2-1.7B (the broken one), 360M, SmolLM3-3B. Waits first until the l2f quality run (unit alta-llm-l2f)
# has finished. Roughly 6-8 hours in all (1.7B and SmolLM3 dominate).
# Run detached: systemd-run --user --unit=alta-llm-l2g --collect llm/l2g-chain.sh [MODEL ...]  (default: all four)
set -u
AI=/home/jordan/Documents/AI_Projects
cd $AI || exit 1
while systemctl --user is-active --quiet alta-llm-l2f; do sleep 60; done
for m in ${@:-SmolLM2-135M-Instruct SmolLM2-1.7B-Instruct SmolLM2-360M-Instruct SmolLM3-3B}; do
   log=$AI/npu-llm/runs/l2g-outliers-$m.log
   /home/jordan/alta-npu/scripts/progress llm-l2g STARTING "$m: llm/ref/l2g_outliers.py, log $log"
   OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u /home/jordan/alta-npu/llm/ref/l2g_outliers.py \
      $AI/npu-llm/models/$m 256 2>&1 | grep --line-buffered -vE '^INFO|PyTorch' > "$log"
   rc=${PIPESTATUS[0]}
   /home/jordan/alta-npu/scripts/progress llm-l2g DONE "$m rc=$rc: $(tail -n 1 "$log")"
done

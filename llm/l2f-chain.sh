#!/usr/bin/env bash
# l2f-chain.sh (2026-11-19, CPU only): CPU + NPU split quality (llm/ref/l2f_hybrid.py) on the SmolLM models, one after
# the other (one heavy job at a time): 135M, 360M, SmolLM2-1.7B, SmolLM3-3B. About 2.5 hours.
# Run detached: systemd-run --user --unit=alta-llm-l2f --collect llm/l2f-chain.sh [MODEL ...]  (default: all four)
set -u
AI=/home/jordan/Documents/AI_Projects
cd $AI || exit 1
for m in ${@:-SmolLM2-135M-Instruct SmolLM2-360M-Instruct SmolLM2-1.7B-Instruct SmolLM3-3B}; do
   log=$AI/npu-llm/runs/l2f-hybrid-$m.log
   /home/jordan/alta-npu/scripts/progress llm-l2f STARTING "$m: llm/ref/l2f_hybrid.py, log $log"
   OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u /home/jordan/alta-npu/llm/ref/l2f_hybrid.py \
      $AI/npu-llm/models/$m 256 2>&1 | grep --line-buffered -vE '^INFO|PyTorch' > "$log"
   rc=${PIPESTATUS[0]}
   /home/jordan/alta-npu/scripts/progress llm-l2f DONE "$m rc=$rc: $(tail -n 1 "$log")"
done

#!/usr/bin/env bash
# l2e-chain.sh (2026-11-19, CPU only): weight channel groups (llm/ref/l2e_groups.py) on the SmolLM models, one after
# the other (one heavy job at a time): 135M, 360M, SmolLM3-3B, then SmolLM2-1.7B. About 3.5 hours.
# Run detached: systemd-run --user --unit=alta-llm-l2e --collect llm/l2e-chain.sh [MODEL ...]  (default: all four)
set -u
AI=/home/jordan/Documents/AI_Projects
cd $AI || exit 1
for m in ${@:-SmolLM2-135M-Instruct SmolLM2-360M-Instruct SmolLM3-3B SmolLM2-1.7B-Instruct}; do
   log=$AI/npu-llm/runs/l2e-groups-$m.log
   /home/jordan/alta-npu/scripts/progress llm-l2e STARTING "$m: llm/ref/l2e_groups.py, log $log"
   OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u /home/jordan/alta-npu/llm/ref/l2e_groups.py \
      $AI/npu-llm/models/$m 256 2>&1 | grep --line-buffered -vE '^INFO|PyTorch' > "$log"
   rc=${PIPESTATUS[0]}
   /home/jordan/alta-npu/scripts/progress llm-l2e DONE "$m rc=$rc: $(tail -n 1 "$log")"
done

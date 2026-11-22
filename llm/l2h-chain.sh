#!/usr/bin/env bash
# l2h-chain.sh (2026-11-21, CPU only): jobs per layer step 4 / 2 / 1 in the fixed 8-bit format (llm/ref/l2h_groups.py)
# on 135M, 360M and SmolLM2-1.7B, one after the other. 135M + 360M ~20 min, 1.7B ~2.5 h.
# Run detached: systemd-run --user --unit=alta-llm-l2h --collect llm/l2h-chain.sh [MODEL ...]
set -u
AI=/home/jordan/Documents/AI_Projects
cd $AI || exit 1
for m in ${@:-SmolLM2-135M-Instruct SmolLM2-360M-Instruct SmolLM2-1.7B-Instruct}; do
   log=$AI/npu-llm/runs/l2h-groups-$m.log
   /home/jordan/alta-npu/scripts/progress llm-l2h STARTING "$m: llm/ref/l2h_groups.py, log $log"
   OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u /home/jordan/alta-npu/llm/ref/l2h_groups.py \
      $AI/npu-llm/models/$m 256 2>&1 | grep --line-buffered -vE '^INFO|PyTorch' > "$log"
   rc=${PIPESTATUS[0]}
   /home/jordan/alta-npu/scripts/progress llm-l2h DONE "$m rc=$rc: $(tail -n 1 "$log")"
done

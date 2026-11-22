#!/usr/bin/env bash
# l2h-mixed-chain.sh (2026-11-21, CPU only): the mixed jobs-per-step layout (qkv 2, o 1, gateup 4, down 2: each job as
# wide as fits the 512 KiB VIP SRAM, runs/llm-wide-ladder-20261121-211302) vs 4 jobs in the same run (and 1 job on
# 1.7B, the row the reboot stopped), llm/ref/l2h_groups.py with L2H_ROWS. 135M done (npu-llm/runs/l2h-mixed-*.log).
# Run detached: systemd-run --user --unit=alta-llm-l2h-mixed --collect llm/l2h-mixed-chain.sh [MODEL:ROWS ...]
set -u
AI=/home/jordan/Documents/AI_Projects
cd $AI || exit 1
for spec in ${@:-SmolLM2-360M-Instruct:4,mixed SmolLM2-1.7B-Instruct:4,1,mixed}; do
   m=${spec%%:*}
   rows=${spec#*:}
   log=$AI/npu-llm/runs/l2h-mixed-$m.log
   /home/jordan/alta-npu/scripts/progress llm-l2h-mixed STARTING "$m rows $rows: llm/ref/l2h_groups.py, log $log"
   L2H_ROWS=$rows OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u /home/jordan/alta-npu/llm/ref/l2h_groups.py \
      $AI/npu-llm/models/$m 256 > "$log.raw" 2>&1
   rc=$?
   grep -vE '^INFO|PyTorch' "$log.raw" > "$log" && rm -f "$log.raw"
   /home/jordan/alta-npu/scripts/progress llm-l2h-mixed DONE "$m rc=$rc: $(tail -n 1 "$log")"
   [ "$rc" = 0 ] || exit 1
done

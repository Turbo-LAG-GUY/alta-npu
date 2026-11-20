#!/usr/bin/env bash
# l2-big-chain.sh (2026-11-19, CPU only): L2 quality checks of SmolLM2-1.7B then SmolLM3-3B, one after the other
# (one heavy job at a time). Run detached: systemd-run --user --unit=alta-llm-l2big --collect llm/l2-big-chain.sh
set -u
AI=/home/jordan/Documents/AI_Projects
cd $AI || exit 1
for m in SmolLM2-1.7B-Instruct SmolLM3-3B; do
   /home/jordan/alta-npu/scripts/progress llm-l2-big STARTING "$m: llm/ref/l2_big.py, log npu-llm/runs/l2-big-$m.log"
   OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u /home/jordan/alta-npu/llm/ref/l2_big.py \
      $AI/npu-llm/models/$m 256 2>&1 | grep --line-buffered -vE '^INFO|PyTorch' > $AI/npu-llm/runs/l2-big-$m.log
   /home/jordan/alta-npu/scripts/progress llm-l2-big DONE "$m rc=${PIPESTATUS[0]}: $(tail -n 1 $AI/npu-llm/runs/l2-big-$m.log)"
done

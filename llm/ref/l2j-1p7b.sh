#!/usr/bin/env bash
# l2j-1p7b.sh (2026-11-22, CPU only): down input split quality on SmolLM2-1.7B (K 2, 4), llm/ref/l2j_down_split.py.
set -u
AI=/home/jordan/Documents/AI_Projects
P=/home/jordan/alta-npu/scripts/progress
LOG=$AI/npu-llm/runs/l2j-SmolLM2-1.7B-Instruct.log
$P llm-l2j STARTING "SmolLM2-1.7B K 2,4: llm/ref/l2j_down_split.py, log $LOG"
L2J_K=2,4 OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u /home/jordan/alta-npu/llm/ref/l2j_down_split.py \
   $AI/npu-llm/models/SmolLM2-1.7B-Instruct 256 > "$LOG" 2>&1
$P llm-l2j DONE "SmolLM2-1.7B rc=$?: $(tail -n 1 "$LOG")"

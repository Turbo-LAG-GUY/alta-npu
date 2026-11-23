#!/usr/bin/env bash
# l2k-1p7b.sh (2026-11-23, CPU only): which 1.7B step kind gains most from 16 bits (llm/ref/l2k_int16.py, ~5-6 h).
set -u
AI=/home/jordan/Documents/AI_Projects
P=/home/jordan/alta-npu/scripts/progress
LOG=$AI/npu-llm/runs/l2k-int16-SmolLM2-1.7B-Instruct.log
$P llm-l2k STARTING "SmolLM2-1.7B int16 rows base,qkv,o,gateup,down,all: llm/ref/l2k_int16.py, log $LOG"
OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u /home/jordan/alta-npu/llm/ref/l2k_int16.py \
   $AI/npu-llm/models/SmolLM2-1.7B-Instruct 256 > "$LOG" 2>&1
$P llm-l2k DONE "SmolLM2-1.7B rc=$?: $(tail -n 1 "$LOG")"

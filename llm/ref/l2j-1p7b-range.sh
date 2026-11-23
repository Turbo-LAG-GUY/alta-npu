#!/usr/bin/env bash
# l2j-1p7b-range.sh (2026-11-23, CPU only): 1.7B down split with RANGE-grouped jobs (DP, min 8 rows per job),
# 4 and 8 jobs per piece, unsplit vs 4 pieces (weight-grouped baseline: K=1 .182/.089, K=4 .251/.125).
set -u
AI=/home/jordan/Documents/AI_Projects
P=/home/jordan/alta-npu/scripts/progress
LOG=$AI/npu-llm/runs/l2j-SmolLM2-1.7B-Instruct-range.log
$P llm-l2j STARTING "SmolLM2-1.7B range-grouped jobs 4,8 x K 1,4: llm/ref/l2j_down_split.py, log $LOG"
L2J_GROUP=range L2J_JOBS=4,8 L2J_KS=1,4 OPENBLAS_NUM_THREADS=4 nice -n 5 $AI/.venv/bin/python -I -u \
   /home/jordan/alta-npu/llm/ref/l2j_down_split.py $AI/npu-llm/models/SmolLM2-1.7B-Instruct 256 > "$LOG" 2>&1
$P llm-l2j DONE "SmolLM2-1.7B range rc=$?: $(tail -n 1 "$LOG")"

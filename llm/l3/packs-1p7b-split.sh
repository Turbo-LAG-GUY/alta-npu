#!/usr/bin/env bash
# packs-1p7b-split.sh (2026-11-23, CPU only): SmolLM2-1.7B packs, 4 jobs per step, the down step's input split into
# 4 pieces of 2048 with CALIBRATED partial scales and range-grouped jobs (real_packs.py DOWN_SPLIT=4range) -> npu-llm/models/jb/real-1p7b-split4.
# Waits for the 1.7B down-split quality run (unit alta-llm-l2j-1p7b) first: two 1.7B processes don't fit the RAM.
# Run detached: systemd-run --user --unit=alta-llm-packs-1p7b-split --collect llm/l3/packs-1p7b-split.sh
set -u
AI=/home/jordan/Documents/AI_Projects
K=/home/jordan/alta-npu
LOG=$AI/npu-llm/runs/real-packs-1p7b-split4.log
export OPENBLAS_CORETYPE=CORTEXA73 OPENBLAS_NUM_THREADS=4
while systemctl --user is-active -q alta-llm-l2j-range; do sleep 30; done
cd "$K" || exit 1
scripts/progress llm-packs-1p7b-split STARTING "real_packs.py SmolLM2-1.7B 4 jobs, down split 4range -> npu-llm/models/jb/real-1p7b-split4, log $LOG"
nice -n 5 $AI/.venv/bin/python -I -u $K/llm/l3/real_packs.py $AI/npu-llm/models/SmolLM2-1.7B-Instruct \
   $AI/npu-llm/models/jb/real-1p7b-split4 4 4range > "$LOG" 2>&1
rc=$?
scripts/progress llm-packs-1p7b-split DONE "rc=$rc: $(tail -n 1 "$LOG")"

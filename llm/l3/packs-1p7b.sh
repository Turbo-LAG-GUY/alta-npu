#!/usr/bin/env bash
# packs-1p7b.sh (2026-11-22, CPU only): real-weight SmolLM2-1.7B packs, 4 jobs per step (l2h: 4 jobs 2.52 / 25.38,
# the SRAM layout 2.48 / 25.87; fewer jobs = fewer input reads with the tile rule), 1 and 256 tokens.
# Run detached: systemd-run --user --unit=alta-llm-packs-1p7b --collect llm/l3/packs-1p7b.sh
set -u
AI=/home/jordan/Documents/AI_Projects
K=/home/jordan/alta-npu
LOG=$AI/npu-llm/runs/real-packs-1p7b.log
export OPENBLAS_CORETYPE=CORTEXA73 OPENBLAS_NUM_THREADS=4
cd "$K" || exit 1
scripts/progress llm-packs-1p7b STARTING "real_packs.py SmolLM2-1.7B 4 jobs -> npu-llm/models/jb/real-1p7b, log $LOG"
nice -n 5 $AI/.venv/bin/python -I -u $K/llm/l3/real_packs.py $AI/npu-llm/models/SmolLM2-1.7B-Instruct \
   $AI/npu-llm/models/jb/real-1p7b 4 > "$LOG" 2>&1
rc=$?
scripts/progress llm-packs-1p7b DONE "rc=$rc: $(tail -n 1 "$LOG")"

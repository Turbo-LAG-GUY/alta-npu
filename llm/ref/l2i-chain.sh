#!/usr/bin/env bash
# l2i-chain.sh (2026-11-22, CPU only): FFN-on-NPU quality variants on 135M (llm/ref/l2i_npu_mlp.py), one after another.
set -u
AI=/home/jordan/Documents/AI_Projects
for spec in "w 8" "w 16"; do
   set -- $spec
   L2I_PAIR_BY=$1 L2I_JOBS=$2 L2I_ROWS=cpu,npu,hadconv OPENBLAS_NUM_THREADS=2 nice -n 5 $AI/.venv/bin/python -I -u \
      /home/jordan/alta-npu/llm/ref/l2i_npu_mlp.py $AI/npu-llm/models/SmolLM2-135M-Instruct 256 \
      > $AI/npu-llm/runs/l2i-135m-pairs$1-jobs$2.log 2>&1
done

#!/usr/bin/env bash
# llm-l1-ladder.sh - LLM engine step L1: real LLM matrix shapes as int8 1x1 convs on the NPU (the workaround for
# FULLY_CONNECTED, which hangs this NPU). Written 2026-11-18 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS.
# Questions: (a) are wide 1x1 convs correct (vs the CPU, XNNPACK; NN rounding allows <= 1 level),
#            (b) how fast does the NPU stream LLM weights (invoke time vs weight bytes = effective GB/s).
# Models: npu-llm/models/shapes (llm/tools/llm_shapes.py), per-tensor int8, LLM-like weights.
# Offline (fake NPU, npu-llm/runs/offline-20261118): all 18 = one TP + one NN job (+ a TP after for M = 16),
# no driver error, weights uncompressed (zrl_bits 0); M = 16 tile 1x16 = the pattern of the proven conv_1x64x384.
# Order: smallest weights first, one token (M = 1) before 16; stops at the first failure or STOP.
# Usage: llm/llm-l1-ladder.sh   (run detached: systemd-run --user --unit=alta-llm-l1 --collect llm/llm-l1-ladder.sh)
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
CHECK=$AI/npu-shader/teflon_check.py
LIB=$AI/mesa-upstream-fixes/build/src/gallium/targets/teflon/libteflon.so
M=$AI/npu-llm/models/shapes
RD=$K/runs/llm-l1-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$LIB" | cut -c1-8)" = b6c2a28d ] || { echo "upstream lib is not b6c2a28d"; exit 1; }

run() { # NAME ITERATIONS
   local n=$1 it=$2 label="LLM1-$1"
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM \
      scripts/npu-attempt "$label" -- $PY -I $CHECK -m $M/$n.tflite -r $M/${n}_reference.npz -e $LIB -n $it \
      --save "$RD/$n.npy" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc $(grep -oE '"invoke_ms_median_after_first": [0-9.]+' "$RD/$label.stdout")"
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-l1-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}
scripts/progress llm-l1-ladder STARTING "llm/llm-l1-ladder.sh: $RD"
for m in 1 16; do
   run smol_kv_576x192_m$m 20
   run smol_q_576x576_m$m 20
   run smol_qkv_576x960_m$m 20
   run smol_down_1536x576_m$m 20
   run smol_gateup_576x3072_m$m 20
   run qwen_o_2048x1024_m$m 20
   run qwen_down_3072x1024_m$m 20
   run qwen_qkv_1024x4096_m$m 20
   run qwen_gateup_1024x6144_m$m 20
done
scripts/progress llm-l1-ladder DONE "all 18 rungs passed: $RD"
echo "ALL PASSED: $RD"

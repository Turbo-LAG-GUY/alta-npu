#!/usr/bin/env bash
# llm-l1b-ladder.sh - LLM engine step L1b: which layout makes 16 (or 64) tokens cheap on the NN unit.
# Written 2026-11-19 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS.
# Models: npu-llm/models/layouts (llm/tools/llm_layouts.py): 3 real shapes (smol_gateup 576->3072, qwen_gateup
# 1024->6144, qwen_down 3072->1024) x 8 layouts, same weights and tokens per shape. CPU references = XNNPACK.
# Offline (fake NPU, npu-llm/runs/offline-l1b-20261119): all 24 accepted, no driver error; planar = one NN job;
# sq tile 4x4 (kpc 19-21); swap tiles 64x12 / 48x12 / 16x24 (kpc 2).
# Order: planar L1 layout first (L1's proven jobs minus the TPs), then sq, col, row64, row (control), swap last
# (new: the weight matrix as a 0.9-6.3 MB input image). Smallest shape first in each group.
# Stops at the first failure or STOP. Usage: systemd-run --user --unit=alta-llm-l1b --collect llm/llm-l1b-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
CHECK=$AI/npu-shader/teflon_check.py
LIB=$AI/mesa-upstream-fixes/build/src/gallium/targets/teflon/libteflon.so
M=$AI/npu-llm/models/layouts
RD=$K/runs/llm-l1b-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$LIB" | cut -c1-8)" = b6c2a28d ] || { echo "upstream lib is not b6c2a28d"; exit 1; }

run() { # NAME
   local n=$1 label="LLM1B-$1"
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM \
      scripts/npu-attempt "$label" -- $PY -I $CHECK -m $M/$n.tflite -r $M/${n}_reference.npz -e $LIB -n 20 \
      --save "$RD/$n.npy" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc $(grep -oE '"invoke_ms_median_after_first": [0-9.]+' "$RD/$label.stdout")"
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-l1b-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}
scripts/progress llm-l1b-ladder STARTING "llm/llm-l1b-ladder.sh: $RD"
for lay in rowP sq sqP col row64 row swapP swap; do
   for s in smol_gateup qwen_down qwen_gateup; do
      run ${s}_$lay
   done
done
scripts/progress llm-l1b-ladder DONE "all 24 rungs passed: $RD"
echo "ALL PASSED: $RD"

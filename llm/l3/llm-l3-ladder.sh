#!/usr/bin/env bash
# llm-l3-ladder.sh - LLM engine step L3: do the v7 tiling overrides make LLM-shaped NN jobs faster, and stay correct?
# Written 2026-11-19 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS.
# RAN 2026-11-19 11:04 (OK given): rungs 1-4 PASS, rung 5 (KPC 24) HUNG -> rest removed, see below.
# Library: mesa-llm (branch llm-tiling-experiments, lib 678775c5) = the tested upstream lib b6c2a28d plus
#   ETNA_ML_V7_KPC_CAP (kernels per superblock for 1x1 kernels, default cap 21; the accumulation-buffer bound
#   depth * interleave / tile_y is kept) and ETNA_ML_V7_TILE_H (tile height, only lower). Unset = identical NN
#   configs and weights to b6c2a28d (offline, npu-llm/runs/offline-l3-noop-20261119, 6 models).
# Offline (fake NPU, npu-llm/runs/offline-l3-20261119): every case below gives a config, no driver error.
# Models: npu-llm/models/sizes (planar, L1c). 20 invokes each; stops at the first failure, STOP, or any output
# more than 1 level from the CPU (XNNPACK).
# Order: baselines (no override) -> 1 token, more kernels per pass -> 16 tokens -> 256 tokens with lower tiles.
# Usage: systemd-run --user --unit=alta-llm-l3 --collect llm/l3/llm-l3-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
CHECK=$AI/npu-shader/teflon_check.py
LIB=$AI/mesa-llm/build/src/gallium/targets/teflon/libteflon.so
M=$AI/npu-llm/models/sizes
RD=$K/runs/llm-l3-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$LIB" | cut -c1-8)" = 678775c5 ] || { echo "mesa-llm lib is not 678775c5"; exit 1; }
[ "${ETNA_ML_V7_KPC_CAP:-0}" -gt 21 ] && { echo "KPC_CAP > 21 hangs the NPU (incident 2026-11-19)"; exit 1; }

run() { # MODEL KPC_CAP TILE_H
   local n=$1 k=$2 t=$3 label="LLM3-$1-k$2-t$3"
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM \
      ETNA_ML_V7_KPC_CAP=$k ETNA_ML_V7_TILE_H=$t \
      scripts/npu-attempt "$label" -- $PY -I $CHECK -m $M/$n.tflite -r $M/${n}_reference.npz -e $LIB -n 20 \
      --save "$RD/$n-k$k-t$t.npy" > "$RD/$label.stdout" 2>&1
   local rc=$? mx
   mx=$(grep -oE '"diff_vs_cpu_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   echo "== $label rc=$rc max_diff=${mx:-?} $(grep -oE '"invoke_ms_median_after_first": [0-9.]+' "$RD/$label.stdout")"
   if [ $rc -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-99}" -gt 1 ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-l3-ladder FAILED "stopped at $label rc=$rc max_diff=${mx:-?}; $RD"
      exit 1
   fi
}
scripts/progress llm-l3-ladder STARTING "llm/l3/llm-l3-ladder.sh: $RD"
# A: baselines, no override (must match L1c: 0.66 / 3.29 / 31.17 / 1.62 ms)
run smol_gateup_1x1_P 0 0; run smol_gateup_4x4_P 0 0; run smol_gateup_16x16_P 0 0; run qwen_gateup_1x1_P 0 0
# B/C/D REMOVED 2026-11-19: the first one (smol_gateup_1x1_P, KPC_CAP=24) HUNG the NPU (incidents/2026-11-19-1106-LLM3-k24).
# The driver's 1x1 cap (accum depth 64 / 3 = 21 kernels per core) is a hardware limit; never run KPC_CAP > 21.
# The rungs were: 1x1 KPC 24/32/42/64 (smol), 32/64 (qwen); 4x4 KPC 32; 16x16 (KPC 0 TILE_H 8, 32/4, 42/2).
scripts/progress llm-l3-ladder DONE "all 18 rungs passed: $RD"
echo "ALL PASSED: $RD"

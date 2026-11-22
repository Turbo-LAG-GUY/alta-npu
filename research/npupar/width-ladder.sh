#!/usr/bin/env bash
# width-ladder.sh - how the upscaler net's NPU time grows with width (8, 16, 32, 64 channels), 2026-11-21.
# Written CPU only. Jordan's OK 2026-11-21: "go ahead with the upscaler width test".
#
# Models: npu-upscale/width_models.py (random Gaussian weights, real layout: 3x3 convs 5 -> N -> N -> 4, uint8,
# planar 960x540 like the live hist2_n8_planar model; CPU references from the real game-frame input).
# Offline (fake NPU, kit library): all 3 convs on the NPU in one submit; synthetic n8's NN configs = the real n8's;
# kernels per core 1 / 2 / 4 / 4 (n64 takes two kernel rounds, image caching on), tiles 64x10 everywhere, all
# within proven settings (kpc cap 21). Default mode, no npu_parallel, no AXI SRAM (as the npupar ladder).
#   1 W-REAL8     live model hist2_n8_planar, 10 invokes (same-boot baseline)
#   2 W-SYN8      width_n8_planar, 10
#   3 W-SYN16     width_n16_planar, 10
#   4 W-SYN32-1   width_n32_planar, 1 invoke (first run of this size)
#   5 W-SYN32     10
#   6 W-SYN64-1   width_n64_planar, 1 invoke (first run of this size)
#   7 W-SYN64     10
#   8 W-REAL16    hist2_n16_planar, 10 (synthetic vs real at 16)
# Every rung: one scripts/npu-attempt job; each invoke checked against the first and the CPU answer; the ladder
# stops at the first failure or STOP.
# Usage: research/npupar/width-ladder.sh   (run detached: systemd-run --user --unit=alta-width-ladder --collect)
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
CS=$AI/npu-shader
PY=$AI/.venv/bin/python
LIB=$AI/mesa-axi-sram/build/src/gallium/targets/teflon/libteflon.so
U=$AI/npu-upscale/models
RD=$K/runs/width-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD/cache"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi

rung() { # LABEL MODEL_STEM N
   local label=$1 stem=$2 n=$3
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_ML_AXI_SRAM -u ETNA_CS_BARRIER_REG_BUDGET \
      ETNA_ML_DISK_CACHE_DIR="$RD/cache" \
      scripts/npu-attempt "$label" -- $PY -I $CS/teflon_check.py -m $U/${stem}_960x540.tflite \
      -r $U/${stem}_reference_960x540.npz -e $LIB -n "$n" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc"
   grep -E 'invoke_ms_median"|all_invokes|diff_vs_cpu_max|"match"' "$RD/$label.stdout" | head -4
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress width-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}

scripts/progress width-ladder STARTING "research/npupar/width-ladder.sh: $RD"
rung W-REAL8   hist2_n8_planar   10
rung W-SYN8    width_n8_planar   10
rung W-SYN16   width_n16_planar  10
rung W-SYN32-1 width_n32_planar  1
rung W-SYN32   width_n32_planar  10
rung W-SYN64-1 width_n64_planar  1
rung W-SYN64   width_n64_planar  10
rung W-REAL16  hist2_n16_planar  10
rm -rf "$RD/cache"
scripts/progress width-ladder DONE "all rungs passed: $RD"
echo "all rungs passed: $RD"

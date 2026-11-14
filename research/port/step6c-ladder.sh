#!/usr/bin/env bash
# step6c-ladder.sh - the step 6c NPU ladder for AFTER the reboot that clears incident 2026-11-13-2109-S6C-SM1001.
# Written 2026-11-13 (CPU only). Jordan gave a blanket OK for the step 6c steps ("I give you the OK for all steps");
# a new boot still deserves a look at PROGRESS.md and the HANDOFF before running this.
#
# Every rung is one scripts/npu-attempt job (STARTING/DONE in PROGRESS.md, kernel log capture, npu-run checks);
# the ladder stops at the first rung that fails or sets runs/STOP. Order: lowest risk first.
# Revised 2026-11-13 21:50 after incident 2026-11-13-2143-S6C-REGP6 (64 threads x 9 temps = 576 registers hung;
# 64 x 7 = 448 passed; the driver's default budget is now 448 and the softmax picks 32 threads = 352 registers):
#   1 S6C-SM1001-W32   softmax kernel, 32 threads (11 temps, 352 registers), 1 row of 1001 (MobileNet), 10 launches
#   2 S6C-SM4x1000-W32 4 rows of 1000
#   3 S6C-MN-OFF       MobileNet through the port tree's Teflon, ETNA_ML_SHADER unset (baseline, output saved)
#   4 S6C-MN-SM        ETNA_ML_SHADER=softmax: SQUEEZE + SOFTMAX on the NPU (driver picks 32 threads), output
#                      within 1 level of MN-OFF (exact count reported too)
#   5 S6C-MN-SM50      same, 50 invokes
# Optional --map (may hang the NPU, then npu-run refuses NPU jobs until a reboot): S6C-REGP5 = 64 x 8 = 512.
# Usage: research/port/step6c-ladder.sh [--map]     (run detached: systemd-run --user --unit=alta-s6c-ladder --collect)
set -u
K=/home/jordan/alta-npu
CS=/home/jordan/Documents/AI_Projects/npu-shader
PY=/home/jordan/Documents/AI_Projects/.venv/bin/python
LIB=/home/jordan/Documents/AI_Projects/mesa-compute/build/src/gallium/targets/teflon/libteflon.so
MN=$K/models/mobilenet_v1_mesa.tflite
MNR=$CS/models/mobilenet_v1_mesa_reference.npz
RD=$K/runs/cs6c-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi

rung() { # LABEL ENV... -- CMD...
   local label=$1; shift
   local envs=()
   while [ "$1" != "--" ]; do envs+=("$1"); shift; done
   shift
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET "${envs[@]}" \
      scripts/npu-attempt "$label" -- "$@" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc"
   grep -E '^softmax:|"result"|"match"|invoke_ms_median"|all_invokes|bytes_differing' "$RD/$label.stdout" | head -6
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress compute-port-step6c-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}

scripts/progress compute-port-step6c-ladder STARTING "research/port/step6c-ladder.sh (revised) $*: $RD"
rung S6C-SM1001-W32 X=1 -- $CS/cs_run --kernel softmax --len 1001 --rows 1 -w 32 --repeat 10 --out $RD/SM1001-W32
rung S6C-SM4x1000-W32 X=1 -- $CS/cs_run --kernel softmax --len 1000 --rows 4 -w 32 --repeat 10 --out $RD/SM4x1000-W32
rung S6C-MN-OFF X=1 -- $PY -I $CS/teflon_check.py -m $MN -r $MNR -e $LIB -n 10 --save $RD/mn_off.npy
rung S6C-MN-SM ETNA_ML_SHADER=softmax -- $PY -I $CS/teflon_check.py -m $MN -r $MNR -e $LIB -n 10 --compare $RD/mn_off.npy --tolerance 1
rung S6C-MN-SM50 ETNA_ML_SHADER=softmax -- $PY -I $CS/teflon_check.py -m $MN -r $MNR -e $LIB -n 50 --compare $RD/mn_off.npy --tolerance 1
if [ "${1:-}" = "--map" ]; then
   rung S6C-REGP5 ETNA_CS_BARRIER_REG_BUDGET=512 -- $CS/cs_run --kernel regp5 -n 256 -w 64 --out $RD/REGP5
fi
scripts/progress compute-port-step6c-ladder DONE "all rungs passed: $RD"
echo "all rungs passed: $RD"

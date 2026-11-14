#!/usr/bin/env bash
# step6d-ladder.sh - the step 6d NPU ladder: global average pool on the shader core (ETNA_ML_SHADER=avgpool).
# Written 2026-11-13 (CPU only). Needs Jordan's OK before it runs; check PROGRESS.md and the HANDOFF first.
#
# Every rung is one scripts/npu-attempt job (STARTING/DONE in PROGRESS.md, kernel log capture, npu-run checks);
# the ladder stops at the first rung that fails or sets runs/STOP. Order: lowest risk first. The pool kernel has
# no barriers or shared memory (only the proven load/store/integer path), so the register limit of barrier
# workgroups (512-576 vec4 registers, incidents 2026-11-13-2109 / -2143) doesn't apply.
#   1 S6D-AP7      the driver's pool kernel, 7x7x1024 (MobileNet's pool: 392 instructions, 5 temps, 17 vec4
#                  uniforms), 256 threads in workgroups of 64, 10 launches
#   2 S6D-AP1      1x1x256 (one value per channel: every word holds 4 channels; 19 instructions)
#   3 S6D-MN-OFF   MobileNet through the port tree's Teflon, ETNA_ML_SHADER unset (baseline, output saved)
#   4 S6D-MN-AP    ETNA_ML_SHADER=avgpool: the pool on the NPU, output byte-identical to MN-OFF
#   5 S6D-MN-APSM  ETNA_ML_SHADER=avgpool,softmax: all of MobileNet in one NPU submit, byte-identical to MN-OFF
#   6 S6D-MN-APSM50 same, 50 invokes
# Last, because they are the longest programs and largest uniform sets yet run on this core (softmax: 605
# instructions, 12 vec4 uniforms), so a hang there doesn't block the rungs above:
#   7 S6D-AP15     15x15x64 (225 values per channel: odd, words straddle two channels; 1668 instructions, 61 vec4)
#   8 S6D-AP16     16x16x32 (256 values per channel: the largest lane sums, division a plain shift; 1883 / 67 vec4)
# Usage: research/port/step6d-ladder.sh     (run detached: systemd-run --user --unit=alta-s6d-ladder --collect)
set -u
K=/home/jordan/alta-npu
CS=/home/jordan/Documents/AI_Projects/npu-shader
PY=/home/jordan/Documents/AI_Projects/.venv/bin/python
LIB=/home/jordan/Documents/AI_Projects/mesa-compute/build/src/gallium/targets/teflon/libteflon.so
MN=$K/models/mobilenet_v1_mesa.tflite
MNR=$CS/models/mobilenet_v1_mesa_reference.npz
RD=$K/runs/cs6d-ladder-$(date +%Y%m%d-%H%M%S)
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
   grep -E '^avgpool:|"result"|"match"|invoke_ms_median"|all_invokes|bytes_differing' "$RD/$label.stdout" | head -6
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress compute-port-step6d-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}

scripts/progress compute-port-step6d-ladder STARTING "research/port/step6d-ladder.sh $*: $RD"
rung S6D-AP7 X=1 -- $CS/cs_run --kernel avgpool --hwc 7 7 1024 -w 64 --repeat 10 --out $RD/AP7
rung S6D-AP1 X=1 -- $CS/cs_run --kernel avgpool --hwc 1 1 256 -w 64 --repeat 10 --out $RD/AP1
rung S6D-MN-OFF X=1 -- $PY -I $CS/teflon_check.py -m $MN -r $MNR -e $LIB -n 10 --save $RD/mn_off.npy
rung S6D-MN-AP ETNA_ML_SHADER=avgpool -- $PY -I $CS/teflon_check.py -m $MN -r $MNR -e $LIB -n 10 --compare $RD/mn_off.npy
rung S6D-MN-APSM ETNA_ML_SHADER=avgpool,softmax -- $PY -I $CS/teflon_check.py -m $MN -r $MNR -e $LIB -n 10 --compare $RD/mn_off.npy
rung S6D-MN-APSM50 ETNA_ML_SHADER=avgpool,softmax -- $PY -I $CS/teflon_check.py -m $MN -r $MNR -e $LIB -n 50 --compare $RD/mn_off.npy
rung S6D-AP15 X=1 -- $CS/cs_run --kernel avgpool --hwc 15 15 64 -w 16 --repeat 10 --out $RD/AP15
rung S6D-AP16 X=1 -- $CS/cs_run --kernel avgpool --hwc 16 16 32 -w 8 --repeat 10 --out $RD/AP16
scripts/progress compute-port-step6d-ladder DONE "all rungs passed: $RD"
echo "all rungs passed: $RD"

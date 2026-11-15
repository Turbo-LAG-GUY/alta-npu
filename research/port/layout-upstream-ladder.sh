#!/usr/bin/env bash
# layout-upstream-ladder.sh - the layout (patch 1) and V7 tiling (patch 2) fixes with the UPSTREAM build
# (mesa-upstream-fixes, main 8a9009e0482 + 7 patches, lib 549408ab). Written 2026-11-15 (CPU only).
# Needs Jordan's OK before it runs. Offline this build gives the port lib 2e21b93a's streams on all five models
# (runs/cs-offline/step6i-upstream). Same jobs as the passed 6f/6h rungs with the switch off; --compare checks the
# output byte for byte against what the port build produced on the NPU (6f run 2 / 6h).
#   1 UPL-C8x8     [1,8,8,64] 3D control, 10 invokes
#   2 UPL-C1x64    [1,1,64,384] (H = 1, patch 1), 10, = port output
#   3 UPL-C1x50    [1,1,50,96], 10
#   4 UPL-C64x1    [1,64,1,384] (W = 1, patch 2: hung before the fix), 1 invoke, = port output
#   5 UPL-C64x8    [1,64,8,64] (H > W), 1 invoke, = port output
#   6 UPL-C64x1-10 / 7 UPL-C64x8-10   the same, 10 invokes
# Usage: research/port/layout-upstream-ladder.sh   (run detached: systemd-run --user --unit=alta-upl --collect)
set -u
K=/home/jordan/alta-npu
CS=/home/jordan/Documents/AI_Projects/npu-shader
PY=/home/jordan/Documents/AI_Projects/.venv/bin/python
LIB=/home/jordan/Documents/AI_Projects/mesa-upstream-fixes/build/src/gallium/targets/teflon/libteflon.so
L=$CS/models/layout
P6F=$K/runs/cs6f-ladder-20261114-012542
P6H=$K/runs/cs6h-ladder-20261114-015507
RD=$K/runs/cs-layout-upstream-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$LIB" | cut -c1-8)" = 549408ab ] || { echo "upstream lib is not 549408ab"; exit 1; }

check() { # MODEL_PATH_WITHOUT_EXT N ARGS...
   local m=$1 n=$2; shift 2
   echo $PY -I $CS/teflon_check.py -m $m.tflite -r ${m}_reference.npz -e $LIB -n $n "$@"
}
run() { # DOC LABEL CMD...
   local docrung=$1 label=$2; shift 2
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM \
      scripts/npu-attempt "$label" -- "$@" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc"
   if { [ $rc -ne 0 ] && ! { [ "$docrung" = doc ] && [ $rc -eq 3 ]; }; } || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress upstream-layout-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
   [ $rc -eq 3 ] && scripts/progress "$label" NOTE "documentation rung rc=3 (3 = differs from the CPU)"
   return 0
}
scripts/progress upstream-layout-ladder STARTING "research/port/layout-upstream-ladder.sh: $RD"
run gate UPL-C8x8 $(check $L/conv_8x8x64_u8 10)
run gate UPL-C1x64 $(check $L/conv_1x64x384_u8 10 --compare $P6F/c1x64_off.npy)
run gate UPL-C1x50 $(check $L/conv_1x50x96_u8 10)
run gate UPL-C64x1 $(check $L/conv_64x1x384_u8 1 --compare $P6H/c64x1_off.npy)
run gate UPL-C64x8 $(check $L/conv_64x8x64_u8 1 --compare $P6H/c64x8_off.npy)
run gate UPL-C64x1-10 $(check $L/conv_64x1x384_u8 10 --compare $P6H/c64x1_off.npy)
run gate UPL-C64x8-10 $(check $L/conv_64x8x64_u8 10 --compare $P6H/c64x8_off.npy)
scripts/progress upstream-layout-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"

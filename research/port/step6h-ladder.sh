#!/usr/bin/env bash
# step6h-ladder.sh - the v7 NN tiling fix (step 6h, patch research/port/step6h-tiling.diff, port lib 599bed8f).
# Written 2026-11-14 (CPU only). Needs Jordan's OK before it runs; check PROGRESS.md and the HANDOFF first.
#
# The fix: etna_ml_calculate_tiling_v7 computed the NN tile from the unswapped dims while the config's x size is the
# other one, so [1,H,1,C] (and any H > W with W < 64) got a tile wider than the image; such NN jobs never finished
# (incidents 2026-11-14-0020-S6F-C64x1-OFF and -0127-S6F-W1-NN, devcoredump: FE past the NN kick, core busy; the
# 2026-11-12 column-strips hang had the same misfit). Now only there the tile is computed from the configured image.
# Offline (runs/cs-offline/step6h-tiling, tools/nn_config_compare.sh): NN configs + compressed weights of every proven
# model = kit library's (151 configs/weights over 16 models: MobileNet, conv3x1/1x1/1x4/planar, layout H = 1 and 3D,
# MEAN, all upscaler frame/strip models); only the W = 1 and H > W models change (tile 64x1 -> 1x24, 64x8 -> 8x24).
# Every rung is one invoke first: npu-run stops a job at its first hang line.
#   1 S6H-C8x8-OFF      conv_8x8x64_u8 control (config unchanged), 10 invokes
#   2 S6H-C64x1-OFF     conv_64x1x384_u8 [1,64,1,384], switch off: TP, NN (fixed tile), TP - the model of incident
#                       0020; 1 invoke, vs the CPU, output saved
#   3 S6H-W1-NN         same with ETNA_ML_SHADER=transpose (only the NN on the NN/TP units: the job of incident 0127)
#   4 S6H-C64x8-OFF     conv_64x8x64_u8 [1,64,8,64] (3D, H > W): 1 invoke
#   5 S6H-C64x1-OFF10   rung 2 with 10 invokes, all identical, == rung 2's output
#   6 S6H-C64x8-OFF10   rung 4 with 10 invokes
# Usage: research/port/step6h-ladder.sh   (run detached: systemd-run --user --unit=alta-s6h --collect)
set -u
K=/home/jordan/alta-npu
CS=/home/jordan/Documents/AI_Projects/npu-shader
PY=/home/jordan/Documents/AI_Projects/.venv/bin/python
LIB=/home/jordan/Documents/AI_Projects/mesa-compute/build/src/gallium/targets/teflon/libteflon.so
L=$CS/models/layout
RD=$K/runs/cs6h-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi

check() { # MODEL_PATH_WITHOUT_EXT N ARGS...
   local m=$1 n=$2; shift 2
   echo $PY -I $CS/teflon_check.py -m $m.tflite -r ${m}_reference.npz -e $LIB -n $n "$@"
}
rung() { # LABEL ENV... -- CMD...
   local label=$1; shift
   local envs=()
   while [ "$1" != "--" ]; do envs+=("$1"); shift; done
   shift
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM "${envs[@]}" \
      scripts/npu-attempt "$label" -- "$@" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc"
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress compute-port-step6h-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}

scripts/progress compute-port-step6h-ladder STARTING "research/port/step6h-ladder.sh: $RD"
rung S6H-C8x8-OFF X=1 -- $(check $L/conv_8x8x64_u8 10)
rung S6H-C64x1-OFF X=1 -- $(check $L/conv_64x1x384_u8 1 --save $RD/c64x1_off.npy)
rung S6H-W1-NN ETNA_ML_SHADER=transpose -- $(check $L/conv_64x1x384_u8 1 --compare $RD/c64x1_off.npy)
rung S6H-C64x8-OFF X=1 -- $(check $L/conv_64x8x64_u8 1 --save $RD/c64x8_off.npy)
rung S6H-C64x1-OFF10 X=1 -- $(check $L/conv_64x1x384_u8 10 --compare $RD/c64x1_off.npy)
rung S6H-C64x8-OFF10 X=1 -- $(check $L/conv_64x8x64_u8 10 --compare $RD/c64x8_off.npy)
scripts/progress compute-port-step6h-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"

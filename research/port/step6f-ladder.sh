#!/usr/bin/env bash
# step6f-ladder.sh - the partition-output detranspose fix (step 6f, patch research/port/step6f-detranspose.diff).
# Written 2026-11-14 (CPU only). Needs Jordan's OK before it runs; check PROGRESS.md and the HANDOFF first.
#
# The fix: etnaviv_ml.c adds the detranspose behind a partition output when C > 1 && H*W > 1 (was H > 1 && W > 1 &&
# C > 1, so [1,1,T,C] / [1,H,1,C] NN outputs reached the CPU in channel planes, kit library too), and for H = 1 the TP
# config takes the [1,T,1,C] form (x = T, y = C: no x size of 1, no 16-bit overflow of T*C). Offline (shim,
# runs/cs-offline/step6f-detranspose): MobileNet, conv3x1 and a 3D conv = kit library streams byte for byte; the
# [1,1,T,C] / [1,T,1,C] models gain one TP detranspose (H = 1 and W = 1 configs identical word for word); the 6e MEAN
# ON streams (proven on the NPU) unchanged. New on hardware: a TP detranspose with out_loop_0_count 1 (the pre-2025
# MobileNet path had that too) and the proven t4x4b shader transpose in 16- and 8-thread workgroups.
# Models: npu-shader/models/layout/ (layout_models.py: one 1x1 CONV with identity weights and a different output
# scale; a channel-plane misread would differ from the CPU by mean ~100 levels, NN rounding by <= 1).
#   1 S6F-C8x8-OFF    conv_8x8x64_u8   3D control, stream = kit library (detransposed before the fix too)
#   2 S6F-C1x64-OFF   conv_1x64x384_u8 [1,1,64,384]: TP transpose, NN, TP detranspose (new), vs the CPU; saved
#   (S6F-C64x1-OFF conv_64x1x384_u8 [1,64,1,384] HUNG the NPU in run 1, incident 2026-11-14-0020-S6F-C64x1-OFF:
#    W = 1 is etnaviv's existing TP transpose / NN path (kit library too), not the detranspose; W = 1 rungs dropped)
#   4 S6F-C1x50-OFF   conv_1x50x96_u8  [1,1,50,96] (T % 4 != 0: TP even with the transpose switch)
#   5 S6F-T384-OFF    mean_ch384_u8    the 6e model: conv on the NN, MEAN on the CPU; before the fix max 22 / mean
#                                      6.8 levels off, now it must match the CPU (default limit 2.0)
#   6 S6F-C1x64-TR    rung 2 with ETNA_ML_SHADER=transpose (shader transpose + detranspose), == rung 2 byte for byte
#   8 S6F-C1x32I8-OFF conv_1x32x768_i8 documentation rung (rc 3 doesn't stop): 6e finding 2, the int8 conv that
#                                      saturated (all 127), now readable directly against the CPU
# Then the W = 1 localization (Jordan 2026-11-14: "Localize first"), last because either may hang; one invoke each,
# and npu-run now stops a job at its first hang line:
#   9 S6F-W1-TP       layout/mean_64x1x384_in_u8 with ETNA_ML_SHADER=mean: the hung model's TP transpose (config
#                     identical word for word incl. addresses) + the proven shader channel MEAN, no NN; exact vs CPU
#  10 S6F-W1-NN       conv_64x1x384_u8 with ETNA_ML_SHADER=transpose: shader grids replace both TP jobs, so the
#                     hung model's NN config (identical but for the weights address) is the only new job; rung 6
#                     proves the 16-thread shader transposes first. Hang at 9 = the TP transpose; pass at 9 and hang
#                     at 10 = the NN; both pass = only the combination hangs.
# Run 1 2026-11-14 00:19 (runs/cs6f-ladder-20261114-001951): C8x8-OFF PASS, C1x64-OFF PASS (the fix works: max 1 /
# mean 0.20 vs the CPU), then the W = 1 hang above. Run 2 (after the reboot) = this list: the W = 1 Teflon rungs dropped, the two localization probes added at the end.
# Usage: research/port/step6f-ladder.sh   (run detached: systemd-run --user --unit=alta-s6f --collect)
set -u
K=/home/jordan/alta-npu
CS=/home/jordan/Documents/AI_Projects/npu-shader
PY=/home/jordan/Documents/AI_Projects/.venv/bin/python
LIB=/home/jordan/Documents/AI_Projects/mesa-compute/build/src/gallium/targets/teflon/libteflon.so
L=$CS/models/layout
M=$CS/models/mean
RD=$K/runs/cs6f-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi

check() { # MODEL_PATH_WITHOUT_EXT N ARGS... -> the teflon_check command
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
   grep -E '"match"|invoke_ms_median"|all_invokes|diff_vs_cpu|bytes_differing' "$RD/$label.stdout" | head -6
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress compute-port-step6f-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}
# documentation rung: a comparison failure (rc 3) is recorded and the ladder ends normally; any other nonzero rc or a
# STOP is a failure
doc() { # LABEL CMD...
   local label=$1; shift
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM \
      scripts/npu-attempt "$label" -- "$@" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc (documentation rung: rc 3 does not count as a failure)"
   grep -E 'invoke_ms_median"|diff_vs_cpu|"match"' "$RD/$label.stdout" | head -4
   if { [ $rc -ne 0 ] && [ $rc -ne 3 ]; } || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress compute-port-step6f-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
   scripts/progress "$label" NOTE "documentation rung rc=$rc (3 = differs from the CPU)"
}

scripts/progress compute-port-step6f-ladder STARTING "research/port/step6f-ladder.sh: $RD"
rung S6F-C8x8-OFF X=1 -- $(check $L/conv_8x8x64_u8 10)
rung S6F-C1x64-OFF X=1 -- $(check $L/conv_1x64x384_u8 10 --save $RD/c1x64_off.npy)
rung S6F-C1x50-OFF X=1 -- $(check $L/conv_1x50x96_u8 10)
rung S6F-T384-OFF X=1 -- $(check $M/mean_ch384_u8 10)
rung S6F-C1x64-TR ETNA_ML_SHADER=transpose -- $(check $L/conv_1x64x384_u8 10 --compare $RD/c1x64_off.npy)
doc S6F-C1x32I8-OFF $(check $L/conv_1x32x768_i8 10 --save $RD/c1x32i8_off.npy)
rung S6F-W1-TP ETNA_ML_SHADER=mean -- $(check $L/mean_64x1x384_in_u8 1 --cpu-mean-limit 0)
rung S6F-W1-NN ETNA_ML_SHADER=transpose -- $(check $L/conv_64x1x384_u8 1)
scripts/progress compute-port-step6f-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"

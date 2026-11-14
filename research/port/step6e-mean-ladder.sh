#!/usr/bin/env bash
# step6e-mean-ladder.sh - MEAN on the NPU's shader core (ETNA_ML_SHADER=mean), step 6e.
# Written 2026-11-13 (CPU only). Needs Jordan's OK before it runs; check PROGRESS.md and the HANDOFF first.
#
# Kernels (driver builders in mesa-compute etnaviv_ml_shader.c, no barriers, no shared memory):
#   channel MEAN: a loop over the channels (the loop form proven by S6E-LOOP*), 89 instructions, 6 temps;
#   spatial MEAN: the avgpool kernel (S6D-*) with the requantization table, 448 instructions at 7x7x1024.
#   Requantization = XNNPACK's (the CPU path beside Teflon), exact by construction: estimate + a 257-entry threshold
#   table as a third buffer. New on hardware: signed shift and signed selects in the estimate's clamp (a wrong result,
#   not a hang, is the expected failure), a third buffer bound to a grid.
#   1 S6E-MCH384     cs_run mean channels, 1x64 positions x 384 channels, uint8, 16 threads, 10 launches
#   2 S6E-MCH768I8   1x32 x 768, int8, 8 threads
#   3 S6E-MSP7       mean spatial 7x7x1024, uint8, 256 threads in workgroups of 64
#   4 S6E-MCH128x768 1x128 x 768, uint8, 32 threads (timing at a transformer size)
#   then per Teflon model (npu-shader/models/mean/, 1x1 CONV on the NN in front unless noted): OFF (switch unset,
#   MEAN on the CPU = XNNPACK, output saved), then ON (ETNA_ML_SHADER=mean), byte-identical to OFF:
#   5-6   S6E-T384-OFF / -ON     mean_ch384_u8   (64 tokens x 384 channels)
#   7-8   S6E-T768-OFF / -ON     mean_ch768_i8   (32 x 768, int8)
#   9-10  S6E-TSP7-OFF / -ON     mean_sp7x7_u8   (Keras-style global pooling, 7x7x1024)
#   11-12 S6E-TIN64-OFF / -ON    mean_ch64_in_u8 (MEAN on the graph input: TP transpose in front; OFF has no NPU part)
#   13    S6E-T384-ON50          mean_ch384_u8 ON, 50 invokes
# Run 1 (2026-11-13 23:38, runs/cs6e-mean-20261113-233843): rungs 1-4 PASS exact; stopped at S6E-T384-OFF because the
# synthetic 1x1 CONV in front of MEAN differs on the NN from the CPU (MEAN after it off by max 22 / mean 6.8 levels
# vs teflon_check's 2.0 limit), not an NPU or shader problem. So the conv models' OFF/ON rungs now pass
# --cpu-mean-limit 32 (the deviation is still reported) and are judged by ON == OFF byte for byte; the graph-input
# model (no NN part) keeps the default 2.0 and is checked against the CPU directly.
# Run 2 (23:42, runs/cs6e-mean-20261113-234220): T384-OFF PASS; T384-ON output EXACT vs the CPU reference but != OFF
# because OFF is wrong: etnaviv adds the partition-output detranspose only if H>1 && W>1 && C>1 (is_3d), so the
# [1,1,64,384] conv output reaches the CPU MEAN still in channel planes (confirmed byte-exact on the CPU; the bug is in
# the kit library too). So now: every ON rung must equal the CPU reference exactly (--cpu-mean-limit 0); OFF rungs of
# the [1,1,T,C] models only document the bug (limit 32, no comparison with ON); the 7x7x1024 and graph-input models,
# where OFF is right, also need ON == OFF byte for byte.
# Usage: research/port/step6e-mean-ladder.sh [--teflon-only]   (--teflon-only skips rungs 1-4)
#        (run detached: systemd-run --user --unit=alta-s6e-mean --collect)
set -u
K=/home/jordan/alta-npu
CS=/home/jordan/Documents/AI_Projects/npu-shader
PY=/home/jordan/Documents/AI_Projects/.venv/bin/python
LIB=/home/jordan/Documents/AI_Projects/mesa-compute/build/src/gallium/targets/teflon/libteflon.so
M=$CS/models/mean
RD=$K/runs/cs6e-mean-$(date +%Y%m%d-%H%M%S)
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
   grep -E '^mean |"result"|"match"|invoke_ms_median"|all_invokes|bytes_differing' "$RD/$label.stdout" | head -6
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress compute-port-step6e-mean-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}
# doc rung: runs like rung(), but a comparison failure (rc 3) is recorded and the ladder goes on; any other nonzero
# rc or a STOP still ends it (run 3, 23:45: T384-ON PASS exact; the T768-OFF documentation rung blocked the rest)
doc() { # LABEL MODEL N ARGS...
   local label=$1 model=$2 n=$3; shift 3
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET \
      scripts/npu-attempt "$label" -- $PY -I $CS/teflon_check.py -m $M/$model.tflite -r $M/${model}_reference.npz \
      -e $LIB -n "$n" "$@" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc (documentation rung: rc 3 does not stop the ladder)"
   grep -E 'invoke_ms_median"|diff_vs_cpu|"match"' "$RD/$label.stdout" | head -4
   if { [ $rc -ne 0 ] && [ $rc -ne 3 ]; } || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress compute-port-step6e-mean-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
   scripts/progress "$label" NOTE "documentation rung rc=$rc (3 = differs from the CPU, expected for [1,1,T,C] OFF)"
}
TEFLON_ONLY=0
[ "${1:-}" = "--teflon-only" ] && TEFLON_ONLY=1
NNLIM="--cpu-mean-limit 32" # models with the synthetic NN conv in front (see above)
tcheck() { # LABEL MODEL SWITCH(or -) N [--save F | --compare F] [--cpu-mean-limit X]
   local label=$1 model=$2 sw=$3 n=$4; shift 4
   if [ "$sw" = "-" ]; then
      rung "$label" X=1 -- $PY -I $CS/teflon_check.py -m $M/$model.tflite -r $M/${model}_reference.npz -e $LIB -n "$n" "$@"
   else
      rung "$label" ETNA_ML_SHADER="$sw" -- $PY -I $CS/teflon_check.py -m $M/$model.tflite -r $M/${model}_reference.npz -e $LIB -n "$n" "$@"
   fi
}

scripts/progress compute-port-step6e-mean-ladder STARTING "research/port/step6e-mean-ladder.sh $*: $RD"
if [ $TEFLON_ONLY = 0 ]; then
rung S6E-MCH384 X=1 -- $CS/cs_run --kernel mean --axes ch --hwc 1 64 384 -w 16 --q 0.05 120 0.03 128 --repeat 10 --out $RD/MCH384
rung S6E-MCH768I8 X=1 -- $CS/cs_run --kernel mean --axes ch --hwc 1 32 768 -w 8 --q 0.07 5 0.02 -3 --int8 --repeat 10 --out $RD/MCH768I8
rung S6E-MSP7 X=1 -- $CS/cs_run --kernel mean --axes sp --hwc 7 7 1024 -w 64 --q 0.0235 0 0.02 3 --repeat 10 --out $RD/MSP7
rung S6E-MCH128x768 X=1 -- $CS/cs_run --kernel mean --axes ch --hwc 1 128 768 -w 32 --q 0.05 120 0.03 128 --repeat 10 --out $RD/MCH128x768
fi
EXACT="--cpu-mean-limit 0"  # every byte equal to the CPU reference (XNNPACK)
tcheck S6E-T384-ON mean_ch384_u8 mean 10 $EXACT
doc S6E-T768-OFF mean_ch768_i8 10 --save $RD/t768_off.npy   # documents the [1,1,T,C] OFF path (int8)
# run 4 (23:47): int8 OFF and ON both all 127: the NN's int8 conv saturates on this synthetic model, so ON can only
# be checked against OFF (same NN output in, same MEAN out); int8 inside Teflon is tested by mean_ch64_in_i8 below
tcheck S6E-T768-ON mean_ch768_i8 mean 10 --compare $RD/t768_off.npy --cpu-mean-limit 255
tcheck S6E-TSP7-OFF mean_sp7x7_u8 - 10 --save $RD/tsp7_off.npy
tcheck S6E-TSP7-ON mean_sp7x7_u8 mean 10 --compare $RD/tsp7_off.npy $EXACT
tcheck S6E-TIN64-OFF mean_ch64_in_u8 - 10 --save $RD/tin64_off.npy
tcheck S6E-TIN64-ON mean_ch64_in_u8 mean 10 --compare $RD/tin64_off.npy $EXACT
tcheck S6E-TIN64I8-OFF mean_ch64_in_i8 - 10 --save $RD/tin64i8_off.npy
tcheck S6E-TIN64I8-ON mean_ch64_in_i8 mean 10 --compare $RD/tin64i8_off.npy $EXACT
tcheck S6E-T384-ON50 mean_ch384_u8 mean 50 $EXACT
scripts/progress compute-port-step6e-mean-ladder DONE "all rungs passed: $RD"
echo "all rungs passed: $RD"

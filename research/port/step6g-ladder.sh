#!/usr/bin/env bash
# step6g-ladder.sh - 8-bit elementwise ops on the shader core through lookup tables (ETNA_ML_SHADER=eltwise: MUL,
# SQUARED_DIFFERENCE, RSQRT, ADD/SUB with a constant) and a whole int8 layer normalization in one NPU submit.
# Patch research/port/step6g-eltwise.diff (base-step6f). Written 2026-11-14 (CPU only). Needs Jordan's OK.
#
# Kernel rungs: npu-shader/cs_run --kernel eltwise = the driver's lookup kernel with a synthetic table (a pseudo-random
# byte function), random operands, a 4 KiB 0xdeadbeef guard after the output: checks the addressing of every operand
# mode (full = channel planes, pos = one value per position, chan = per channel, scalar), the byte packing and the 2D
# grid; no barriers, no shared memory, ~31-48 instructions, 4-6 temps. The tables themselves were checked offline
# against the TFLite interpreter entry by entry (tools/eltwise_table_check.py: all EXACT for both models).
#   1 S6G-K-FF     full,full   1x64 positions x 384 channels (65536-word table, the MUL(x, rg) case)
#   2 S6G-K-FP     full,pos    (SQUARED_DIFFERENCE(x, mean))
#   3 S6G-K-PC     pos,chan    (MUL(rsqrt, gamma))
#   4 S6G-K-CF7    chan,full   7 channels (the per-channel byte select with C % 4 != 0; SUB(beta, ms))
#   5 S6G-K-U1     full        1 channel, 256-word table (RSQRT, ADD(var, eps) folded)
#   6 S6G-K-FS     full,scalar
# Teflon rungs (npu-shader/models/layernorm/, layernorm_model.py: Keras LayerNormalization decomposition, int8):
#   OFF = switch unset: everything on the CPU except the final ADD (NN); ON = ETNA_ML_SHADER=mean,eltwise: all ten
#   ops in one submit (3 TP transposes, 2 shader MEANs, 7 table ops, the NN ADD, the TP detranspose of step 6f).
#   The NN ADD gets the same inputs either way, so ON must equal OFF byte for byte.
#   7 S6G-LN64-OFF   ln_64x384_i8 [1,64,384]   output saved (vs the CPU: the NN ADD may differ by a level)
#   8 S6G-LN64-ON    == OFF byte for byte
#   9 S6G-LN32-OFF   ln_32x768_i8 [1,32,768]
#  10 S6G-LN32-ON    == OFF byte for byte
#  11 S6G-LN64-ON50  50 invokes, all == OFF
# Run 1 2026-11-14 01:58 (runs/cs6g-ladder-20261114-015844): kernel rungs 1-6 PASS (0 wrong, 0 stray); stopped at
# LN64-OFF: the final int8 two-tensor ADD on the NN outputs all -128 (the int8 NN path is broken here; the int8 conv
# gave all 127). So ETNA_ML_SHADER=eltwise now also takes int8 two-tensor ADD/SUB (XNNPACK vadd, exact; uint8 stays
# on the NN): ON = no NN job at all (3 TP transposes, 10 shader jobs, the TP detranspose), tables all EXACT offline.
# Run 2 (--teflon-only): OFF rungs document the broken NN ADD (rc 3 doesn't stop); ON must equal the CPU exactly.
# Usage: research/port/step6g-ladder.sh [--teflon-only]   (run detached: systemd-run --user --unit=alta-s6g --collect)
set -u
K=/home/jordan/alta-npu
CS=/home/jordan/Documents/AI_Projects/npu-shader
PY=/home/jordan/Documents/AI_Projects/.venv/bin/python
LIB=/home/jordan/Documents/AI_Projects/mesa-compute/build/src/gallium/targets/teflon/libteflon.so
LN=$CS/models/layernorm
RD=$K/runs/cs6g-ladder-$(date +%Y%m%d-%H%M%S)
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
      scripts/progress compute-port-step6g-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}

doc() { # LABEL CMD...: a comparison failure (rc 3) is recorded, the ladder goes on; other errors or a STOP end it
   local label=$1; shift
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM \
      scripts/npu-attempt "$label" -- "$@" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc (documentation rung)"
   if { [ $rc -ne 0 ] && [ $rc -ne 3 ]; } || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress compute-port-step6g-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
   scripts/progress "$label" NOTE "documentation rung rc=$rc (3 = differs from the CPU)"
}

scripts/progress compute-port-step6g-ladder STARTING "research/port/step6g-ladder.sh $*: $RD"
if [ "${1:-}" != "--teflon-only" ]; then
   rung S6G-K-FF X=1 -- $CS/cs_run --kernel eltwise --hwc 1 64 384 --modes full,full --repeat 10 --out $RD/K-FF
   rung S6G-K-FP X=1 -- $CS/cs_run --kernel eltwise --hwc 1 64 384 --modes full,pos --repeat 10 --out $RD/K-FP
   rung S6G-K-PC X=1 -- $CS/cs_run --kernel eltwise --hwc 1 64 384 --modes pos,chan --repeat 10 --out $RD/K-PC
   rung S6G-K-CF7 X=1 -- $CS/cs_run --kernel eltwise --hwc 1 64 7 --modes chan,full --repeat 10 --out $RD/K-CF7
   rung S6G-K-U1 X=1 -- $CS/cs_run --kernel eltwise --hwc 1 64 1 --modes full --repeat 10 --out $RD/K-U1
   rung S6G-K-FS X=1 -- $CS/cs_run --kernel eltwise --hwc 1 64 384 --modes full,scalar --repeat 10 --out $RD/K-FS
fi
# OFF: everything on the CPU except the int8 NN ADD (broken: documentation); ON: no NN job, must equal the CPU
doc S6G-LN64-OFF $(check $LN/ln_64x384_i8 10)
rung S6G-LN64-ON ETNA_ML_SHADER=mean,eltwise -- $(check $LN/ln_64x384_i8 10 --cpu-mean-limit 0)
doc S6G-LN32-OFF $(check $LN/ln_32x768_i8 10)
rung S6G-LN32-ON ETNA_ML_SHADER=mean,eltwise -- $(check $LN/ln_32x768_i8 10 --cpu-mean-limit 0)
rung S6G-LN64-ON50 ETNA_ML_SHADER=mean,eltwise -- $(check $LN/ln_64x384_i8 50 --cpu-mean-limit 0)
scripts/progress compute-port-step6g-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"

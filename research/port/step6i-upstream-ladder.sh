#!/usr/bin/env bash
# UPSTREAM VARIANT (2026-11-15): same 14 rungs with the upstream build (mesa-upstream-fixes, main 8a9009e0482 + 7 patches,
# lib 549408ab; branch tip f3ac5799d2d has the same tree). Offline it gives the port lib 2e21b93a's streams on all 14 models.
# step6i-ladder.sh - etnaviv's int8 NN path (step 6i, patch research/port/step6i-int8nn.diff, port lib 2e21b93a).
# Written 2026-11-14 (CPU only). Needs Jordan's OK before it runs; check PROGRESS.md and the HANDOFF first.
#
# Three fixes, all in the kit library and upstream main too:
#  1 etnaviv_ml_nn_v7.c: int8 weights went into the v7 weight stream (and the bias correction) as int8 bytes next to
#    the +128-shifted zero point, so a weight of 0 counted as -128 (the "int8 1x1 conv -> all 127" of 6e/6f).
#    Now shifted by +128 first (as the v8 encoder does).
#  2 etnaviv_ml_nn.c etna_ml_lower_add_v7: compute_bias_add() got the raw int8 zero points (wrapped in its uint8_t
#    parameters, 128 off in the offset term): now etna_tensor_zero_point() as for every other NN operand.
#  3 etnaviv_ml_nn_v7.c calculate_bias_correction: for an ADD it summed weight_width*height*input_channels bytes of an
#    8-byte weight tensor (out-of-bounds heap read; the ADD bias changed from run to run, even uint8). Now 1 channel.
#  4 etnaviv_ml.c lower_operations: an NN ADD of a graph input (NHWC) and a conv output (NCHW) added the bytes across
#    the two layouts and its output kept the NHWC label (no detranspose): run 1 (runs/cs6i-ladder-20261114-022710,
#    UP6I-U8-ADD rc 3, max 154) fits exactly that (out = 0.069 x_HWC + 1.039 c_CHW - 12.1 vs the CPU's 0.070/1.039/-12.4,
#    so the ADD arithmetic + fixed bias are right). Now, when any operand is planar, all are transposed to planes and
#    the output is detransposed (TP x->6, NN ADD(6,3)->7, TP 7->out). ADD-free models and the 6g layernorm streams
#    (ON: byte-identical to 97cacd63; OFF: = ef649abe) unchanged.
# Run 1 with ef649abe passed rungs 1-10: every int8 conv max 1 vs the CPU, NPU u8-128 == i8 byte for byte.
# Offline (runs/cs-offline/step6i-int8nn, tools/twin_compare.py): every int8 model below gives exactly the NN/TP
# configs, compressed weights and command streams of its uint8 twin (bytes and zero points +128,
# npu-shader/models/int8nn_models.py); the 18 proven models (MobileNet, upscaler frames/strips, layout, MEAN) are
# unchanged word for word; the ADD bias word = the formula (-A*z1 - w0*z0).
# Each U8 rung is the control for the I8 rung after it (same hardware job; only the CPU-side +128 upload differs).
# New shapes run 1 invoke first (npu-run stops a job at its first hang line); U8-ADD is the first v7 NN ADD here.
#    1 UP6I-U8-PW64   / 2 UP6I-I8-PW64    1x1 conv [1,8,8,64], random weights + bias, 10 invokes each
#    3 UP6I-U8-PW64R  / 4 UP6I-I8-PW64R   same + fused RELU
#    5 UP6I-U8-C3     / 6 UP6I-I8-C3      3x3 SAME [1,16,16,16] -> 32 (1 invoke, then 10)
#    7 UP6I-U8-C3S2   / 8 UP6I-I8-C3S2    3x3 SAME stride 2
#    9 UP6I-U8-DW3    / 10 UP6I-I8-DW3    depthwise 3x3 [1,16,16,32]
#   11 UP6I-U8-ADD    / 12 UP6I-I8-ADD    x -> 1x1 conv -> y, ADD(x, y) (scale ratio 14.8), 1 then 10
#   13 UP6I-I8-ID768                      layout/conv_1x32x768_i8, the model that gave all 127 in 6e/6f
#   14 UP6I-LN64-OFF (documentation)      layernorm/ln_64x384_i8 switch off: the int8 NN ADD [1,1,64,384] on the NN
# Pass (teflon_check): all invokes identical, mean diff vs the CPU <= 2; afterwards compare each U8/I8 pair's saved
# outputs (U8 - 128 must equal I8 byte for byte) and report max/mean vs the CPU.
# Usage: research/port/step6i-ladder.sh   (run detached: systemd-run --user --unit=alta-s6i --collect)
set -u
K=/home/jordan/alta-npu
CS=/home/jordan/Documents/AI_Projects/npu-shader
PY=/home/jordan/Documents/AI_Projects/.venv/bin/python
LIB=/home/jordan/Documents/AI_Projects/mesa-upstream-fixes/build/src/gallium/targets/teflon/libteflon.so
M=$CS/models/int8nn
RD=$K/runs/cs6i-upstream-ladder-$(date +%Y%m%d-%H%M%S)
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
      scripts/progress upstream-step6i-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
   [ $rc -eq 3 ] && scripts/progress "$label" NOTE "documentation rung rc=3 (3 = differs from the CPU)"
   return 0
}
pair() { # NAME TAG N_FIRST
   run gate UP6I-U8-$2 $(check $M/${1}_u8 $3 --save $RD/${1}_u8.npy)
   run gate UP6I-I8-$2 $(check $M/${1}_i8 10 --save $RD/${1}_i8.npy)
}

scripts/progress upstream-step6i-ladder STARTING "research/port/step6i-upstream-ladder.sh: $RD"
pair pw64 PW64 10
pair pw64r PW64R 10
pair c3 C3 1
pair c3s2 C3S2 1
pair dw3 DW3 1
pair add ADD 1
run gate UP6I-I8-ID768 $(check $CS/models/layout/conv_1x32x768_i8 10 --save $RD/id768_i8.npy)
run doc UP6I-LN64-OFF $(check $CS/models/layernorm/ln_64x384_i8 10 --save $RD/ln64_off.npy)
scripts/progress upstream-step6i-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"

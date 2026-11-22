#!/usr/bin/env bash
# jb_offline.sh (2026-11-19, LLM engine L3b, CPU only): jb_run against the FAKE NPU (npu-shader/shim), inside bwrap with
# an empty /dev, so nothing can reach /dev/dri; the shim executes nothing. Every submit lands in OUT/shim_submit_*.
# Usage: [JB_ENV='ETNA_ML_AXI_SRAM=all ...'] jb_offline.sh OUT PACK.jbp [jb_run options]   (JB_DUMPS=1: also the driver's NN config / weight dumps;
#   JB_NOSTRACE=1: no strace, for timing)
set -u
OUT=$(realpath -m "$1") PACK=$(realpath "$2"); shift 2
AI=/home/jordan/Documents/AI_Projects
SHIM=$AI/npu-shader/shim/libnpu_etna_shim.so
JB=${JB_BIN:-$AI/npu-llm/bin/jb_run}   # JB_BIN: another jb_run build
rm -rf "$OUT"; mkdir -p "$OUT"
dbg=ml_msgs; [ "${JB_DUMPS:-0}" = 1 ] && dbg=ml_msgs,dump_shaders
cd "$OUT" || exit 2
env -u ETNA_ML_SHADER -u ETNA_MESA_DEBUG -u ETNA_ML_AXI_SRAM -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_V7_KPC_CAP \
   -u ETNA_ML_V7_TILE_H NPU_SHIM_DUMP_DIR="$OUT" ETNA_MESA_DEBUG=$dbg ${JB_ENV:-} \
   bwrap --dev-bind / / --dev /dev \
   $( [ "${JB_NOSTRACE:-0}" = 1 ] && echo env LD_PRELOAD="$SHIM" || echo strace -f -o "$OUT/strace.txt" -E LD_PRELOAD="$SHIM" -e trace=openat,open,ioctl ) \
   "$JB" --offline "$@" "$PACK" > "$OUT/stdout.txt" 2>&1
rc=$?
grep -q "npu shim: fake GC8000" "$OUT/stdout.txt" || { echo "jb_offline: the shim did not load ($OUT)"; exit 2; }
# JB_NOSTRACE=1 (timing runs): no strace (it stops on every syscall); the empty /dev of bwrap still hides every DRM node
if [ -e "$OUT/strace.txt" ] && grep -E 'openat\(.*"/dev/dri/(card|renderD129)' "$OUT/strace.txt" | grep -v ENOENT | grep -q .; then
   echo "jb_offline: a real DRM node was opened?! ($OUT/strace.txt)"; exit 2
fi
echo "jb_offline: rc=$rc submits=$(ls "$OUT"/shim_submit_*.stream 2>/dev/null | wc -l) $(grep -h '^{' "$OUT/stdout.txt") $OUT"

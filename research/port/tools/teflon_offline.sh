#!/usr/bin/env bash
# teflon_offline.sh - run npu-shader/teflon_check.py through a Teflon library against the FAKE NPU (the drm shim,
# npu-shader/shim/libnpu_etna_shim.so). CPU only, needs no OK: inside bwrap with an empty /dev nothing can reach
# /dev/dri, and the shim executes nothing. Every submit lands in OUT/shim_submit_NNNN.{stream,bos}; the outputs are
# meaningless (teflon_check's rc 3 is expected), the streams are the point.
# Written 2026-11-14 (the commands of the 6b-6e offline checks, which weren't saved as a script).
# Usage: teflon_offline.sh OUT LIB MODEL REF [ETNA_ML_SHADER value]
#   TEFLON_OFFLINE_DEBUG=ml_msgs,dump_shaders also writes every TP/NN config as OUT/mesa-*.bin (runs inside OUT)
set -u
OUT=$(realpath -m "$1") LIB=$(realpath "$2") MODEL=$(realpath "$3") REF=$(realpath "$4") SHADER=${5:-}
AI=/home/jordan/Documents/AI_Projects
SHIM=$AI/npu-shader/shim/libnpu_etna_shim.so
PY=$AI/.venv/bin/python
rm -rf "$OUT"; mkdir -p "$OUT/cache"
envs=(NPU_SHIM_DUMP_DIR="$OUT" ETNA_ML_DISK_CACHE_DIR="$OUT/cache" TEFLON_DEBUG=verbose ETNA_MESA_DEBUG="${TEFLON_OFFLINE_DEBUG:-ml_msgs}")
[ -n "$SHADER" ] && envs+=(ETNA_ML_SHADER="$SHADER")
cd "$OUT" || exit 2
env -u ETNA_ML_SHADER -u ETNA_MESA_DEBUG -u ETNA_ML_AXI_SRAM -u ETNA_CS_BARRIER_REG_BUDGET "${envs[@]}" \
   bwrap --dev-bind / / --dev /dev \
   strace -f -o "$OUT/strace.txt" -E LD_PRELOAD="$SHIM" -e trace=openat,open,ioctl \
   "$PY" -I "$AI/npu-shader/teflon_check.py" -m "$MODEL" -r "$REF" -e "$LIB" -n 1 > "$OUT/stdout.txt" 2>&1
rc=$?
rm -rf "$OUT/cache"
grep -q "npu shim: fake GC8000" "$OUT/stdout.txt" || { echo "teflon_offline: the shim did not load ($OUT)"; exit 2; }
if grep -E 'openat\(.*"/dev/dri/(card|renderD129)' "$OUT/strace.txt" | grep -v ENOENT | grep -q .; then
   echo "teflon_offline: a real DRM node was opened?! ($OUT/strace.txt)"; exit 2
fi
echo "teflon_offline: rc=$rc submits=$(ls "$OUT"/shim_submit_*.stream 2>/dev/null | wc -l) $OUT"

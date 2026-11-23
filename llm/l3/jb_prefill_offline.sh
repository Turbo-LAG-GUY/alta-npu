#!/usr/bin/env bash
# jb_prefill_offline.sh (2026-11-23, CPU only): jb_prefill --npu against the FAKE NPU (npu-shader/shim), inside bwrap with
# an empty /dev (nothing can reach /dev/dri). Plumbing only: the shim executes nothing, so the values are garbage.
# Usage: jb_prefill_offline.sh OUT JB_PREFILL_BIN PACK_DIR AUX PROMPT [jb_prefill options]
set -u
OUT=$(realpath -m "$1") BIN=$2; shift 2
AI=/home/jordan/Documents/AI_Projects
SHIM=$AI/npu-shader/shim/libnpu_etna_shim.so
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$OUT" || exit 2
env -u ETNA_ML_SHADER -u ETNA_MESA_DEBUG -u ETNA_ML_AXI_SRAM -u ETNA_ML_NO_FLUSH_INDEPENDENT NPU_SHIM_DUMP_DIR="$OUT" \
   bwrap --dev-bind / / --dev /dev env LD_PRELOAD="$SHIM" "$BIN" --npu --offline "$@" > "$OUT/stdout.txt" 2>&1
rc=$?
grep -q "npu shim: fake GC8000" "$OUT/stdout.txt" || { echo "jb_prefill_offline: the shim did not load ($OUT)"; exit 2; }
echo "jb_prefill_offline: rc=$rc submits=$(ls "$OUT"/shim_submit_*.stream 2>/dev/null | wc -l) $(grep -h '^{' "$OUT/stdout.txt")"

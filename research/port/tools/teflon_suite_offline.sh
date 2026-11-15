#!/usr/bin/env bash
# teflon_suite_offline.sh - run every test of Mesa's test_teflon against the FAKE NPU (drm shim), one process per
# test. CPU only, needs no OK: inside bwrap with an empty /dev nothing can reach /dev/dri, the shim executes nothing.
# The tests "fail" (the fake NPU returns zeros); the point is what each test would send to the NPU, and crashes.
# Written 2026-11-15.
# Usage: teflon_suite_offline.sh OUT [asan]     asan = use the sanitizer build of the delegate
#   OUT/<family>/<test>/out.txt + shim_submit_*.{stream,bos}; OUT/summary.tsv: family test rc submits ops
set -u
OUT=$(realpath -m "$1"); MODE=${2:-}
AI=/home/jordan/Documents/AI_Projects
B=$AI/mesa-upstream-fixes/build-tests/src/gallium/targets/teflon
SHIM=$AI/npu-shader/shim/libnpu_etna_shim.so
LIB=$AI/mesa-upstream-fixes/build/src/gallium/targets/teflon/libteflon.so
PRE=$SHIM
if [ "$MODE" = asan ]; then
   LIB=$AI/mesa-upstream-asan/build/src/gallium/targets/teflon/libteflon.so
   PRE="$(gcc -print-file-name=libasan.so) $SHIM"
fi
export LD_LIBRARY_PATH=$AI/tflite-2.13/lib TEFLON_TEST_DATA=$B/tests TEFLON_TEST_DELEGATE=$LIB
mkdir -p "$OUT"; : > "$OUT/summary.tsv"
for fam in $(ls "$B/tests/models"); do
   for m in "$B/tests/models/$fam"/*.tflite; do
      t=$(basename "$m" .tflite); d=$OUT/$fam/$t; rm -rf "$d"; mkdir -p "$d"; cd "$d" || exit 2
      env -u ETNA_ML_SHADER -u ETNA_MESA_DEBUG -u ETNA_ML_AXI_SRAM NPU_SHIM_DUMP_DIR="$d" TEFLON_DEBUG=verbose \
         ASAN_OPTIONS=detect_leaks=0:log_path="$d/asan" LD_PRELOAD="$PRE" \
         timeout 300 bwrap --dev-bind / / --dev /dev "$B/test_teflon" --gtest_filter="teflon/$fam.$t" > out.txt 2>&1
      rc=$?
      grep -q "npu shim: fake GC8000" out.txt || { echo "the shim did not load: $d"; exit 2; }
      n=$(ls shim_submit_*.stream 2>/dev/null | wc -l)
      ops=$(awk '/^idx type +ver support/{f=1;next} f&&/^$/{f=0} f&&$1~/^[0-9]+$/{printf "%s:%s ", $2, $4}' out.txt)
      printf '%s\t%s\t%s\t%s\t%s\n' "$fam" "$t" "$rc" "$n" "$ops" >> "$OUT/summary.tsv"
   done
done
echo "teflon_suite_offline done: $(wc -l < "$OUT/summary.tsv") tests, $OUT/summary.tsv"

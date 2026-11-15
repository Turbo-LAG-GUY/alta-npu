#!/usr/bin/env bash
# build-tflite.sh - build TensorFlow Lite 2.13 as a shared C/C++ library for Mesa's Teflon tests. NO ROOT, no NPU.
# Written 2026-11-15, NOT yet run (cmake was not installed). Run root-steps/190-teflon-test-deps.sh first.
#
# Why: Mesa's test_teflon loads "libtensorflow-lite.so" and calls TensorFlow Lite's C API to run each model on
# the CPU and on the NPU. The board only has the Python package (ai_edge_litert), whose libraries do not export
# that API (checked 2026-11-15), and Ubuntu 24.04 has no TensorFlow Lite package.
# Version: 2.13.1. Mesa's bundled headers (include/tensorflow/lite, last builtin op RIGHT_SHIFT = 161) are from
# 2.13, and test_teflon reads fields of TfLiteTensor directly, so the library has to match them.
#
# What it does (everything stays under ~/Documents/AI_Projects/tflite-2.13; nothing is installed system-wide):
#   1. downloads the TensorFlow 2.13.1 source from github.com (about 70 MB, shallow clone)
#   2. cmake configure: shared library, XNNPACK off (the plain CPU kernels are the reference; less to build).
#      cmake downloads TensorFlow Lite's own dependencies (abseil, eigen, flatbuffers, ruy, ...) from the network
#   3. builds at -j2 (expect one to two hours on the A311D)
#   4. checks that the library exports what test_teflon needs: the C API functions, and the C++ kernel
#      registration functions (tflite::ops::builtin::Register_*). test_teflon creates its interpreter with only
#      the kernels it finds through those, so a library without them cannot run any model. Stops if either is
#      missing (whether 2.13's cmake build puts the C API into this library is not verified yet)
#   5. leaves the result at ~/Documents/AI_Projects/tflite-2.13/lib/libtensorflow-lite.so
# Use afterwards: LD_LIBRARY_PATH=~/Documents/AI_Projects/tflite-2.13/lib for test_teflon.
# Remove: rm -rf ~/Documents/AI_Projects/tflite-2.13
# Usage: bash ~/alta-npu/scripts/build-tflite.sh        (safe to rerun: it continues where it stopped)
set -euo pipefail
TAG=v2.13.1
TOP=$HOME/Documents/AI_Projects/tflite-2.13
SRC=$TOP/tensorflow
BLD=$TOP/build
OUT=$TOP/lib
JOBS=2
NEED="TfLiteInterpreterCreateWithSelectedOps TfLiteInterpreterOptionsAddBuiltinOp TfLiteInterpreterOptionsAddCustomOp
TfLiteInterpreterOptionsAddDelegate TfLiteInterpreterOptionsSetErrorReporter TfLiteInterpreterOptionsCreate
TfLiteInterpreterAllocateTensors TfLiteInterpreterInvoke TfLiteInterpreterGetInputTensor
TfLiteInterpreterGetOutputTensor TfLiteTensorCopyFromBuffer TfLiteTensorCopyToBuffer TfLiteModelCreate"

command -v cmake >/dev/null || { echo "cmake is missing: run sudo bash ~/alta-npu/root-steps/190-teflon-test-deps.sh first"; exit 1; }
mkdir -p "$TOP" "$OUT"

missing() { # LIB -> prints the needed C functions the library does not export
   local syms; syms=$(nm -D --defined-only "$1" | awk '{print $3}')
   for s in $NEED; do echo "$syms" | grep -qx "$s" || echo "$s"; done
}

if [ ! -d "$SRC/.git" ]; then
   echo "== 1. download TensorFlow $TAG"
   git clone --depth 1 --branch "$TAG" https://github.com/tensorflow/tensorflow.git "$SRC"
fi

# GCC 13 no longer pulls <cstdint> in through other headers; sources of 2023 sometimes rely on that.
FLAGS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_GPU=OFF
       "-DCMAKE_CXX_FLAGS=-include cstdint")

echo "== 2. configure"
[ -f "$BLD/build.ninja" ] || [ -f "$BLD/Makefile" ] || cmake -S "$SRC/tensorflow/lite" -B "$BLD" "${FLAGS[@]}"
echo "== 3. build at -j$JOBS (long)"
nice cmake --build "$BLD" -j"$JOBS"

LIB=$(find "$BLD" -maxdepth 1 -name 'libtensorflow-lite.so*' | head -1)
[ -n "$LIB" ] || { echo "the build made no libtensorflow-lite.so in $BLD"; exit 1; }

echo "== 4. check the C API"
m=$(missing "$LIB")
if [ -n "$m" ]; then
   echo "the library does not export: $(echo $m)"
   echo "TensorFlow's separate C library (tensorflow/lite/c) is no substitute: it hides the C++ kernel functions"
   echo "the tests also need. The C API sources would have to be added to this build (not done; stop here)."
   exit 1
fi
cp -f "$LIB" "$OUT/libtensorflow-lite.so"
# the shared build links its dependencies as separate shared libraries: keep them beside it
find "$BLD/_deps" -name '*.so*' -exec cp -Pf {} "$OUT/" \; 2>/dev/null || true

echo "== 5. result"
ls -l "$OUT/libtensorflow-lite.so"
m=$(missing "$OUT/libtensorflow-lite.so"); [ -z "$m" ] || { echo "still missing: $m"; exit 1; }
if ldd "$OUT/libtensorflow-lite.so" | grep -q 'not found'; then
   echo "note: needs LD_LIBRARY_PATH=$OUT to find:"; ldd "$OUT/libtensorflow-lite.so" | grep 'not found'
fi
ref=$(nm -D --defined-only "$OUT/libtensorflow-lite.so" | grep -c 'tflite3ops7builtin.*Register_' || true)
echo "kernel registration functions exported: $ref"
[ "$ref" -gt 50 ] || { echo "too few kernel registration functions: test_teflon could not run models on the CPU"; exit 1; }
echo "build-tflite done: $OUT/libtensorflow-lite.so"

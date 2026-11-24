#!/usr/bin/env bash
# build.sh (2026-11-23): llm/l3/glue/jb_glue.c -> npu-llm/bin/libjbglue.so (+ a frozen copy libjbglue-<sha8>.so)
set -euo pipefail
# -ffp-contract=off (2026-11-24): no fused multiply-adds GCC invents, as in jb_prefill (-std=c11) and numpy
here="$(cd "$(dirname "$0")" && pwd)"
OUT=/home/jordan/Documents/AI_Projects/npu-llm/bin/libjbglue.so
gcc -O3 -mcpu=cortex-a73 -ffp-contract=off -fopenmp -fPIC -shared -Wall -Wextra -Wno-unused-parameter -o "$OUT" "$here/jb_glue.c" -lm
s=$(sha256sum "$OUT" | cut -c1-8)
cp "$OUT" "${OUT%.so}-$s.so"
cp "$here/jb_glue.c" "$here/jb_glue-$s.c"
echo "built $OUT ($s)"

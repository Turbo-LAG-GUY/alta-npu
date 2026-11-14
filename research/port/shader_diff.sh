#!/usr/bin/env bash
# shader_diff.sh A.bin [B.bin]: disassemble etnaviv shader binaries with the port tree's Mesa decoder
# (mesa-compute/build/src/etnaviv/isa/etnaviv-disasm, built 2026-11-13) and, with two files, diff them.
# CPU only. Reference: runs/cl2023-180102/COPY256/mesa-shader-00000000.bin (2023 branch, proven copy kernel).
set -euo pipefail
D=/home/jordan/Documents/AI_Projects/mesa-compute/build/src/etnaviv/isa/etnaviv-disasm
if [ $# -eq 1 ]; then exec "$D" "$1"; fi
diff -u --label "$1" --label "$2" <("$D" "$1") <("$D" "$2") && echo "identical"

#!/usr/bin/env bash
# build_jb_run.sh (2026-11-19, LLM engine L3b, CPU only): build jb_run against the mesa-llm tree's already-built static
# libraries (read-only; the same libraries as its libteflon.so 678775c5), like npu-shader/build_cs_run.sh.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
S=/home/jordan/Documents/AI_Projects/mesa-llm
B=$S/build
OUT=/home/jordan/Documents/AI_Projects/npu-llm/bin/jb_run
FLAGS=$(python3 - "$B" <<'PY'
import json, os, shlex, sys
b = sys.argv[1]
for c in json.load(open(os.path.join(b, "compile_commands.json"))):
    if c["file"].endswith("drivers/etnaviv/etnaviv_ml.c"):
        out = []
        for a in shlex.split(c["command"]):
            if a.startswith("-I"):
                out.append("-I" + os.path.normpath(os.path.join(b, a[2:])))
            elif a.startswith(("-D", "-std=")) or a in ("-pthread", "-fno-math-errno"):
                out.append(a)
        print(" ".join(shlex.quote(x) for x in out))
        break
PY
)
LIBS=$(find "$B/src" -name "*.a" -not -path "*.p/*" -not -name "libgtest*" | sort | tr '\n' ' ')
eval "gcc -O2 -g $FLAGS -I$S/src/gallium/winsys -Wall -Wno-unused-function -o $OUT $here/jb_run.c \
  -Wl,--start-group $LIBS -Wl,--end-group $(pkg-config --libs libdrm) -lstdc++ -lm -lpthread -lz -lzstd -lexpat -ldl"
echo "built $OUT"

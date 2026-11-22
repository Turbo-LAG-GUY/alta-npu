#!/bin/bash
# sram-limits offline check, NO NPU: bubblewrap with an empty /dev + LD_PRELOAD fake NPU shim (as validate_offline.py).
# usage: offline_check.sh OUTDIR LABEL LIBRARY MODEL [ENV=VALUE ...]
set -u
AI=/home/jordan/Documents/AI_Projects
out=$1/$2; lib=$3; model=$4; shift 4
mkdir -p "$out/cache"
cd "$out"
bwrap --dev-bind / / --dev /dev --chdir "$out" \
  env -u ETNA_ML_AXI_SRAM -u ETNA_MESA_DEBUG -u ETNA_ML_STRIPS -u ETNA_ML_STRIPS_COLUMNS -u LD_PRELOAD \
  LD_PRELOAD=$AI/npu-shader/shim/libnpu_etna_shim.so NPU_SHIM_DUMP_DIR="$out" ETNA_ML_DISK_CACHE_DIR="$out/cache" \
  "$@" $AI/.venv/bin/python -I -c '
import os, sys
assert not os.path.exists("/dev/dri"), "real DRM device visible"
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
model, lib = sys.argv[1:]
it = Interpreter(model_path=model, experimental_delegates=[load_delegate(lib)])
it.allocate_tensors()
i = it.get_input_details()[0]
it.set_tensor(i["index"], np.zeros(i["shape"], i["dtype"]))
it.invoke()
print("invoke OK", [o["name"] for o in it.get_output_details()])
' "$model" "$lib" > stdout.txt 2>&1
echo "rc=$?" >> stdout.txt
grep -E "AXI SRAM|strips:|invoke OK|abort|rc=|shim: fake" stdout.txt | grep -v "strips: strip " | head -8

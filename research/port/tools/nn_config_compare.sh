#!/usr/bin/env bash
# nn_config_compare.sh - for each model: dump the NN/TP configs and compressed weights offline (fake NPU, nothing runs)
# with the kit library and with the port tree, and compare them field by field, ignoring buffer addresses
# (allocation order differs between the libraries). CPU only. Written 2026-11-14 for the v7 tiling fix (step 6h).
# Usage: nn_config_compare.sh OUTDIR MODEL.tflite...   (ETNA_ML_SHADER for the port run: $PORT_SHADER, default unset)
set -u
OUT=$1; shift
AI=/home/jordan/Documents/AI_Projects
KIT=$AI/mesa-axi-sram/build/src/gallium/targets/teflon/libteflon.so
PORT=${PORT_LIB:-$AI/mesa-compute/build/src/gallium/targets/teflon/libteflon.so}
T=/home/jordan/alta-npu/research/port/tools
PY=$AI/.venv/bin/python
mkdir -p "$OUT"
for m in "$@"; do
   name=$(basename "$m" .tflite)
   ref=${m%.tflite}_reference.npz
   if [ ! -e "$ref" ]; then   # a dummy reference: only the input shape matters offline
      ref=$OUT/$name.dummy.npz
      $PY -I -c "
import numpy as np, sys
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path=sys.argv[1]); it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
np.savez(sys.argv[2], input=np.zeros(i['shape'][1:], i['dtype']), output=np.zeros(o['shape'][1:], o['dtype']))" "$m" "$ref" 2>/dev/null
   fi
   TEFLON_OFFLINE_DEBUG=ml_msgs,dump_shaders $T/teflon_offline.sh "$OUT/$name-kit" "$KIT" "$m" "$ref" >/dev/null
   TEFLON_OFFLINE_DEBUG=ml_msgs,dump_shaders $T/teflon_offline.sh "$OUT/$name-port" "$PORT" "$m" "$ref" ${PORT_SHADER:-} >/dev/null
   $PY -I - "$OUT/$name-kit" "$OUT/$name-port" "$name" <<'EOF'
import glob, os, sys
sys.path.insert(0, "/home/jordan/alta-npu/research/port/tools")
import tp_decode as D
a, b, name = sys.argv[1:4]
diffs, n = [], 0
for f in sorted(glob.glob(os.path.join(a, "mesa-nn-*.bin")) + glob.glob(os.path.join(a, "mesa-tp-*.bin"))):
    g = os.path.join(b, os.path.basename(f))
    n += 1
    if not os.path.exists(g):
        diffs.append("%s missing in port" % os.path.basename(f)); continue
    x, y = D.decode(f), D.decode(g)
    for k in x:
        if "address" in k or k.startswith("unused"):
            continue
        if x[k] != y[k]:
            diffs.append("%s %s %d -> %d" % (os.path.basename(f), k, x[k], y[k]))
for f in sorted(glob.glob(os.path.join(a, "mesa-compressed-*.bin"))):
    g = os.path.join(b, os.path.basename(f))
    n += 1
    if not os.path.exists(g) or open(f, "rb").read() != open(g, "rb").read():
        diffs.append("%s weights differ" % os.path.basename(f))
extra = set(map(os.path.basename, glob.glob(os.path.join(b, "mesa-nn-*.bin")))) - set(map(os.path.basename, glob.glob(os.path.join(a, "mesa-nn-*.bin"))))
print("%-28s %3d configs/weights: %s" % (name, n, "IDENTICAL (addresses aside)" if not diffs and not extra else "; ".join(diffs[:12]) + (" (+%d more)" % (len(diffs) - 12) if len(diffs) > 12 else "")))
EOF
done

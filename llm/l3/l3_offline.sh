#!/usr/bin/env bash
# l3_offline.sh (2026-11-19, LLM engine L3, CPU only): the tiling overrides of the mesa-llm test tree on the FAKE NPU.
# For each (model, ETNA_ML_V7_KPC_CAP, ETNA_ML_V7_TILE_H) case: one teflon_offline.sh run with config dumps, then
# the decoded NN config (tile, kernels per core, superblocks) and any driver error. Nothing reaches /dev/dri.
# Usage: l3/l3_offline.sh [OUT]
set -u
AI=/home/jordan/Documents/AI_Projects
LIB=$AI/mesa-llm/build/src/gallium/targets/teflon/libteflon.so
M=$AI/npu-llm/models/sizes
OUT=${1:-$AI/npu-llm/runs/offline-l3-$(date +%Y%m%d-%H%M%S)}
T=/home/jordan/alta-npu/research/port/tools
mkdir -p "$OUT"
cases=(
   "smol_gateup_1x1_P 0 0" "smol_gateup_1x1_P 24 0" "smol_gateup_1x1_P 32 0" "smol_gateup_1x1_P 42 0" "smol_gateup_1x1_P 64 0"
   "qwen_gateup_1x1_P 0 0" "qwen_gateup_1x1_P 32 0" "qwen_gateup_1x1_P 64 0"
   "smol_gateup_4x4_P 0 0" "smol_gateup_4x4_P 32 0" "qwen_gateup_4x4_P 32 0"
   "smol_gateup_16x16_P 0 0" "smol_gateup_16x16_P 0 8" "smol_gateup_16x16_P 32 4" "smol_gateup_16x16_P 42 2"
   "qwen_gateup_16x16_P 0 8" "qwen_gateup_16x16_P 32 4" "qwen_gateup_16x16_P 42 2"
)
for c in "${cases[@]}"; do
   set -- $c
   n=$1 kpc=$2 th=$3 d="$OUT/$1_k$2_t$3"
   env ETNA_ML_V7_KPC_CAP=$kpc ETNA_ML_V7_TILE_H=$th TEFLON_OFFLINE_DEBUG=ml_msgs,dump_shaders \
      bash $T/teflon_offline.sh "$d" "$LIB" "$M/$n.tflite" "$M/${n}_reference.npz" > "$d.run.txt" 2>&1
done
/home/jordan/Documents/AI_Projects/.venv/bin/python -I - "$OUT" <<'EOF'
import glob, os, re, sys
sys.path.insert(0, "/home/jordan/alta-npu/research/port/tools")
import tp_decode as D
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*_k*_t*"))):
    if not os.path.isdir(d):
        continue
    s = open(os.path.join(d, "stdout.txt"), errors="replace").read()
    bad = len(re.findall(r"assert|abort|error|not supported|unsupported|fallback", s, re.I))
    til = re.findall(r"v7 tiling: tile (\d+)x(\d+) interleave (\d+) superblocks (\d+)", s)
    nn = glob.glob(os.path.join(d, "mesa-nn-*.bin"))
    x = D.decode(nn[0]) if nn else {}
    print("%-34s tile %sx%s kpc %-3s superblocks %-3s interleave %-2s submits %d bad %d" % (
        os.path.basename(d), x.get("out_image_tile_x_size"), x.get("out_image_tile_y_size"), x.get("kernels_per_core"),
        til[-1][3] if til else "?", til[-1][2] if til else "?", len(glob.glob(os.path.join(d, "shim_submit_*.stream"))), bad))
EOF

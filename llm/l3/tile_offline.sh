#!/usr/bin/env bash
# tile_offline.sh (2026-11-22, LLM engine, CPU only): every pack of a set on the FAKE NPU with given driver switches
# (default: the 1.7B tile rule ETNA_ML_V7_TILE_FIT=262144 + ETNA_ML_KEEP_IMAGE_CACHE=1 + ETNA_ML_NO_FLUSH_INDEPENDENT=1),
# jb_run one pack at a time (memory-light: a 1.7B layer step is up to 34 MB of weights), driver dumps for every pack.
# Per job: rc 0, no driver complaint, kernels per core <= 21, tile <= image (the H < W trap), kernel area not empty,
# image cache inside the SRAM and above the kernels, only the last job of a pack flushes when NO_FLUSH is on.
# Usage: tile_offline.sh PACK_DIR OUT [ENV...]     e.g. tile_offline.sh $J/real-1p7b $OUT "ETNA_ML_V7_TILE_FIT=262144 ..."
set -u
AI=/home/jordan/Documents/AI_Projects
L3=/home/jordan/alta-npu/llm/l3
PD=$1 OUT=$2
ENVS=${3:-ETNA_ML_V7_TILE_FIT=262144 ETNA_ML_KEEP_IMAGE_CACHE=1}
JR=${JB_RUN_BIN:-$AI/npu-llm/bin/jb_run-674d417f}
echo "jb_run $(sha256sum $JR | cut -c1-8)" > "$2.jb_run.txt" 2>/dev/null || true
mkdir -p "$OUT"; echo "$ENVS" > "$OUT/env.txt"
for f in "$PD"/L*.jbp; do
   n=$(basename "$f" .jbp)
   JB_BIN=$JR JB_ENV="$ENVS" JB_DUMPS=1 bash $L3/jb_offline.sh "$OUT/$n" "$f" -n 1 --wait spin > "$OUT/$n.run.txt" 2>&1
   rm -f "$OUT/$n"/mesa-input-*.bin "$OUT/$n"/mesa-compressed-*.bin "$OUT/$n"/strace.txt
done
$AI/.venv/bin/python -I - "$OUT" "$ENVS" <<'PY'
import glob, json, os, re, sys
sys.path.insert(0, "/home/jordan/alta-npu/research/port/tools")
import tp_decode as D
out, envs = sys.argv[1], sys.argv[2]
noflush = "ETNA_ML_NO_FLUSH_INDEPENDENT=1" in envs
SRAM, problems, kinds = 524288, [], {}
for d in sorted(glob.glob(os.path.join(out, "L*"))):
    if not os.path.isdir(d):
        continue
    n = os.path.basename(d)
    s = open(os.path.join(d, "stdout.txt"), errors="replace").read()
    run = open(d + ".run.txt", errors="replace").read()
    if "rc=0" not in run:
        problems.append("%s: rc" % n)
    if re.search(r"assert|abort|error|not supported|fallback|failed|can't", s, re.I):
        problems.append("%s: driver complaint" % n)
    nn = [D.decode(p) for p in sorted(glob.glob(os.path.join(d, "mesa-nn-*.bin")))]
    if not nn:
        problems.append("%s: no NN configs" % n)
    for j, c in enumerate(nn):
        km, ks, ke = c["kernel_caching_mode"], c["kernel_cache_start_address"], c["kernel_cache_end_address"]
        im, ist, ie = c["image_caching_mode"], c["image_cache_start_address"], c["image_cache_end_address"]
        tx, ty = c["out_image_tile_x_size"], c["out_image_tile_y_size"]
        ok = (c["kernels_per_core"] <= 21 and ke <= SRAM and (km == 0 or ke > ks) and tx <= c["out_image_x_size"]
              and ty <= c["out_image_y_size"])
        if im == 1:
            ok = ok and ist >= ke and ie <= SRAM and ie > ist
        if noflush and c["nn_layer_flush"] != (1 if j == len(nn) - 1 else 0):
            ok = False
        if not ok:
            problems.append("%s job %d: kpc %d tile %dx%d of %dx%d k%d %#x..%#x img%d %#x..%#x flush %d" % (
                n, j, c["kernels_per_core"], tx, ty, c["out_image_x_size"], c["out_image_y_size"], km, ks, ke, im,
                ist, ie, c["nn_layer_flush"]))
        k = n.split("_", 1)[1]
        kinds.setdefault(k, set()).add("tile %dx%d kpc %d k%d ..%#x img%d %#x..%#x" % (
            tx, ty, c["kernels_per_core"], km, ke, im, ist, ie))
print("%d packs; env: %s" % (len(glob.glob(os.path.join(out, "L*.run.txt"))), envs))
for k in sorted(kinds):
    for l in sorted(kinds[k]):
        print("  %-12s %s" % (k, l))
json.dump(dict(env=envs, kinds={k: sorted(v) for k, v in kinds.items()}, problems=problems),
          open(os.path.join(out, "summary.json"), "w"), indent=1)
print("problems %d" % len(problems))
for p in problems[:30]:
    print("  " + p)
print("PASS" if not problems else "FAIL")
sys.exit(1 if problems else 0)
PY

#!/usr/bin/env bash
# axi_offline.sh (2026-11-22, LLM engine, CPU only): the 1.7B step input in the AXI SRAM, on the FAKE NPU.
# 1.7B at 256 tokens: the step input (2048 ch x 256 tokens = 512 KiB; down 8192 ch) never fits the 512 KiB VIP SRAM, so
# every conv job re-reads it from DDR once per superblock. The CPU can't write the AXI SRAM (no CPU mapping in the
# kernel patch), so the NPU copies it there: a channels-last graph input, ONE shared driver transpose (TP) job
# (ETNA_ML_SHARE_INPUT_TRANSPOSE=1) whose output the AXI placement (ETNA_ML_AXI_SRAM=all, patch 0002 ported to
# mesa-llm) puts in the AXI SRAM, and every conv reads it from there.
# Packs npu-llm/models/jb/axi-1p7b (llm/l3/jb_pack.py synth; jobs <= 8 and weights < 512 KiB per job, inside what ran):
#   A8 / A2: IC 2048, 8 / 2 jobs x 224 out ch, 256 tokens (16x16)    D4: IC 8192, 4 jobs x 56, 64 tokens (8x8)
# Variants: P0 planar (today), PT planar + ETNA_ML_V7_TILE_H (A 8, D 4) + ETNA_ML_KEEP_IMAGE_CACHE=1 (the no-AXI
# alternative: a smaller tile so the image cache fits), N1 nhwc + shared transpose (input copy in DDR),
# N2 = N1 + ETNA_ML_AXI_SRAM=all.
# Checks per job: rc 0, no driver complaint, NN / TP job counts, every NN config's cache layout valid (kernel area not
# empty, image inside the SRAM and above the kernels), kernels per core <= 21, N2: the convs read the address the TP
# job writes, and that is not the address they read in N1 (the AXI BO), one tensor placed.
# Usage: llm/l3/axi_offline.sh [OUT]
set -u
AI=/home/jordan/Documents/AI_Projects
L3=/home/jordan/alta-npu/llm/l3
P=$AI/npu-llm/models/jb/axi-1p7b
OUT=${1:-$AI/npu-llm/runs/offline-axi-$(date +%Y%m%d-%H%M%S)}
JR=$AI/npu-llm/bin/jb_run-943bf0c2
[ "$(sha256sum $JR | cut -c1-8)" = 943bf0c2 ] || { echo "jb_run is not 943bf0c2"; exit 1; }
mkdir -p "$OUT"
for pk in A8 A2 D4; do
   case $pk in D*) th=4 ;; *) th=8 ;; esac
   for v in P0 PT N1 N2; do
      case $v in
         P0) f=planar e="" ;;
         PT) f=planar e="ETNA_ML_V7_TILE_H=$th ETNA_ML_KEEP_IMAGE_CACHE=1" ;;
         N1) f=nhwc e="ETNA_ML_SHARE_INPUT_TRANSPOSE=1" ;;
         N2) f=nhwc e="ETNA_ML_SHARE_INPUT_TRANSPOSE=1 ETNA_ML_AXI_SRAM=all" ;;
      esac
      JB_BIN=$JR JB_ENV="$e" JB_DUMPS=1 bash $L3/jb_offline.sh "$OUT/${pk}_$v" "$P/${pk}_$f.jbp" -n 1 --wait spin \
         > "$OUT/${pk}_$v.run.txt" 2>&1
      echo "$e" > "$OUT/${pk}_$v.env"
      rm -f "$OUT/${pk}_$v"/mesa-input-*.bin
   done
done
$AI/.venv/bin/python -I - "$OUT" <<'PY'
import glob, json, os, re, sys
sys.path.insert(0, "/home/jordan/alta-npu/research/port/tools")
import tp_decode as D
out = sys.argv[1]
SRAM, problems, rows = 524288, [], []
for d in sorted(glob.glob(os.path.join(out, "*_*"))):
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
    tp = [D.decode(p, "tp") for p in sorted(glob.glob(os.path.join(d, "mesa-tp-*.bin")))]
    placed = re.findall(r"AXI SRAM: placed (\d+) of (\d+).*using (\d+) of", s)
    lay = set()
    for c in nn:
        km, ks, ke = c["kernel_caching_mode"], c["kernel_cache_start_address"], c["kernel_cache_end_address"]
        im, ist, ie = c["image_caching_mode"], c["image_cache_start_address"], c["image_cache_end_address"]
        ok = c["kernels_per_core"] <= 21 and ke <= SRAM and (km == 0 or ke > ks)
        if im == 1:
            ok = ok and ist >= ke and ie <= SRAM and ie > ist
        if not ok:
            problems.append("%s: bad NN config kpc %d k %d %#x..%#x img %d %#x..%#x" % (n, c["kernels_per_core"], km, ks, ke, im, ist, ie))
        lay.add("k%d %#x..%#x img%d %#x..%#x tile %dx%d kpc %d" % (km, ks, ke, im, ist, ie, c["out_image_tile_x_size"],
                                                                 c["out_image_tile_y_size"], c["kernels_per_core"]))
    ins = sorted({c["in_image_address"] for c in nn})
    rows.append(dict(run=n, nn=len(nn), tp=len(tp), placed=placed, layouts=sorted(lay), conv_inputs=["%#x" % a for a in ins]))
    print("%-6s NN %d TP %d placed %s conv input addr %s" % (n, len(nn), len(tp), placed or "-", rows[-1]["conv_inputs"]))
    for l in sorted(lay):
        print("         " + l)
by = {r["run"]: r for r in rows}
for pk in ("A8", "A2", "D4"):
    r1, r2 = by.get(pk + "_N1"), by.get(pk + "_N2")
    if not r1 or not r2:
        problems.append(pk + ": missing N1/N2")
        continue
    for r in (r1, r2):
        if r["tp"] != 1 or len(r["conv_inputs"]) != 1:
            problems.append("%s: %d TP jobs, %d conv input addresses (want 1 shared)" % (r["run"], r["tp"], len(r["conv_inputs"])))
    if not r2["placed"] or r2["placed"][0][0] != "1":
        problems.append(pk + "_N2: input copy not placed in the AXI SRAM")
    if r1["placed"]:
        problems.append(pk + "_N1: something placed without ETNA_ML_AXI_SRAM")
    if r1["conv_inputs"] == r2["conv_inputs"]:
        problems.append(pk + "_N2: convs read the same address as N1 (not the AXI BO?)")
    if by.get(pk + "_P0", {}).get("tp", 1) != 0:
        problems.append(pk + "_P0: planar pack has a TP job")
json.dump(dict(rows=rows, problems=problems), open(os.path.join(out, "summary.json"), "w"), indent=1)
print("problems %d" % len(problems))
for p in problems:
    print("  " + p)
print("PASS" if not problems else "FAIL")
sys.exit(1 if problems else 0)
PY

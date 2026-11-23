#!/usr/bin/env bash
# pieces_offline.sh (2026-11-22, LLM engine, CPU only): token pieces through the AXI SRAM, on the FAKE NPU.
# The 1.7B down step at 256 tokens has a 2 MiB input (8192 ch), more than the 1 MiB AXI SRAM. Cut into P token pieces
# (P2 = 2 x 128 tokens (16 high x 8 wide: H > W takes the fixed V7 tiling path 348aef38; 8 x 16 got a 16-row tile on an 8-row image) = 1 MiB each, P4 = 4 x 64 tokens = 512 KiB each) in ONE submit: per piece one shared input
# transpose (ETNA_ML_SHARE_INPUT_TRANSPOSE=1) whose output is placed in the AXI SRAM (ETNA_ML_AXI_SRAM=all), then the
# piece's conv jobs read it there. A piece's copy is live only until its last conv, so Mesa's allocator (patch 0002,
# lifetime-aware first fit) gives every piece the SAME SRAM space: the input is read from DDR once, the re-reads of the
# jobs are on chip. Trade: the weights are read once per piece instead of once.
# Packs npu-llm/models/jb/axi-pieces (llm/l3/jb_pack.py): 4 jobs x 56 out ch, IC 8192, 256 tokens in all.
#   W1 whole step (16x16 tokens, input 2 MiB: never fits)  P2 2 pieces of 16x8  P4 4 pieces of 8x8
# Variants: P0 planar (W1 only, today's path), N1 nhwc + shared transpose (copies in DDR), N2 = N1 + AXI SRAM.
# Library: mesa-llm 4eb662fc (= 4b666c08 AXI port + the one-submit guards of patch 0006) (no npu_no_batching / npu_parallel, 0x4000-word
# bound, pre-flush, split warning); jb_run 21b8ac1f (several graph inputs).
# Usage: llm/l3/pieces_offline.sh [OUT]
set -u
AI=/home/jordan/Documents/AI_Projects
L3=/home/jordan/alta-npu/llm/l3
P=$AI/npu-llm/models/jb/axi-pieces
OUT=${1:-$AI/npu-llm/runs/offline-pieces-$(date +%Y%m%d-%H%M%S)}
JR=$AI/npu-llm/bin/jb_run-21b8ac1f
[ "$(sha256sum $JR | cut -c1-8)" = 21b8ac1f ] || { echo "jb_run is not 21b8ac1f"; exit 1; }
mkdir -p "$OUT"
for run in W1_P0 W1_N1 W1_N2 P2_N1 P2_N2 P4_N1 P4_N2; do
   pk=${run%_*} v=${run#*_}
   case $v in
      P0) f=planar e="" ;;
      N1) f=nhwc e="ETNA_ML_SHARE_INPUT_TRANSPOSE=1" ;;
      N2) f=nhwc e="ETNA_ML_SHARE_INPUT_TRANSPOSE=1 ETNA_ML_AXI_SRAM=all" ;;
   esac
   JB_BIN=$JR JB_ENV="$e" JB_DUMPS=1 bash $L3/jb_offline.sh "$OUT/$run" "$P/${pk}_$f.jbp" -n 2 --wait spin \
      > "$OUT/$run.run.txt" 2>&1
   echo "$e" > "$OUT/$run.env"
   rm -f "$OUT/$run"/mesa-input-*.bin "$OUT/$run"/mesa-output-*.bin
done
$AI/.venv/bin/python -I - "$OUT" "$P" <<'PY'
import glob, json, os, re, sys
sys.path.insert(0, "/home/jordan/alta-npu/research/port/tools")
sys.path.insert(0, "/home/jordan/alta-npu/llm/l3")
import numpy as np
import tp_decode as D
import jb_pack as J
out, packs = sys.argv[1], sys.argv[2]
VIP, AXI = 524288, 1 << 20
PIECES = dict(W1=1, P2=2, P4=4)
G = 4
problems, rows = [], {}

# 1. the reference: the pieces' expected outputs = the whole 256-token conv, recomputed from the pack's own tensors
for pk in ("P2", "P4"):
    T, ops, x, exp, _ = J.read_pack(os.path.join(packs, pk + "_nhwc.jbp"))
    p = PIECES[pk]
    ins = [t for t in T if not t["const"] and t["idx"] not in {o[4] for o in ops}]
    if len(ins) != p:
        problems.append("%s: %d graph inputs, want %d" % (pk, len(ins), p))
    convs = [o for o in ops if o[0] == 1]
    xa = np.frombuffer(x, np.int8).reshape(-1, 8192)                        # all 256 tokens, channels-last
    n = xa.shape[0] // p
    worst = 0
    for k, (typ, i, a, b, o) in enumerate(convs):
        piece, g = divmod(k, G)
        w = np.frombuffer(T[a]["data"], np.int8).reshape(T[a]["dims"][0], -1)
        y = J.ref_conv(xa[piece * n:(piece + 1) * n], T[i]["q"], w, T[a]["q"], np.zeros(w.shape[0], np.int32), T[o]["q"])
        e = np.frombuffer(exp[k], np.int8).reshape(w.shape[0], -1)           # planar output [C, tokens]
        worst = max(worst, int(np.abs(y.T.astype(int) - e.astype(int)).max()))
        if k < G and convs[k + G * (p - 1)][2] != a:
            problems.append("%s: job %d of the last piece uses other weights" % (pk, k))
    print("%s reference: stored expected vs direct conv of the token slices: max %d" % (pk, worst))
    if worst:
        problems.append("%s: reference mismatch %d" % (pk, worst))

# 2. the driver's jobs on the fake NPU
for d in sorted(glob.glob(os.path.join(out, "*_*"))):
    if not os.path.isdir(d):
        continue
    n = os.path.basename(d)
    pk, v = n.split("_")
    s = open(os.path.join(d, "stdout.txt"), errors="replace").read()
    run = open(d + ".run.txt", errors="replace").read()
    if "rc=0" not in run:
        problems.append("%s: rc (%s)" % (n, run.strip()[-200:]))
    m = re.search(r"submits=(\d+)", run)
    if not m or m.group(1) != "3":                  # init + 2 invokes, each invoke one submit
        problems.append("%s: %s submits, want 3 (init + one per invoke)" % (n, m and m.group(1)))
    if re.search(r"assert|abort|error|not supported|fallback|failed|can't|split across|more than one submit", s, re.I):
        problems.append("%s: driver complaint" % n)
    # op order from the dump index of the first invoke (the counter runs on across invokes)
    seq = []
    for f in glob.glob(os.path.join(d, "mesa-nn-*-000.bin")) + glob.glob(os.path.join(d, "mesa-tp-*-000.bin")):
        k = int(os.path.basename(f).split("-")[2])
        seq.append((k, "tp" if "-tp-" in f else "nn", f))
    seq.sort()
    nops = len(seq) // 2 if len(seq) % 2 == 0 and seq else len(seq)
    first = seq[:nops]                               # dumps of invoke 1 (invoke 2 repeats them)
    order = "".join("T" if t == "tp" else "n" for _, t, _ in first)
    nn = [D.decode(f) for _, t, f in first if t == "nn"]
    tp = [D.decode(f, "tp") for _, t, f in first if t == "tp"]
    placed = re.findall(r"AXI SRAM: placed (\d+) of (\d+).*using (\d+) of", s)
    for c in nn:
        km, ks, ke = c["kernel_caching_mode"], c["kernel_cache_start_address"], c["kernel_cache_end_address"]
        im, ist, ie = c["image_caching_mode"], c["image_cache_start_address"], c["image_cache_end_address"]
        ok = c["kernels_per_core"] <= 21 and ke <= VIP and (km == 0 or ke > ks)
        if im == 1:
            ok = ok and ist >= ke and ie <= VIP and ie > ist
        ok = ok and c["out_image_tile_x_size"] <= c["in_image_x_size"] and c["out_image_tile_y_size"] <= c["in_image_y_size"]
        if not ok:
            problems.append("%s: bad NN config kpc %d k %d %#x..%#x img %d %#x..%#x tile %dx%d image %dx%d" % (
                n, c["kernels_per_core"], km, ks, ke, im, ist, ie, c["out_image_tile_x_size"], c["out_image_tile_y_size"],
                c["in_image_x_size"], c["in_image_y_size"]))
    p = PIECES[pk]
    conv_in = [c["in_image_address"] for c in nn]
    tp_out = [c["out_image_base_address"] for c in tp]
    r = dict(run=n, order=order, nn=len(nn), tp=len(tp), placed=placed, conv_in=["%#x" % a for a in conv_in],
             tp_out=["%#x" % a for a in tp_out],
             layouts=sorted({"k%d %#x..%#x img%d tile %dx%d kpc %d" % (c["kernel_caching_mode"], c["kernel_cache_start_address"],
                             c["kernel_cache_end_address"], c["image_caching_mode"], c["out_image_tile_x_size"],
                             c["out_image_tile_y_size"], c["kernels_per_core"]) for c in nn}))
    rows[n] = r
    print("%-6s order %-22s NN %d TP %d placed %s conv inputs %s" % (n, order, len(nn), len(tp), placed or "-",
                                                                     sorted(set(r["conv_in"]))))
    for l in r["layouts"]:
        print("         " + l)
    if v == "P0":
        if tp or len(nn) != G:
            problems.append("%s: want 0 TP / %d NN, got %d / %d" % (n, G, len(tp), len(nn)))
        continue
    if order != ("T" + "n" * G) * p:
        problems.append("%s: job order %s, want %s" % (n, order, ("T" + "n" * G) * p))
    for k in range(p):                               # each piece's convs read what its own transpose wrote
        if set(conv_in[k * G:(k + 1) * G]) != {tp_out[k]} if k < len(tp_out) else True:
            problems.append("%s: piece %d convs read %s, its transpose wrote %s" % (
                n, k, conv_in[k * G:(k + 1) * G], tp_out[k:k + 1]))
    if v == "N1":
        if placed:
            problems.append("%s: something placed without ETNA_ML_AXI_SRAM" % n)
        if len(set(tp_out)) != p:
            problems.append("%s: %d distinct DDR copies, want %d" % (n, len(set(tp_out)), p))
    if v == "N2":
        if pk == "W1":
            if placed:
                problems.append("%s: a 2 MiB copy was placed in the 1 MiB SRAM?" % n)
            continue
        if not placed or placed[0][0] != str(p) or placed[0][1] != str(p):
            problems.append("%s: placed %s, want %d of %d" % (n, placed, p, p))
        elif int(placed[0][2]) != 8192 * 256 // p // 1024:
            problems.append("%s: uses %s KiB, want one piece (%d): the pieces must share the space" % (
                n, placed[0][2], 8192 * 256 // p // 1024))
        if len(set(tp_out)) != 1:
            problems.append("%s: the pieces' copies are at %d addresses, want 1 (reuse)" % (n, len(set(tp_out))))
        n1 = rows.get(pk + "_N1")
        if n1 and set(n1["tp_out"]) & set(r["tp_out"]):
            problems.append("%s: copy address also used in N1 (not the AXI BO?)" % n)
json.dump(dict(rows=rows, problems=problems), open(os.path.join(out, "summary.json"), "w"), indent=1)
print("problems %d" % len(problems))
for p in problems:
    print("  " + p)
verdict = "PASS" if not problems else "FAIL"
open(os.path.join(out, "VERDICT.txt"), "w").write(verdict + "\n" + "\n".join(problems) + "\n")
print(verdict)
sys.exit(1 if problems else 0)
PY

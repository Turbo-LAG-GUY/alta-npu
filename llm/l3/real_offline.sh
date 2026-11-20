#!/usr/bin/env bash
# real_offline.sh (2026-11-20, LLM engine R2, CPU only): every real-weight pack (llm/l3/real_packs.py) on the FAKE NPU
# (jb_offline.sh: bwrap with an empty /dev, strace check that no real DRM node is opened). Per pack: rc, submits,
# driver complaints; layers 0 and 29 also with the driver's dumps, decoded (tile, kernels per core) per job.
# Usage: real_offline.sh PACK_DIR OUT_DIR
set -u
PD=$1; OUT=$2
AI=/home/jordan/Documents/AI_Projects
mkdir -p "$OUT"
for f in "$PD"/L*.jbp; do
   n=$(basename "$f" .jbp)
   case "$n" in L00_*|L29_*) d=1 ;; *) d=0 ;; esac
   JB_DUMPS=$d bash /home/jordan/alta-npu/llm/l3/jb_offline.sh "$OUT/$n" "$f" -n 2 --wait spin > "$OUT/$n.run.txt" 2>&1
done
$AI/.venv/bin/python -I - "$OUT" <<'PY'
import glob, json, os, re, sys
sys.path.insert(0, "/home/jordan/alta-npu/research/port/tools")
import tp_decode as D
out = sys.argv[1]
rows, bad_total = [], 0
for d in sorted(glob.glob(os.path.join(out, "L*"))):
    if not os.path.isdir(d):
        continue
    s = open(os.path.join(d, "stdout.txt"), errors="replace").read()
    run = open(d + ".run.txt", errors="replace").read()
    bad = len(re.findall(r"assert|abort|error|not supported|unsupported|fallback|failed", s, re.I))
    rc = re.search(r"rc=(\d+)", run)
    subs = len(glob.glob(os.path.join(d, "shim_submit_*.stream")))
    cfg = [D.decode(p) for p in sorted(glob.glob(os.path.join(d, "mesa-nn-*-000.bin")))]
    kpc = [c.get("kernels_per_core") for c in cfg]
    tiles = ["%sx%s" % (c.get("out_image_tile_x_size"), c.get("out_image_tile_y_size")) for c in cfg]
    bad_total += bad + (rc is None or rc.group(1) != "0") + (subs != 3)
    rows.append(dict(pack=os.path.basename(d), rc=rc.group(1) if rc else "?", submits=subs, complaints=bad, kpc=kpc, tiles=tiles))
json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=1)
print("%d packs, problems %d (rc != 0, submits != 3 = init + 2 invokes, or driver complaints)" % (len(rows), bad_total))
for r in rows:
    if r["kpc"]:
        print("  %-20s kpc %s tiles %s" % (r["pack"], r["kpc"], r["tiles"]))
allk = [k for r in rows for k in r["kpc"] if k is not None]
print("kernels per core over all decoded jobs: max %s" % (max(allk) if allk else "?"))
PY

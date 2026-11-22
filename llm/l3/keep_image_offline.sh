#!/usr/bin/env bash
# keep_image_offline.sh (2026-11-22, LLM engine, CPU only): fix 1 for "wider jobs are slower" on the FAKE NPU.
# mesa-llm hook ETNA_ML_KEEP_IMAGE_CACHE=1 (etnaviv_ml_nn.c): when a job's weights don't fit the 512 KiB VIP SRAM, keep
# the input image cached at the top of the SRAM (weights get a partial cache below it) instead of dropping it.
# Every pack of the 4 / 2 / 1-job and mixed sets through jb_run 84bd6ece (jb_offline.sh, bwrap + empty /dev + strace),
# hook OFF and ON, with the driver's dumps; then per job:
#   OFF vs the old offline dumps (jb_run a79b7b5c, layers 0 and 29): NN config + compressed weights byte-identical
#   ON vs OFF: weights identical; NN config identical except, for jobs whose weights don't fit, exactly
#              kernel_cache_end_address, image_caching_mode, image_cache_start/end_address
#   ON layout: kernel 0x800 .. end >= 0x20000 (the smallest kernel area run on the NPU), image = end .. 512 KiB
# Then jb_engine 39c22a60 --offline (all 120 steps, rounds mode) on the 1-job set with the hook ON.
# Usage: systemd-run --user --unit=alta-llm-keepimg --collect llm/l3/keep_image_offline.sh [OUT]
set -u
AI=/home/jordan/Documents/AI_Projects
L3=/home/jordan/alta-npu/llm/l3
J=$AI/npu-llm/models/jb
OUT=${1:-$AI/npu-llm/runs/offline-keepimg-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
[ "$(sha256sum $AI/npu-llm/bin/jb_run | cut -c1-8)" = 84bd6ece ] || { echo "jb_run is not 84bd6ece"; exit 1; }
[ "$(sha256sum $AI/npu-llm/bin/jb_engine | cut -c1-8)" = 39c22a60 ] || { echo "jb_engine is not 39c22a60"; exit 1; }
/home/jordan/alta-npu/scripts/progress llm-keepimg-offline STARTING "llm/l3/keep_image_offline.sh: $OUT"
t0=$SECONDS
for set in real-135m-g1 real-135m-g2 real-135m real-135m-mixed; do
   for mode in off on; do
      d=$OUT/$set-$mode; mkdir -p "$d"
      for f in "$J/$set"/L*.jbp; do
         n=$(basename "$f" .jbp)
         if [ $mode = on ]; then e=(env ETNA_ML_KEEP_IMAGE_CACHE=1); else e=(env -u ETNA_ML_KEEP_IMAGE_CACHE); fi
         "${e[@]}" JB_DUMPS=1 bash $L3/jb_offline.sh "$d/$n" "$f" -n 1 --wait spin > "$d/$n.run.txt" 2>&1
         rm -f "$d/$n"/mesa-input-*.bin "$d/$n"/strace.txt.gz
      done
      echo "$set $mode done ($((SECONDS - t0)) s)"
   done
done
# the engine, all 120 steps of the 1-job set at both sizes, hook ON
for t in 256 1; do
   ETNA_ML_KEEP_IMAGE_CACHE=1 JB_NOSTRACE=1 bash $L3/jb_engine_offline.sh "$OUT/engine-g1-on-t$t" "$J/real-135m-g1" -n 2 \
      --tokens $t > "$OUT/engine-g1-on-t$t.run.txt" 2>&1
   echo "engine t$t: $(tail -n 1 "$OUT/engine-g1-on-t$t.run.txt")"
done
$AI/.venv/bin/python -I - "$OUT" "$AI/npu-llm/runs" <<'PY'
import glob, json, os, re, sys
sys.path.insert(0, "/home/jordan/alta-npu/research/port/tools")
import tp_decode as D
out, runs = sys.argv[1], sys.argv[2]
OLD = {"real-135m-g1": "offline-jb-20261121-real135m-g1", "real-135m-g2": "offline-jb-20261121-real135m-g2",
       "real-135m": "offline-jb-20261120-real135m"}
CACHE = {"kernel_cache_end_address", "image_caching_mode", "image_cache_start_address", "image_cache_end_address"}
SRAM, KMIN = 524288, 0x20000
problems, rows, summary = [], [], {}

def rd(p):
    return open(p, "rb").read()

def jobs(d):
    return sorted(glob.glob(os.path.join(d, "mesa-nn-*.bin")))

for s in ("real-135m-g1", "real-135m-g2", "real-135m", "real-135m-mixed"):
    changed = same = 0
    for don in sorted(glob.glob(os.path.join(out, s + "-on", "L*"))):
        if not os.path.isdir(don):
            continue
        n = os.path.basename(don)
        doff = os.path.join(out, s + "-off", n)
        for d in (don, doff):
            run = open(d + ".run.txt", errors="replace").read()
            txt = open(os.path.join(d, "stdout.txt"), errors="replace").read()
            if "rc=0" not in run or re.search(r"assert|abort|error|not supported|fallback|failed", txt, re.I):
                problems.append("%s %s: rc / driver complaint" % (s, d))
        a, b = jobs(doff), jobs(don)
        if len(a) != len(b) or not a:
            problems.append("%s %s: %d vs %d jobs" % (s, n, len(a), len(b)))
            continue
        for pa, pb in zip(a, b):
            ca, cb = pa.replace("mesa-nn-", "mesa-compressed-"), pb.replace("mesa-nn-", "mesa-compressed-")
            if rd(ca) != rd(cb):
                problems.append("%s %s %s: weights differ ON vs OFF" % (s, n, os.path.basename(pa)))
            fa, fb = D.decode(pa), D.decode(pb)
            # kernel_address etc. are GPU addresses: identical runs give identical softpin addresses
            diff = {k for k in fa if fa[k] != fb[k]}
            if not diff:
                same += 1
                continue
            changed += 1
            if not diff <= CACHE:
                problems.append("%s %s %s: unexpected fields %s" % (s, n, os.path.basename(pa), sorted(diff - CACHE)))
            ke, ist, ie = fb["kernel_cache_end_address"], fb["image_cache_start_address"], fb["image_cache_end_address"]
            ok = (fa["kernel_caching_mode"] == 2 and fa["image_caching_mode"] == 0 and fb["kernel_caching_mode"] == 2
                  and fb["image_caching_mode"] == 1 and fb["kernel_cache_start_address"] == 0x800
                  and KMIN <= ke == ist < ie == SRAM)
            if not ok:
                problems.append("%s %s %s: bad ON layout k %#x..%#x img %#x..%#x" % (s, n, os.path.basename(pa),
                                fb["kernel_cache_start_address"], ke, ist, ie))
            rows.append(dict(set=s, pack=n, job=os.path.basename(pa), kernel_end=ke, image=[ist, ie],
                             out_z=fb["out_image_z_size"], kpc=fb["kernels_per_core"]))
        # OFF vs the old run's dumps (old runs dumped layers 0 and 29 with -n 2: the first invoke's jobs come first)
        if s in OLD and n[:3] in ("L00", "L29"):
            od = os.path.join(runs, OLD[s], n)
            oj = jobs(od)[: len(a)]
            for pa, po in zip(a, oj):
                if rd(pa) != rd(po) or rd(pa.replace("mesa-nn-", "mesa-compressed-")) != rd(po.replace("mesa-nn-", "mesa-compressed-")):
                    fa, fo = D.decode(pa), D.decode(po)
                    dd = sorted(k for k in fa if fa[k] != fo[k])
                    problems.append("%s %s %s: OFF differs from the old dumps %s" % (s, n, os.path.basename(pa), dd))
    summary[s] = dict(jobs_changed=changed, jobs_identical=same)
    print("%-16s jobs with a new cache layout %3d, byte-identical %3d" % (s, changed, same))
kinds = sorted({(r["pack"].split("_", 1)[1], r["kernel_end"], r["out_z"]) for r in rows})
for k in kinds:
    print("  %-10s kernel area 0x800..%#x (%3d KB) image ..0x80000, out_z %d" % (k[0], k[1], (k[1] - 0x800) // 1024, k[2]))
for t in ("256", "1"):
    p = os.path.join(out, "engine-g1-on-t%s" % t, "stdout.txt")
    s = open(p, errors="replace").read() if os.path.exists(p) else ""
    m = re.search(r"compiled (\d+)", s)
    keep = len(re.findall(r"llm keep image cache", s))
    # offline the fake NPU computes nothing, so every pack is "bad" and jb_engine returns 4 (bad_packs); the old engine
    # 3b2807b6 (clean on the NPU) does the same. Pass = rc 0 or 4, init + 2 rounds x 120 submits, no driver complaint.
    run = open(os.path.join(out, "engine-g1-on-t%s.run.txt" % t), errors="replace").read()
    if not re.search(r"rc=[04] submits=241 ", run) or re.search(r"assert|abort|error|not supported|fallback|failed", s, re.I):
        problems.append("engine t%s: rc / submits / driver complaint" % t)
    print("engine g1 ON t%s: %s, hook taken %d times" % (t, m.group(0) if m else "no 'compiled' line", keep))
json.dump(dict(summary=summary, rows=rows, problems=problems), open(os.path.join(out, "summary.json"), "w"), indent=1)
print("problems %d" % len(problems))
for p in problems[:40]:
    print("  " + p)
print("PASS" if not problems else "FAIL")
sys.exit(1 if problems else 0)
PY
rc=$?
/home/jordan/alta-npu/scripts/progress llm-keepimg-offline DONE "rc=$rc ($([ $rc = 0 ] && echo PASS || echo FAIL)) $OUT, $((SECONDS - t0)) s"

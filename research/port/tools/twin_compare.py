#!/usr/bin/env python3
"""twin_compare.py (2026-11-14, step 6i, CPU only): compare two teflon_offline.sh dumps (TEFLON_OFFLINE_DEBUG=ml_msgs,
dump_shaders) of an int8 model and its uint8 twin (npu-shader/models/int8nn_models.py): every TP/NN config field
except buffer addresses, every compressed weight buffer byte for byte, and the submit command streams word for word
with the words that differ between the two runs listed (addresses move with allocation order).
Usage: twin_compare.py DIR_A DIR_B NAME   (prints one line; exit 0 = configs and weights identical)
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_decode as D  # noqa: E402

a, b, name = sys.argv[1:4]
diffs, n = [], 0
fa = sorted(glob.glob(os.path.join(a, "mesa-nn-*.bin")) + glob.glob(os.path.join(a, "mesa-tp-*.bin")))
fb = sorted(glob.glob(os.path.join(b, "mesa-nn-*.bin")) + glob.glob(os.path.join(b, "mesa-tp-*.bin")))
if [os.path.basename(f) for f in fa] != [os.path.basename(f) for f in fb]:
    diffs.append("config file sets differ: %d vs %d" % (len(fa), len(fb)))
for f in fa:
    g = os.path.join(b, os.path.basename(f))
    if not os.path.exists(g):
        continue
    n += 1
    x, y = D.decode(f), D.decode(g)
    for k in x:
        if "address" in k or k.startswith("unused"):
            continue
        if x[k] != y.get(k):
            diffs.append("%s %s %d -> %s" % (os.path.basename(f), k, x[k], y.get(k)))
wa = sorted(glob.glob(os.path.join(a, "mesa-compressed-*.bin")))
wb = sorted(glob.glob(os.path.join(b, "mesa-compressed-*.bin")))
if len(wa) != len(wb):
    diffs.append("weight buffer counts differ: %d vs %d" % (len(wa), len(wb)))
for f in wa:
    g = os.path.join(b, os.path.basename(f))
    n += 1
    if not os.path.exists(g) or open(f, "rb").read() != open(g, "rb").read():
        diffs.append("%s weights differ" % os.path.basename(f))
sa = sorted(glob.glob(os.path.join(a, "shim_submit_*.stream")))
sb = sorted(glob.glob(os.path.join(b, "shim_submit_*.stream")))
sw = []
for f, g in zip(sa, sb):
    x, y = open(f, "rb").read(), open(g, "rb").read()
    if len(x) != len(y):
        sw.append("%s length %d vs %d" % (os.path.basename(f), len(x), len(y)))
    else:
        nd = sum(1 for i in range(0, len(x), 4) if x[i:i + 4] != y[i:i + 4])
        sw.append("%s %d/%d words differ" % (os.path.basename(f)[-11:-7], nd, len(x) // 4))
ok = not diffs and len(fa) and wa
print("%-8s %2d configs + %d weight bufs: %s | streams: %d vs %d submits; %s" % (
    name, len(fa), len(wa), "IDENTICAL (addresses aside)" if ok else "; ".join(diffs[:10]) +
    (" (+%d more)" % (len(diffs) - 10) if len(diffs) > 10 else ""), len(sa), len(sb), ", ".join(sw)))
sys.exit(0 if ok else 1)

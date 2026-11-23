#!/usr/bin/env python3
"""compare_prefill.py (2026-11-23): jb_prefill's dumps (--dump DIR: x.<layer>.f32, --out logits of tokens 1..T with
--all) vs export_aux.py's reference (<text>.ref: the Python C-glue path). Per layer the residual's worst relative
difference; for the logits top-1 agreement, KL(ref || C) and the perplexity of both on the text.
Usage: compare_prefill.py REF_FILE PROMPT_FILE DUMP_DIR LOGITS_FILE"""
import os
import struct
import sys

import numpy as np

DT = {0: np.float32, 1: np.float64, 2: np.int32, 3: np.uint16, 4: np.int8}


def read_jba(path):
    out, b = {}, open(path, "rb").read()
    assert b[:4] == b"JBA1"
    o = 4
    while o < len(b):
        name = b[o:o + 48].rstrip(b"\0").decode()
        dt, nd, d0, d1, d2, d3, n = struct.unpack_from("<II4IQ", b, o + 48)
        o += 48 + 32
        out[name] = np.frombuffer(b, DT[dt], n // np.dtype(DT[dt]).itemsize, o).reshape([d0, d1, d2, d3][:nd])
        o += n
    return out


def logsoftmax(z):
    z = z.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


ref, pr = read_jba(sys.argv[1]), read_jba(sys.argv[2])
dump, lgp = sys.argv[3], sys.argv[4]
ids = pr["ids"]
T1 = len(ids)
worst = 0.0
for li in range(1000):
    p = os.path.join(dump, "x.%d.f32" % li)
    if not os.path.exists(p):
        break
    xc = np.fromfile(p, np.float32).reshape(T1, -1)[1:]
    xr = ref["x.%d" % li][1:]
    rel = float(np.abs(xc - xr).max() / np.abs(xr).max())
    worst = max(worst, rel)
    if li < 3 or li % 10 == 9:
        print("layer %2d residual: max |C - ref| / max |ref| = %.2e" % (li, rel))
lc = np.fromfile(lgp, np.float32).reshape(T1 - 1, -1)
lr = ref["logits"][1:]
a, b = logsoftmax(lr), logsoftmax(lc)
kl = float((np.exp(a) * (a - b)).sum(-1).mean())
top1 = float((lr.argmax(-1) == lc.argmax(-1)).mean())
tgt = ids[2:]
ppl = lambda lp: float(np.exp(-lp[np.arange(len(tgt)), tgt].mean()))
print("residual worst over layers %.2e; logits: top-1 agreement %.1f%%, KL(ref || C) %.2e, ppl ref %.3f / C %.3f" % (
    worst, 100 * top1, kl, ppl(a[:-1]), ppl(b[:-1])))

#!/usr/bin/env python3
"""score_prefill.py (2026-11-23): jb_prefill --all logits (tokens 1..T, float32 [T, V]) scored against the full-precision
log-probabilities (llm/ref/fp32_logits.py: [T+1, V], the same 257 tokens): KL(fp32 || C), top-1 agreement, perplexity
of both on tokens 2..T; optionally also against another jb_prefill logits file (e.g. --ref vs --npu).
Usage: score_prefill.py LOGITS FP32.npy IDS.npy [OTHER_LOGITS]"""
import sys

import numpy as np


def logsoftmax(z):
    z = z.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


ref = np.load(sys.argv[2]).astype(np.float64)[1:]          # predictions made at positions 1..T
ids = np.load(sys.argv[3])
T, V = ref.shape
lc = logsoftmax(np.fromfile(sys.argv[1], np.float32).reshape(T, V))
tgt = ids[2:]                                                # position i predicts token i+1


def stats(a, b):
    kl = float((np.exp(a) * (a - b)).sum(-1).mean())
    return kl, float((a.argmax(-1) == b.argmax(-1)).mean())


ppl = lambda lp: float(np.exp(-lp[np.arange(T - 1), tgt].mean()))
kl, top1 = stats(ref, lc)
print("vs fp32: KL %.4f, top-1 %.1f%%, ppl %.3f (fp32 %.3f)" % (kl, 100 * top1, ppl(lc[:-1]), ppl(ref[:-1])))
if len(sys.argv) > 4:
    lo = logsoftmax(np.fromfile(sys.argv[4], np.float32).reshape(T, V))
    kl2, t2 = stats(lo, lc)
    print("vs %s: KL %.4f, top-1 %.1f%%" % (sys.argv[4].split("/")[-2], kl2, 100 * t2))

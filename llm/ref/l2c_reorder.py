#!/usr/bin/env python3
"""l2c_reorder.py (2026-11-19, CPU only): can neuron blocks be made skippable by reordering the FFN neurons?

L2: half the single FFN neurons per token can be dropped almost for free (oracle), but blocks of 16 / 64 cannot,
because the important neurons are scattered. Permuting the FFN neurons (gate/up rows and down columns together) does
not change the model, so we may choose an order where neurons that fire together share a block.

Order: per layer, recursive balanced bisection of the neurons along the main direction of their activity patterns
(which tokens each neuron was in the 95%-energy set for) on CALIBRATION text, down to blocks of B.
Then, on the two EVAL texts: blocks needed for 95% energy, and oracle keep-50% / keep-75% of blocks quality,
original order vs reordered.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smollm import SmolLM, Quant, chat_ids  # noqa: E402
from l2_quality import MD, texts, logsoftmax, score  # noqa: E402

RNG = np.random.default_rng(61119)


def split(idx, F, B):
    if len(idx) <= B:
        return [idx]
    left = (len(idx) // B // 2) * B
    X = F[idx] - F[idx].mean(0)
    v = RNG.normal(size=F.shape[1])
    for _ in range(30):
        v = X.T @ (X @ v)
        v /= np.linalg.norm(v) + 1e-12
    o = np.argsort(X @ v, kind="stable")
    return split(idx[o[:left]], F, B) + split(idx[o[left:]], F, B)


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    m = SmolLM(MD)
    calib, evals = texts(tok)
    lit = "/home/jordan/Documents/AI_Projects/LiteRT-LM/README.md"
    extra = tok.encode(open(lit, errors="replace").read(), add_special_tokens=False).ids[:512]
    calib = calib + [extra, chat_ids(tok, "Give me a recipe for pancakes."),
                     chat_ids(tok, "How do airplanes stay in the air?")]
    colnorm2 = [(d["down"].f32() ** 2).sum(0) for d in m.layers]
    act = [[] for _ in range(m.L)]

    def rec_hook(li, h):
        e = h.astype(np.float64) ** 2 * colnorm2[li]
        srt = np.argsort(-e, -1)
        cum = np.cumsum(np.take_along_axis(e, srt, -1), -1) / e.sum(-1, keepdims=True)
        need = (cum < 0.95).sum(-1) + 1
        a = np.zeros_like(e, bool)
        for t in range(e.shape[0]):
            a[t, srt[t, :need[t]]] = True
        act[li].append(a)
        return h
    for ids in calib:
        m.forward(ids, Quant("fp32"), ffn_hook=rec_hook)
    F = [np.concatenate(a).T.astype(np.float64) for a in act]   # [1536 neurons, calibration tokens]
    print("calibration tokens: %d" % F[0].shape[1])
    always = np.mean([np.mean(f.mean(1) > 0.9) for f in F]) * 100
    print("neurons in the 95%% set for > 90%% of calibration tokens (\"always hot\"): %.1f%% (mean over layers)" % always)
    refs = {tn: logsoftmax(m.forward(ids, Quant("fp32"))) for tn, ids in evals.items()}
    print("\n%-6s %-10s %-7s %9s %12s %12s" % ("block", "order", "text", "need95", "keep50 ppl", "keep75 ppl"))
    for B in (16, 64):
        t = time.monotonic()
        perms = [np.concatenate(split(np.arange(f.shape[0]), f, B)) for f in F]
        for oname, P in (("original", [np.arange(m.cfg["intermediate_size"])] * m.L), ("reordered", perms)):
            for tn, ids in evals.items():
                need = []

                def need_hook(li, h):
                    e = (h.astype(np.float64) ** 2 * colnorm2[li])[:, P[li]].reshape(h.shape[0], -1, B).sum(-1)
                    cum = np.cumsum(-np.sort(-e, -1), -1) / e.sum(-1, keepdims=True)
                    need.append(((cum < 0.95).sum(-1) + 1) / e.shape[1])
                    return h
                m.forward(ids, Quant("fp32"), ffn_hook=need_hook)
                ppl = []
                for keep in (0.5, 0.75):
                    def keep_hook(li, h, keep=keep):
                        e = (h.astype(np.float64) ** 2 * colnorm2[li])[:, P[li]].reshape(h.shape[0], -1, B).sum(-1)
                        k = int(round(keep * e.shape[1]))
                        thr = -np.sort(-e, -1)[:, k - 1:k]
                        mask = np.zeros_like(h, bool)
                        mask[:, P[li]] = np.repeat(e >= thr, B, axis=1)
                        return h * mask
                    ppl.append(score(refs[tn], logsoftmax(m.forward(ids, Quant("fp32"), ffn_hook=keep_hook)),
                                     np.asarray(ids))["ppl"])
                print("%-6d %-10s %-7s %8.1f%% %12.2f %12.2f" % (B, oname, tn, 100 * np.mean(np.concatenate(need)),
                                                                ppl[0], ppl[1]))
        print("  (block %d: %.0f s)" % (B, time.monotonic() - t))
    np.savez("/home/jordan/Documents/AI_Projects/npu-llm/runs/l2c-perms-b64.npz", *perms)


if __name__ == "__main__":
    sys.exit(main())

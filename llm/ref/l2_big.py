#!/usr/bin/env python3
"""l2_big.py (2026-11-19, CPU only): L2 quality check for models too big for l2_quality.py (SmolLM2-1.7B, SmolLM3-3B).

Weights stay memory-mapped (bf16 file, one layer expanded at a time), quantized on the fly (no int8 cache), int32
accumulation in float32. Fused jobs only (qkv, gateup = one NPU job each). Variants: fp32, w8, w8a8 static, w8a8
static with position 0 on the CPU. Big vocabularies (> 256 MB as fp32, SmolLM3) keep the head in full precision
in every variant (noted in the output). Then FFN sparsity on GPL-3: neurons for 95% of the down_proj energy, how
much was active in t-1..t-4, and oracle keep-50%.
Usage: l2_big.py MODEL_DIR [TOKENS]     writes npu-llm/runs/l2-big-<model>-<stamp>.json
"""
import json
import os
import sys
import time

import numpy as np

MD = os.path.abspath(sys.argv[1])
os.environ["LLM_MODEL_DIR"] = MD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smollm import SmolLM, Quant, _w  # noqa: E402
import l2_quality  # noqa: E402
from l2_quality import logsoftmax, score  # noqa: E402

T = int(sys.argv[2]) if len(sys.argv) > 2 else 256
l2_quality.T = T


def calibrate2(m, id_lists):
    """One fp32 pass: ranges over all positions and without position 0."""
    rec, rec0 = {}, {}

    class Rec(Quant):
        def linear(self, key, x, W):
            y = x @ _w(W).T
            for r, a, b in ((rec, x, y), (rec0, x[1:] if self.start == 0 else x, y[1:] if self.start == 0 else y)):
                if a.shape[0] == 0:
                    continue
                for k2, t in ((key + ".in", a), (key + ".out", b)):
                    lo, hi = r.get(k2, (np.inf, -np.inf))
                    r[k2] = (min(lo, float(t.min())), max(hi, float(t.max())))
            return y
    for ids in id_lists:
        m.forward(ids, Rec(fused=True))
    return rec, rec0


def main():
    from tokenizers import Tokenizer
    Quant.CACHE, Quant.F64 = False, False
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    t0 = time.monotonic()
    m = SmolLM(MD)
    big_vocab = m.embed.shape[0] * m.embed.shape[1] * 4 > 256 << 20 and m.embed.shape[0] > 100000
    qh = not big_vocab
    name = os.path.basename(MD)
    print("%s: %d layers, hidden %d, FFN %d, vocab %d; head %s" % (
        name, m.L, m.D, m.cfg["intermediate_size"], m.embed.shape[0], "8-bit like the rest" if qh else
        "kept in full precision in every variant (big vocabulary)"), flush=True)
    calib, evals = l2_quality.texts(tok)
    rec, rec0 = calibrate2(m, calib)
    print("calibrated in %.0f s" % (time.monotonic() - t0), flush=True)
    out = {"model": name, "tokens_per_text": T, "head_quantized": qh, "quality": {}, "sparsity": {}, "oracle": {}}
    variants = [("fp32", Quant("fp32")), ("w8", Quant("w8", True, quant_head=qh)),
                ("w8a8 static fused", Quant("w8a8", True, rec, quant_head=qh)),
                ("w8a8 fused, pos 0 on CPU", Quant("w8a8", True, rec0, quant_head=qh, fp_first=True))]
    ref = {}
    print("\n%-28s %-7s %8s %10s %10s" % ("variant", "text", "ppl", "top1=fp32", "KL"), flush=True)
    for vname, q in variants:
        for tn, ids in evals.items():
            t1 = time.monotonic()
            lp = logsoftmax(m.forward(ids, q))
            if vname == "fp32":
                ref[tn] = lp
            s = score(ref[tn], lp, np.asarray(ids))
            s["seconds"] = round(time.monotonic() - t1)
            out["quality"].setdefault(vname, {})[tn] = s
            print("%-28s %-7s %8.2f %9.1f%% %10.4f   (%d s)" % (vname, tn, s["ppl"], 100 * s["top1_vs_fp32"],
                                                              s["kl_vs_fp32"], s["seconds"]), flush=True)
    colnorm2 = [(d["down"].f32() ** 2).sum(0) for d in m.layers]
    need, cov4 = [], []

    def stat_hook(li, h):
        e = h.astype(np.float64) ** 2 * colnorm2[li]
        srt = np.argsort(-e, -1)
        cum = np.cumsum(np.take_along_axis(e, srt, -1), -1) / e.sum(-1, keepdims=True)
        n95 = (cum < 0.95).sum(-1) + 1
        act = np.zeros_like(e, bool)
        for t in range(e.shape[0]):
            act[t, srt[t, :n95[t]]] = True
        share = e / e.sum(-1, keepdims=True)
        need.append(n95 / e.shape[1])
        for t in range(4, e.shape[0]):
            cov4.append(share[t][act[t] & act[t - 4:t].any(0)].sum() / share[t][act[t]].sum())
        return h
    ids = evals["gpl3"]
    m.forward(ids, Quant("fp32"), ffn_hook=stat_hook)
    out["sparsity"] = {"need95": float(100 * np.mean(np.concatenate(need))), "covered_by_prev4": float(100 * np.mean(cov4))}

    def keep_hook(li, h):
        e = h.astype(np.float64) ** 2 * colnorm2[li]
        k = h.shape[1] // 2
        thr = -np.sort(-e, -1)[:, k - 1:k]
        return h * (e >= thr)
    s = score(ref["gpl3"], logsoftmax(m.forward(ids, Quant("fp32"), ffn_hook=keep_hook)), np.asarray(ids))
    out["oracle"]["keep50"] = s
    print("\nFFN (GPL-3): %.1f%% of neurons carry 95%% of the down_proj energy; %.1f%% of that was active in t-1..t-4;"
          " oracle keep-50%% ppl %.2f (fp32 %.2f), top-1 %.1f%%" % (out["sparsity"]["need95"],
                                                                     out["sparsity"]["covered_by_prev4"], s["ppl"],
                                                                     out["quality"]["fp32"]["gpl3"]["ppl"],
                                                                     100 * s["top1_vs_fp32"]), flush=True)
    path = "/home/jordan/Documents/AI_Projects/npu-llm/runs/l2-big-%s-%s.json" % (name, time.strftime("%Y%m%d-%H%M%S"))
    json.dump(out, open(path, "w"), indent=1)
    print("wrote %s (%.0f min)" % (path, (time.monotonic() - t0) / 60))


if __name__ == "__main__":
    sys.exit(main())

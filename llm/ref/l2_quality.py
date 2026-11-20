#!/usr/bin/env python3
"""l2_quality.py (2026-11-18, CPU only, nothing touches the NPU): LLM engine step L2 on SmolLM2-135M-Instruct.

(a) Does the NPU's 8-bit format keep the model's quality? Teacher-forced on two texts the model was not calibrated
    on: perplexity, top-1 agreement with fp32, mean KL(fp32 || variant). Variants: fp32, w8 (int8 weights only),
    w8a8 static (= a compiled NPU job: one scale per tensor, fixed), fused (q/k/v and gate/up as one job each),
    dynamic (scales re-chosen per call: possible in our own engine), head kept fp32.
(b) FFN sparsity (fp32): neurons per token needed for 90/95/99% of the down_proj output energy, per neuron and in
    blocks of 16 / 64; how much of token t's set was already active in token t-1 / t-1..t-4 (TAGE-style history).
(c) Oracle skipping: keep only the top X% FFN neurons (or 64-blocks) per token, quality vs fp32.
(d) A short greedy chat answer, fp32 vs the NPU format.
Writes a JSON next to the printed table: npu-llm/runs/l2-quality-<stamp>.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smollm import SmolLM, Quant, calibrate, chat_ids, generate  # noqa: E402

AI = "/home/jordan/Documents/AI_Projects"
MD = os.environ.get("LLM_MODEL_DIR", AI + "/npu-llm/models/SmolLM2-135M-Instruct")
T = 256


def texts(tok):
    def chunk(path, skip=0):
        ids = tok.encode(open(path, errors="replace").read(), add_special_tokens=False).ids
        return ids[skip:skip + T]
    calib = [chunk("/usr/share/common-licenses/Apache-2.0", 40),
             chat_ids(tok, "Explain in a few sentences how a refrigerator keeps food cold."),
             chat_ids(tok, "Write a short poem about autumn leaves."),
             chat_ids(tok, "What is the capital of France, and what is it famous for?")]
    evals = {"gpl3": chunk("/usr/share/common-licenses/GPL-3", 200),
             "readme": chunk("/home/jordan/alta-npu/README.md", 0)}
    return calib, evals


def logsoftmax(z):
    z = z.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def score(ref_lp, lp, ids):
    nll = -lp[np.arange(len(ids) - 1), ids[1:]].mean()
    kl = (np.exp(ref_lp) * (ref_lp - lp)).sum(-1).mean()
    top1 = (ref_lp.argmax(-1) == lp.argmax(-1)).mean()
    return {"ppl": float(np.exp(nll)), "top1_vs_fp32": float(top1), "kl_vs_fp32": float(kl)}


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    t0 = time.monotonic()
    m = SmolLM(MD)
    print("loaded in %.1f s" % (time.monotonic() - t0))
    calib, evals = texts(tok)
    out = {"model": os.path.basename(MD), "tokens_per_text": T, "quality": {}, "sparsity": {}, "oracle": {}}
    rec_u, rec_f = calibrate(m, calib, False), calibrate(m, calib, True)
    rec_f0 = calibrate(m, calib, True, skip_first=True)   # L2d: position 0 runs in full precision
    variants = [
        ("fp32", Quant("fp32")),
        ("w8", Quant("w8")),
        ("w8a8 static", Quant("w8a8", False, rec_u)),
        ("w8a8 static fused", Quant("w8a8", True, rec_f)),
        ("w8a8 dynamic fused", Quant("w8a8", True, None)),
        ("w8a8 static fused, head fp32", Quant("w8a8", True, rec_f, quant_head=False)),
        ("w8a8 fused, pos 0 on CPU", Quant("w8a8", True, rec_f0, fp_first=True)),
        ("w8a8 fused, pos 0 + down on CPU", Quant("w8a8", True, rec_f0, fp_first=True,
                                                 a8_layers={"qkv", "o", "gateup", "head"})),
    ]
    ref = {}
    print("\n%-30s %-7s %8s %10s %10s" % ("variant", "text", "ppl", "top1=fp32", "KL"))
    for name, q in variants:
        for tn, ids in evals.items():
            t1 = time.monotonic()
            lp = logsoftmax(m.forward(ids, q))
            if name == "fp32":
                ref[tn] = lp
            s = score(ref[tn], lp, np.asarray(ids))
            s["seconds"] = round(time.monotonic() - t1, 1)
            out["quality"].setdefault(name, {})[tn] = s
            print("%-30s %-7s %8.2f %9.1f%% %10.4f   (%.0f s)" % (name, tn, s["ppl"], 100 * s["top1_vs_fp32"],
                                                                  s["kl_vs_fp32"], s["seconds"]))
    # (b) sparsity statistics on the fp32 run
    colnorm2 = [(d["down"].f32() ** 2).sum(0) for d in m.layers]       # ||W_down[:, i]||^2, [1536]
    stats = {}

    def stat_hook(li, h):
        e = h.astype(np.float64) ** 2 * colnorm2[li]               # energy each neuron sends into down_proj
        for bs in (1, 16, 64):
            eb = e.reshape(e.shape[0], -1, bs).sum(-1)
            srt = -np.sort(-eb, -1)
            cum = np.cumsum(srt, -1) / srt.sum(-1, keepdims=True)
            for p in (0.90, 0.95, 0.99):
                need = (cum < p).sum(-1) + 1
                stats.setdefault((bs, p), []).append(need / eb.shape[1])
            # temporal: energy share of token t's 95% set that was in the 95% set of t-1 / of t-1..t-4
            order = np.argsort(-eb, -1)
            need95 = (cum < 0.95).sum(-1) + 1
            act = np.zeros_like(eb, bool)
            for t in range(eb.shape[0]):
                act[t, order[t, :need95[t]]] = True
            share = eb / eb.sum(-1, keepdims=True)
            for hist in (1, 4):
                cov = []
                for t in range(hist, eb.shape[0]):
                    prev = act[t - hist:t].any(0)
                    cov.append(share[t][act[t] & prev].sum() / share[t][act[t]].sum())
                stats.setdefault((bs, "hist%d" % hist), []).append(np.mean(cov))
        return h
    for ids in evals.values():
        m.forward(ids, Quant("fp32"), ffn_hook=stat_hook)
    print("\nFFN neurons per token needed for X%% of down_proj output energy (mean over layers, tokens, texts)")
    print("%-10s %8s %8s %8s %14s %16s" % ("block", "90%", "95%", "99%", "in t-1 set", "in t-1..t-4"))
    for bs in (1, 16, 64):
        row = [100 * np.mean(np.concatenate(stats[(bs, p)])) for p in (0.90, 0.95, 0.99)]
        h1, h4 = (100 * np.mean(stats[(bs, "hist%d" % k)]) for k in (1, 4))
        out["sparsity"]["block%d" % bs] = {"need90": row[0], "need95": row[1], "need99": row[2],
                                           "covered_by_prev1": h1, "covered_by_prev4": h4}
        print("%-10s %7.1f%% %7.1f%% %7.1f%% %13.1f%% %15.1f%%" % ("%d neuron%s" % (bs, "s" if bs > 1 else ""),
                                                                 row[0], row[1], row[2], h1, h4))
    # (c) oracle skipping: keep the top fraction (by energy) per token, zero the rest
    print("\nOracle skipping (knows the exact activations; a predictor can only approach this)")
    print("%-26s %-7s %8s %10s %10s" % ("kept", "text", "ppl", "top1=fp32", "KL"))
    for bs, keep in ((1, 0.5), (1, 0.3), (1, 0.2), (16, 0.5), (64, 0.5), (64, 0.75)):
        def keep_hook(li, h, bs=bs, keep=keep):
            e = (h.astype(np.float64) ** 2 * colnorm2[li]).reshape(h.shape[0], -1, bs).sum(-1)
            k = max(1, int(round(keep * e.shape[1])))
            thr = -np.sort(-e, -1)[:, k - 1:k]
            maskb = e >= thr
            return h * np.repeat(maskb, bs, axis=1)
        for tn, ids in evals.items():
            lp = logsoftmax(m.forward(ids, Quant("fp32"), ffn_hook=keep_hook))
            s = score(ref[tn], lp, np.asarray(ids))
            out["oracle"].setdefault("block%d keep%d%%" % (bs, keep * 100), {})[tn] = s
            print("%-26s %-7s %8.2f %9.1f%% %10.4f" % ("%d%% (blocks of %d)" % (keep * 100, bs), tn, s["ppl"],
                                                       100 * s["top1_vs_fp32"], s["kl_vs_fp32"]))
    # (d) greedy answers
    q = "What are three tips for staying focused while studying?"
    ids = chat_ids(tok, q)
    out["chat"] = {"prompt": q}
    for name, qq in (("fp32", Quant("fp32")), ("w8a8 fused, pos 0 on CPU", Quant("w8a8", True, rec_f0, fp_first=True))):
        t1 = time.monotonic()
        ans = tok.decode(generate(m, ids, 60, qq))
        out["chat"][name] = ans
        print("\n[%s, %.0f s] %s" % (name, time.monotonic() - t1, ans))
    path = AI + "/npu-llm/runs/l2-quality-%s.json" % time.strftime("%Y%m%d-%H%M%S")
    json.dump(out, open(path, "w"), indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""l2f_hybrid.py (2026-11-19, CPU only): quality of the CPU + NPU split that needs fewer NPU submits per token.

The NPU runs only the big FFN steps in its 8-bit format (int8 activations, static scales, G weight-group jobs); the
CPU runs the small attention steps (qkv, o) with int8 weights, one scale per channel and full-precision activations
(what a CPU kernel does anyway, e.g. llama.cpp Q8_0), and every step of position 0. Head fp32. Fused jobs.
Variants: H0 all on the CPU (w8 per-channel); H1 NPU gateup + down; H2 NPU gateup only (down on the CPU);
each with G = 16 and 4 groups on the NPU steps. Usage: l2f_hybrid.py MODEL_DIR [TOKENS]
"""
import json
import os
import sys
import time

import numpy as np

MD = os.path.abspath(sys.argv[1])
os.environ["LLM_MODEL_DIR"] = MD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smollm import SmolLM, Quant  # noqa: E402
import l2_quality  # noqa: E402
from l2_quality import logsoftmax, score  # noqa: E402
from l2_big import calibrate2  # noqa: E402

T = int(sys.argv[2]) if len(sys.argv) > 2 else 256
l2_quality.T = T
RUNS = "/home/jordan/Documents/AI_Projects/npu-llm/runs"


def main():
    from tokenizers import Tokenizer
    single = os.path.join(MD, "model.safetensors")
    big = not os.path.exists(single) or os.path.getsize(single) > 1 << 30
    Quant.CACHE, Quant.F64 = not big, not big
    t0 = time.monotonic()
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    m = SmolLM(MD)
    name = os.path.basename(MD)
    calib, evals = l2_quality.texts(tok)
    rec, rec0 = calibrate2(m, calib)
    refs = {tn: logsoftmax(m.forward(ids, Quant("fp32"))) for tn, ids in evals.items()}
    print("%s, %d tokens per text, head fp32; calibration + fp32 %.0f s\n%-40s %-7s %8s %10s %10s" % (
        name, T, time.monotonic() - t0, "variant", "text", "ppl", "top1=fp32", "KL"), flush=True)
    out = {"model": name, "tokens_per_text": T, "rows": []}
    for tn, ids in evals.items():
        s = score(refs[tn], refs[tn], np.asarray(ids))
        out["rows"].append({"variant": "fp32", "text": tn, **{k: float(v) for k, v in s.items()}})
        print("%-40s %-7s %8.2f" % ("fp32", tn, s["ppl"]), flush=True)
    rows = [("H0 all CPU (w8 per-channel)", {}, None),
            ("H1 NPU gateup+down, 16 groups", {"gateup": 16, "down": 16}, {"gateup", "down"}),
            ("H1 NPU gateup+down, 4 groups", {"gateup": 4, "down": 4}, {"gateup", "down"}),
            ("H2 NPU gateup, 16 groups", {"gateup": 16}, {"gateup"}),
            ("H2 NPU gateup, 4 groups", {"gateup": 4}, {"gateup"})]
    for label, groups, npu in rows:
        Quant.WGROUPS = 0                      # CPU steps: one scale per channel
        Quant.WGROUPS_BY = dict(groups)
        Quant.WQ.clear()
        q = (Quant("w8", True, quant_head=False) if npu is None else
             Quant("w8a8", True, rec0, quant_head=False, a8_layers=npu, fp_first=True))
        for tn, ids in evals.items():
            t = time.monotonic()
            s = score(refs[tn], logsoftmax(m.forward(ids, q)), np.asarray(ids))
            out["rows"].append({"variant": label, "text": tn, **{k: float(v) for k, v in s.items()}})
            print("%-40s %-7s %8.2f %9.1f%% %10.4f   (%.0f s)" % (
                label, tn, s["ppl"], 100 * s["top1_vs_fp32"], s["kl_vs_fp32"], time.monotonic() - t), flush=True)
    Quant.WQ.clear()
    path = os.path.join(RUNS, "l2f-hybrid-%s-%s.json" % (name, time.strftime("%Y%m%d-%H%M%S")))
    json.dump(out, open(path, "w"), indent=1)
    print("wrote %s (%.0f min)" % (path, (time.monotonic() - t0) / 60), flush=True)


if __name__ == "__main__":
    sys.exit(main())

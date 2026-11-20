#!/usr/bin/env python3
"""l2e_groups.py (2026-11-19, CPU only): weight channel groups for the NPU's 8-bit format.

The NN unit takes one weight scale per job (no per-channel quantization), but a layer can be split into G jobs over
groups of output channels, each with its own scale (channels sorted by max |w| so each group has a narrow range).
G = 0 = one scale per channel (G = OC jobs, the upper bound). Fused jobs (qkv, gateup), position 0 on the CPU for the
8-bit-activation variants (one output range per layer, as calibrated; per-group output scales would be finer still).
The head stays fp32 in EVERY row, so this measures the transformer layers only (and, on SmolLM2-1.7B, shows how much
of its earlier w8 loss, 3.57 vs 2.18 with an 8-bit head, came from the head).
Usage: l2e_groups.py MODEL_DIR [TOKENS]  -> npu-llm/runs/l2e-groups-<model>-<stamp>.json
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
    print("%s, %d tokens per text, head fp32 in every row; calibration + fp32 %.0f s\n%-34s %-7s %8s %10s %10s" % (
        name, T, time.monotonic() - t0, "variant", "text", "ppl", "top1=fp32", "KL"), flush=True)
    out = {"model": name, "tokens_per_text": T, "head_quantized": False, "rows": []}
    for tn, ids in evals.items():
        s = score(refs[tn], refs[tn], np.asarray(ids))
        out["rows"].append({"variant": "fp32", "groups": None, "text": tn, **{k: float(v) for k, v in s.items()}})
        print("%-34s %-7s %8.2f" % ("fp32", tn, s["ppl"]), flush=True)
    rows = [("w8", 1), ("w8", 4), ("w8", 16), ("w8", 0), ("w8a8 pos0 CPU", 16), ("w8a8 pos0 CPU", 0)]
    for kind, g in rows:
        Quant.WGROUPS = g
        Quant.WQ.clear()   # one group setting cached at a time (small models)
        q = (Quant("w8", True, quant_head=False) if kind == "w8" else
             Quant("w8a8", True, rec0, quant_head=False, fp_first=True))
        label = "%s, %s" % (kind, "per-channel" if g == 0 else "%d group%s" % (g, "s" if g > 1 else ""))
        for tn, ids in evals.items():
            t = time.monotonic()
            s = score(refs[tn], logsoftmax(m.forward(ids, q)), np.asarray(ids))
            out["rows"].append({"variant": kind, "groups": g, "text": tn, **{k: float(v) for k, v in s.items()}})
            print("%-34s %-7s %8.2f %9.1f%% %10.4f   (%.0f s)" % (
                label, tn, s["ppl"], 100 * s["top1_vs_fp32"], s["kl_vs_fp32"], time.monotonic() - t), flush=True)
    Quant.WQ.clear()
    path = os.path.join(RUNS, "l2e-groups-%s-%s.json" % (name, time.strftime("%Y%m%d-%H%M%S")))
    json.dump(out, open(path, "w"), indent=1)
    print("wrote %s (%.0f min)" % (path, (time.monotonic() - t0) / 60), flush=True)


if __name__ == "__main__":
    sys.exit(main())

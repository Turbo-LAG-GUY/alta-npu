#!/usr/bin/env python3
"""l2h_groups.py (2026-11-21, CPU only): how few jobs per layer step can the fixed 8-bit format use? Wider NPU jobs use
the NPU much better (width ladder 2026-11-21: 3x3 at 960x540, 8 channels 97 GOPS -> 64 channels 603 GOPS), and each
weight group is one job (its own weight scale and output scale), so fewer groups = wider jobs.
Same setup as l2g_outliers.py row RB: rotation (residual Q folded into the weights, per-head Hadamard into v / o,
block Hadamard on the down input), per-job output scales, jobs grouped by max |w|, every step 8-bit on the NPU,
position 0 on the CPU, head fp32. Rows: 4, 2 and 1 jobs per step (1 = one weight scale and one output scale per step).
Usage: l2h_groups.py MODEL_DIR [TOKENS]     writes npu-llm/runs/l2h-groups-<model>-<stamp>.json
"""
import json
import os
import sys
import time

import numpy as np

MD = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l2g_outliers as G  # noqa: E402  (reads MODEL_DIR and TOKENS from argv)
import l2_quality  # noqa: E402
from l2_quality import logsoftmax, score  # noqa: E402
from smollm import SmolLM, Quant  # noqa: E402

RUNS = "/home/jordan/Documents/AI_Projects/npu-llm/runs"
GROUPS = (4, 2, 1)


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
    refs = {tn: logsoftmax(m.forward(ids, Quant("fp32"))) for tn, ids in evals.items()}
    what = G.rotate(m, m.layers)
    rec, chin, chout = G.calibrate3(m, calib)
    print("%s, %d tokens per text; %s; calibration + fp32 %.0f s" % (name, G.T, what, time.monotonic() - t0),
          flush=True)
    print("%-36s %-7s %8s %10s %10s" % ("jobs per step (rotation + per-job out)", "text", "ppl", "top1=fp32", "KL"),
          flush=True)
    out = {"model": name, "tokens_per_text": G.T, "rotation": what, "rows": []}
    for tn, ids in evals.items():
        s = score(refs[tn], refs[tn], np.asarray(ids))
        out["rows"].append({"jobs": None, "text": tn, **{k: float(v) for k, v in s.items()}})
        print("%-36s %-7s %8.2f" % ("fp32", tn, s["ppl"]), flush=True)
    for g in GROUPS:
        Quant.WGROUPS, Quant.WGROUPS_BY = g, None
        Quant.WQ.clear()
        G.QuantG.GRP.clear()
        q = G.QuantG("w8a8", True, rec, chout=chout, per_job_out=True, quant_head=False, fp_first=True)
        for tn, ids in evals.items():
            t = time.monotonic()
            s = score(refs[tn], logsoftmax(m.forward(ids, q)), np.asarray(ids))
            out["rows"].append({"jobs": g, "text": tn, **{k: float(v) for k, v in s.items()}})
            print("%-36s %-7s %8.2f %9.1f%% %10.4f   (%.0f s)" % (
                "%d job%s per step" % (g, "s" if g > 1 else ""), tn, s["ppl"], 100 * s["top1_vs_fp32"],
                s["kl_vs_fp32"], time.monotonic() - t), flush=True)
    Quant.WQ.clear()
    path = os.path.join(RUNS, "l2h-groups-%s-%s.json" % (name, time.strftime("%Y%m%d-%H%M%S")))
    json.dump(out, open(path, "w"), indent=1)
    print("wrote %s (%.0f min)" % (path, (time.monotonic() - t0) / 60), flush=True)


if __name__ == "__main__":
    sys.exit(main())

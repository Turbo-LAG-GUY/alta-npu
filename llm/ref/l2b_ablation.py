#!/usr/bin/env python3
"""l2b_ablation.py (2026-11-19, CPU only): which layers' 8-bit ACTIVATIONS cause L2's quality loss?
SmolLM2-135M-Instruct, GPL-3 text (256 tokens), static per-tensor scales from the L2 calibration set.
Each row: int8 weights everywhere (w8), plus 8-bit activations only where named (side: in / out / both).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smollm import SmolLM, Quant, calibrate  # noqa: E402
from l2_quality import MD, texts, logsoftmax, score  # noqa: E402


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    m = SmolLM(MD)
    calib, evals = texts(tok)
    ids = evals["gpl3"]
    rec = calibrate(m, calib, False)
    ref = logsoftmax(m.forward(ids, Quant("fp32")))
    rows = [("w8 (no 8-bit activations)", Quant("w8"))]
    for n in ("q", "k", "v", "o", "gate", "up", "down", "head"):
        rows.append(("a8 %s both" % n, Quant("w8a8", False, rec, a8_layers={n})))
    for n in ("down", "o", "up", "gate"):
        for side in ("in", "out"):
            rows.append(("a8 %s %s" % (n, side), Quant("w8a8", False, rec, a8_layers={n}, a8_side=side)))
    rows.append(("a8 all but down", Quant("w8a8", False, rec, a8_layers={"q", "k", "v", "o", "gate", "up", "head"})))
    print("%-30s %8s %10s %10s" % ("variant (GPL-3)", "ppl", "top1=fp32", "KL"))
    s = score(ref, ref, np.asarray(ids))
    print("%-30s %8.2f %9.1f%% %10.4f" % ("fp32", s["ppl"], 100, 0))
    for name, q in rows:
        t = time.monotonic()
        s = score(ref, logsoftmax(m.forward(ids, q)), np.asarray(ids))
        print("%-30s %8.2f %9.1f%% %10.4f   (%.0f s)" % (name, s["ppl"], 100 * s["top1_vs_fp32"], s["kl_vs_fp32"],
                                                        time.monotonic() - t))
    # the calibrated ranges themselves: how wide is each kind of activation (max |value|, median over layers)
    print("\ncalibrated range, median over the layers (max |x| of the input / output)")
    for n in ("q", "k", "v", "o", "gate", "up", "down"):
        ins = [max(abs(rec["%d.%s.in" % (i, n)][0]), abs(rec["%d.%s.in" % (i, n)][1])) for i in range(m.L)]
        outs = [max(abs(rec["%d.%s.out" % (i, n)][0]), abs(rec["%d.%s.out" % (i, n)][1])) for i in range(m.L)]
        print("  %-5s in %8.2f (max %8.2f, layer %2d)   out %8.2f (max %8.2f)" % (
            n, np.median(ins), max(ins), int(np.argmax(ins)), np.median(outs), max(outs)))


if __name__ == "__main__":
    sys.exit(main())

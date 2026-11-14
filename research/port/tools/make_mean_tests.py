#!/usr/bin/env python3
"""make_mean_tests.py (step 6e, CPU only): the MEAN test models for Teflon and their CPU references (XNNPACK, the
interpreter's default path, which is also what runs MEAN beside Teflon when the switch is off).

Writes into npu-shader/models/mean/: NAME.tflite and NAME_reference.npz ("input" without the batch dimension, as
teflon_check.py expects, and "output").
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reduce_models as R  # noqa: E402

OUT = os.path.join(HERE, "mean")
MODELS = [
    # name, shape, axes, int8, in_q, out_q, conv in front
    ("mean_ch384_u8", (1, 1, 64, 384), [3], False, (0.05, 120), (0.03, 128), True),   # layernorm-like: 64 tokens x 384
    ("mean_ch768_i8", (1, 1, 32, 768), [3], True, (0.07, 5), (0.02, -3), True),       # 32 tokens x 768, int8
    ("mean_sp7x7_u8", (1, 7, 7, 1024), [1, 2], False, (0.0235, 0), (0.02, 3), True),  # Keras global pooling
    ("mean_ch64_in_u8", (1, 4, 4, 64), [3], False, (0.1, 3), (0.08, 40), False),     # on the graph input
    ("mean_ch64_in_i8", (1, 4, 4, 64), [3], True, (0.1, -3), (0.08, -40), False),    # int8 on the graph input
]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(613)
    for name, shape, axes, int8, iq, oq, conv in MODELS:
        path = os.path.join(OUT, name + ".tflite")
        R.build_mean(path, shape, axes, int8, iq, oq, conv)
        lo, hi = (-128, 127) if int8 else (0, 255)
        x = rng.integers(lo, hi + 1, size=shape).astype(np.int8 if int8 else np.uint8)
        y = R.cpu_run(path, x, default_delegates=True)
        np.savez(os.path.join(OUT, name + "_reference.npz"), input=x[0], output=y[0])
        print("%-16s %s -> %s, output range %d..%d" % (name, list(shape), list(y.shape), int(y.min()), int(y.max())))


if __name__ == "__main__":
    main()

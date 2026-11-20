#!/usr/bin/env python3
"""llm_shapes.py (2026-11-18, CPU only, nothing here touches the NPU): single int8 1x1 CONV_2D models with the
matrix shapes of real small LLMs, for the LLM engine's first question: does the NN unit take LLM-sized matrix
multiplies when they are written as 1x1 convolutions (FULLY_CONNECTED hangs this NPU, incidents/2026-11-15-*)?

A linear layer y[M, OC] = x[M, IC] @ W[OC, IC]^T + b becomes a 1x1 conv over an image of M pixels: input
[1, 1, M, IC] (H = 1, W = M tokens, as Teflon sees [1, T, C]), weights [OC, 1, 1, IC], output [1, 1, M, OC].
M = 1 is one generated token, M > 1 is a prompt chunk or a batch of draft tokens to verify.

Per-tensor int8 only (this NPU has no per-channel quantization: hwdb NN_PER_CHANNEL_QUANT = 0). Weights are
LLM-like (rounded normal, std 24 of 127, clipped) rather than uniform, so the NN weight compression (zero run
length, NN_ZRL_BITS = 5) sees realistic bytes. Output scale from the float result's range.

Writes MODELS_DIR/NAME.tflite + NAME_reference.npz ("input" without batch, "output" = the CPU interpreter's output,
XNNPACK), the format teflon_check.py and research/port/tools/teflon_offline.sh expect.
Usage: llm_shapes.py [MODELS_DIR] [NAME_FILTER]
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/jordan/Documents/AI_Projects/npu-shader/models")
from int8nn_models import Graph, add_conv, qrange, cpu_run  # noqa: E402  (proven int8 builder, step 6i)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/jordan/Documents/AI_Projects/npu-llm/models/shapes"
FILTER = sys.argv[2] if len(sys.argv) > 2 else ""

IN_Q = (0.05, -3)  # int8 input scale, zero point

# (model, layer, IC, OC). SmolLM2-135M: hidden 576, 9 q heads x 64, 3 kv heads x 64, FFN 1536.
# Qwen3-0.6B: hidden 1024, 16 q heads x 128, 8 kv heads x 128, FFN 3072. "qkv"/"gateup" = the fused projections.
LAYERS = [
    ("smol", "q", 576, 576),
    ("smol", "kv", 576, 192),
    ("smol", "qkv", 576, 960),
    ("smol", "gateup", 576, 3072),
    ("smol", "down", 1536, 576),
    ("qwen", "qkv", 1024, 4096),
    ("qwen", "o", 2048, 1024),
    ("qwen", "gateup", 1024, 6144),
    ("qwen", "down", 3072, 1024),
]
TOKENS = [1, 16]


def build(path, ic, oc, m, rng):
    w = np.clip(np.rint(rng.normal(0, 24, size=(oc, 1, 1, ic))), -127, 127).astype(np.int8)
    ws = 0.02 / np.sqrt(ic / 64.0)  # keeps the outputs in a similar range for every IC
    bias = rng.integers(-500, 501, size=oc).astype(np.int32)
    x = np.clip(np.rint(rng.normal(IN_Q[1], 30, size=(1, m, ic))), -128, 127).astype(np.int8)
    xr = IN_Q[0] * (x.astype(np.float64) - IN_Q[1])
    yr = xr.reshape(m, ic) @ (w.reshape(oc, ic).astype(np.float64) * ws).T + bias * IN_Q[0] * ws
    yq = qrange(yr, True)
    g = Graph()
    S = g.S
    xt = g.tensor("input", (1, 1, m, ic), S.TensorType.INT8, IN_Q[0], IN_Q[1])
    out = add_conv(g, xt, IN_Q, (1, 1, m, ic), oc, 1, 1, False, False, w, ws, 0, bias, yq, True, "linear")
    g.save(path, [xt], [out])
    return x


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(61118)
    for model, layer, ic, oc in LAYERS:
        for m in TOKENS:
            name = "%s_%s_%dx%d_m%d" % (model, layer, ic, oc, m)
            if FILTER and FILTER not in name:
                continue
            path = os.path.join(OUT, name + ".tflite")
            x = build(path, ic, oc, m, rng)
            y = cpu_run(path, x)
            np.savez(os.path.join(OUT, name + "_reference.npz"), input=x, output=y)
            rails = np.mean((y == -128) | (y == 127)) * 100
            print("%-28s weights %5.2f MB  out %s range %d..%d (%.1f%% at the rails)"
                  % (name, ic * oc / 1e6, list(y.shape), y.min(), y.max(), rails))


if __name__ == "__main__":
    sys.exit(main())

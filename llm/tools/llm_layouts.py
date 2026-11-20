#!/usr/bin/env python3
"""llm_layouts.py (2026-11-19, CPU only, nothing here touches the NPU): LLM engine step L1b models.

L1 found that 16 tokens in the [1,1,M,IC] layout cost 13-20x one token (4-8 GOPS). These models ask which layout
lets the NN unit reuse each weight across many tokens. Same weights and tokens for every layout of one shape:
  row     [1,1,16,IC]            the L1 layout (control)
  rowP    same, planar           explicit TRANSPOSEs at both ends: Teflon skips its TP jobs (npu-bench/npu_planar)
  sq      [1,4,4,IC]             tokens as a 2D image
  sqP     same, planar
  col     [1,16,1,IC]            tokens along H (W = 1, the v7 tiling fix case)
  row64   [1,1,64,IC]            64 tokens: does the per-token cost fall with more tokens?
  swap    image = the WEIGHTS [1, OC/64, 64, IC], kernels = the 16 tokens [16,1,1,IC] -> [1, OC/64, 64, 16]
          (y[o,t] = sum_i W[o,i] x[t,i], the same math; the NN streams the weights as a big image, which is the
          shape of work it does at 790 GOPS; the engine would re-encode the 16 token kernels each step)
  swapP   same, planar (otherwise Teflon transposes the whole weight image on every call)
Per-tensor int8 everywhere; kernels symmetric (zero point 0, TFLite's int8 rule), so tokens are quantized
symmetrically in swap. Writes MODELS_DIR/NAME.tflite + NAME_reference.npz (CPU interpreter = XNNPACK).
Usage: llm_layouts.py [MODELS_DIR]
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/jordan/Documents/AI_Projects/npu-shader/models")
sys.path.insert(0, "/home/jordan/Documents/AI_Projects/npu-bench")
from int8nn_models import Graph, add_conv, qrange, cpu_run  # noqa: E402
from npu_planar import make_planar  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/jordan/Documents/AI_Projects/npu-llm/models/layouts"
SHAPES = [("smol_gateup", 576, 3072), ("qwen_gateup", 1024, 6144), ("qwen_down", 3072, 1024)]
XQ = 0.05  # token activation scale (symmetric)


def conv_model(path, img_shape, img_q, wq, ws, yq):
    g = Graph()
    S = g.S
    x = g.tensor("input", img_shape, S.TensorType.INT8, img_q[0], img_q[1])
    oc = wq.shape[0]
    out = add_conv(g, x, img_q, img_shape, oc, 1, 1, False, False, wq, ws, 0, np.zeros(oc, np.int32), yq, True,
                   "linear")
    g.save(path, [x], [out])


def save(name, path, x):
    y = cpu_run(path, x)
    np.savez(os.path.join(OUT, name + "_reference.npz"), input=x, output=y)
    return y


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(61119)
    for sname, ic, oc in SHAPES:
        W = np.clip(np.rint(rng.normal(0, 24, size=(oc, ic))), -127, 127).astype(np.int8)
        ws = 0.02 / np.sqrt(ic / 64.0)
        X64 = np.clip(np.rint(rng.normal(0, 30, size=(64, ic))), -127, 127).astype(np.int8)  # 64 tokens
        Yr = (X64.astype(np.float64) * XQ) @ (W.astype(np.float64) * ws).T                  # [64, OC]
        yq = qrange(Yr, True)
        w4 = W.reshape(oc, 1, 1, ic)
        for lay, shape, ntok in (("row", (1, 1, 16, ic), 16), ("sq", (1, 4, 4, ic), 16),
                                 ("col", (1, 16, 1, ic), 16), ("row64", (1, 1, 64, ic), 64)):
            name = "%s_%s" % (sname, lay)
            p = os.path.join(OUT, name + ".tflite")
            conv_model(p, shape, (XQ, 0), w4, ws, yq)
            x = X64[:ntok].reshape(shape[1:])
            y = save(name, p, x)
            if lay in ("row", "sq"):
                pp = os.path.join(OUT, name + "P.tflite")
                make_planar(p, pp)
                save(name + "P", pp, np.ascontiguousarray(np.moveaxis(x, -1, 0)))  # (C, H, W)
            ok = np.abs(y.reshape(ntok, oc).astype(int) -
                        np.clip(np.round(Yr[:ntok] / yq[0]) + yq[1], -128, 127)).max()
            print("%-22s in %-18s out %-18s max |CPU - numpy| %d" % (name, shape, list(y.shape), ok))
        # swapped roles: the weights are the image, 16 tokens are the kernels
        hh = oc // 64
        img = W.reshape(hh, 64, ic)
        name = "%s_swap" % sname
        p = os.path.join(OUT, name + ".tflite")
        conv_model(p, (1, hh, 64, ic), (ws, 0), X64[:16].reshape(16, 1, 1, ic), XQ, yq)
        y = save(name, p, img)
        ok = np.abs(y.reshape(oc, 16).T.astype(int) -
                    np.clip(np.round(Yr[:16] / yq[0]) + yq[1], -128, 127)).max()
        print("%-22s in %-18s out %-18s max |CPU - numpy| %d" % (name, (1, hh, 64, ic), list(y.shape), ok))
        pp = os.path.join(OUT, name + "P.tflite")
        make_planar(p, pp)
        save(name + "P", pp, np.ascontiguousarray(np.moveaxis(img, -1, 0)))


if __name__ == "__main__":
    sys.exit(main())

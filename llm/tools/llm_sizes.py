#!/usr/bin/env python3
"""llm_sizes.py (2026-11-19, CPU only, nothing here touches the NPU): LLM engine step L1c models.

Cost of one NN job vs the number of tokens M when the tokens form a 2D image (L1b: 16 tokens as 4x4 cost 2-4x one
token; a 1x16 row cost 11-16x). Sizes (H x W = M): 1x1, 1x2, 2x2, 2x4, 4x4, 4x8, 8x8, 16x16 -> M = 1..256.
Small M decides speculative decoding (L6a: 3 guesses = a 4-position check must cost < 1.24x one step), large M
decides prompt reading. Shapes: SmolLM2-135M's four linear layers (qkv 576->960, o 576->576, gateup 576->3072,
down 1536->576) and Qwen3-0.6B gateup 1024->6144. All planar (explicit TRANSPOSE at both ends: no Teflon TP jobs,
as the engine would feed the NPU), per-tensor int8, same weights for every size of a shape.
Writes MODELS_DIR/NAME_P.tflite + NAME_P_reference.npz (input channels-first, CPU interpreter = XNNPACK).
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/jordan/alta-npu/llm/tools")
from llm_layouts import conv_model, XQ  # noqa: E402
sys.path.insert(0, "/home/jordan/Documents/AI_Projects/npu-shader/models")
sys.path.insert(0, "/home/jordan/Documents/AI_Projects/npu-bench")
from int8nn_models import qrange, cpu_run  # noqa: E402
from npu_planar import make_planar  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/jordan/Documents/AI_Projects/npu-llm/models/sizes"
SHAPES = [("smol_qkv", 576, 960), ("smol_o", 576, 576), ("smol_gateup", 576, 3072), ("smol_down", 1536, 576),
          ("qwen_gateup", 1024, 6144)]
SIZES = [(1, 1), (1, 2), (2, 2), (2, 4), (4, 4), (4, 8), (8, 8), (16, 16)]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(61120)
    for sname, ic, oc in SHAPES:
        W = np.clip(np.rint(rng.normal(0, 24, size=(oc, ic))), -127, 127).astype(np.int8)
        ws = 0.02 / np.sqrt(ic / 64.0)
        X = np.clip(np.rint(rng.normal(0, 30, size=(256, ic))), -127, 127).astype(np.int8)
        Yr = (X.astype(np.float64) * XQ) @ (W.astype(np.float64) * ws).T
        yq = qrange(Yr, True)
        for h, w in SIZES:
            m = h * w
            name = "%s_%dx%d" % (sname, h, w)
            tmp = os.path.join(OUT, name + ".tflite")
            conv_model(tmp, (1, h, w, ic), (XQ, 0), W.reshape(oc, 1, 1, ic), ws, yq)
            p = os.path.join(OUT, name + "_P.tflite")
            make_planar(tmp, p)
            os.remove(tmp)
            x = X[:m].reshape(h, w, ic)
            xp = np.ascontiguousarray(np.moveaxis(x, -1, 0))
            y = cpu_run(p, xp)                                   # (OC, H, W)
            np.savez(os.path.join(OUT, name + "_P_reference.npz"), input=xp, output=y)
            want = np.clip(np.round(Yr[:m] / yq[0]) + yq[1], -128, 127).reshape(h, w, oc)
            d = np.abs(np.moveaxis(y, 0, -1).astype(int) - want).max()
            print("%-22s M=%-3d out %-16s max |CPU - numpy| %d" % (name + "_P", m, list(y.shape), d))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""real_packs.py (2026-11-20, LLM engine R2, CPU only): NPU job packs from the REAL weights of a SmolLM model, in the
8-bit format that the simulator found nearly lossless on all four models (llm/ref/l2g_outliers.py, row RB4):
the residual stream rotated (Q folded into the weights), the attention output rotated per head (folded into v / o),
the down_proj input rotated by a block Hadamard transform (the CPU applies it between the NPU steps), every layer
step in 4 weight-group jobs (groups by max |w|, own weight scale each) with per-job output scales, one static input
scale per step (calibration, position 0 left out: position 0 runs on the CPU), fused qkv and gateup.

One pack = one layer step = one NPU submit (jb_pack.py format JBP1): a planar input [1, IC, H, W] (the tokens as an
H x W image), 4 convolution jobs with zero bias, 4 planar outputs. The test input is the step's real 8-bit input
from a forward pass over 257 tokens of GPL-3 (position 0 on the CPU, tokens 1..256 on the NPU); the expected outputs
come from the int8 reference that matches XNNPACK exactly (jb_pack.ref_conv arithmetic, float64 accumulation, exact
for these sizes) and are checked against the levels the simulator itself produced.
Sizes: 1 token (decode) and 256 tokens as 16 x 16 (prompt chunk).

Usage: real_packs.py MODEL_DIR OUT_DIR       writes OUT_DIR/L<layer>_<step>_t<tokens>.jbp, manifest.json, groups.npz
"""
import json
import os
import sys
import time

import numpy as np

MD = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], MD]                       # l2g_outliers reads the model directory from argv at import
sys.path.insert(0, os.path.join(HERE, "..", "ref"))
sys.path.insert(0, HERE)
import l2g_outliers as G  # noqa: E402
import l2_quality  # noqa: E402
from smollm import SmolLM, Quant  # noqa: E402
from jb_pack import Pack, read_pack, run_graph  # noqa: E402

GROUPS = 4
SIZES = [(1, 1, 1), (256, 16, 16)]                 # tokens, H, W
STEPS = ("qkv", "o", "gateup", "down")


class Capture(G.QuantG):
    """The simulator's NPU format (QuantG, per-job output scales), recording each step's NPU input (positions 1..)
    and the output levels it produced."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cap = {}

    def linear(self, key, x, W):
        y = super().linear(key, x, W)
        if key == "head" or self.mode == "fp32":
            return y
        xs, ys = (x[1:], y[1:]) if (self.fp_first and self.start == 0) else (x, y)
        sx, zx = self.qparams(*self.rng(key + ".in", xs))
        (Wq, sw), gid = self.wgroups(key, W)
        ng = int(gid.max()) + 1
        glo, ghi = np.zeros(ng), np.zeros(ng)
        np.minimum.at(glo, gid, self.chout[0][key])
        np.maximum.at(ghi, gid, self.chout[1][key])
        sy = np.maximum(ghi - glo, 1e-8) / 255.0
        zy = np.clip(np.round(-glo / sy) - 128, -128, 127)
        levels = np.clip(np.round(ys / sy[gid]) + zy[gid], -128, 127)   # the simulator's output levels
        self.cap[key] = dict(xq=self.q8(xs, sx, zx).astype(np.int8), sx=sx, zx=zx, Wq=Wq, sw=sw, gid=gid,
                             sy=sy, zy=zy, levels=levels.astype(np.int8))
        return y


def conv_levels(xq, sx, zx, w, sw, sy, zy):
    """jb_pack.ref_conv arithmetic with float64 accumulation (exact: |acc| < 2^53)."""
    acc = (xq.astype(np.float64) - zx) @ w.astype(np.float64).T
    return np.clip(np.round(acc * (sx * sw / sy)) + zy, -128, 127).astype(np.int8)


def main():
    from tokenizers import Tokenizer
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)
    Quant.CACHE, Quant.F64 = True, True
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    m = SmolLM(MD)
    l2_quality.T = 256
    calib, _ = l2_quality.texts(tok)               # the same calibration texts as the l2g runs
    l2_quality.T = 257
    _, evals = l2_quality.texts(tok)
    ids = evals["gpl3"]
    assert len(ids) == 257
    what = G.rotate(m, m.layers)
    rec, chin, chout = G.calibrate3(m, calib)
    Quant.WGROUPS, Quant.WGROUPS_BY = GROUPS, None
    Quant.WQ.clear()
    G.QuantG.GRP.clear()
    cap = Capture("w8a8", True, rec, chout=chout, per_job_out=True, quant_head=False, fp_first=True)
    m.forward(ids, cap)
    print("%s: %s; captured %d steps from 257 GPL-3 tokens (%.0f s)" % (os.path.basename(MD), what, len(cap.cap),
                                                                         time.monotonic() - t0), flush=True)
    manifest = {"model": os.path.basename(MD), "rotation": what, "groups": GROUPS, "text": "gpl3 tokens 0..256",
                "packs": []}
    perm = {}
    worst_sim = 0
    for li in range(m.L):
        for st in STEPS:
            key = "%d.%s" % (li, st)
            c = cap.cap[key]
            OC, IC = c["Wq"].shape
            idx = [np.sort(np.nonzero(c["gid"] == g)[0]) for g in range(GROUPS)]
            perm[key] = c["gid"]
            for T, H, W in SIZES:
                xq = c["xq"][:T]                           # [T, IC], tokens 1..T
                p = Pack()
                xi = p.planar_in(H, W, IC, (c["sx"], c["zx"]))
                exp, outs = [], []
                for g in range(GROUPS):
                    w = c["Wq"][idx[g]]
                    sw = float(c["sw"][idx[g][0]])
                    sy, zy = float(c["sy"][g]), int(c["zy"][g])
                    n = len(idx[g])
                    wi = p.tensor([n, 1, 1, IC], (sw, 0), data=w)
                    bi = p.tensor([1, 1, 1, n], (c["sx"] * sw, 0), data=np.zeros(n, np.int32), size=4)
                    yi = p.tensor([1, H, W, n], (sy, zy))
                    p.conv(xi, wi, bi, yi)
                    p.planar_out(yi, H, W, n, (sy, zy))
                    lv = conv_levels(xq, c["sx"], c["zx"], w, sw, sy, zy)          # [T, n]
                    d = int(np.abs(lv.astype(int) - c["levels"][:T][:, idx[g]].astype(int)).max())
                    worst_sim = max(worst_sim, d)
                    exp.append(np.ascontiguousarray(lv.T).reshape(-1))            # planar: [n, H, W]
                    outs.append({"rows": n, "scale": sy, "zero_point": zy, "weight_scale": sw})
                name = "L%02d_%s_t%d.jbp" % (li, st, T)
                path = os.path.join(OUT, name)
                p.write(path, np.ascontiguousarray(xq.T).reshape(-1), exp)
                manifest["packs"].append({"file": name, "layer": li, "step": st, "tokens": T, "image": [H, W],
                                          "ic": IC, "oc": OC, "input_scale": c["sx"], "input_zero_point": c["zx"],
                                          "jobs": outs})
    # read back and run the pack's own graph: layer 0 at both sizes and every one-token pack
    worst_pack = 0
    for e in manifest["packs"]:
        if e["layer"] != 0 and e["tokens"] != 1:
            continue
        Tn, ops, x, exp, _ = read_pack(os.path.join(OUT, e["file"]))
        got = run_graph(Tn, ops, x)
        worst_pack = max(worst_pack, max(int(np.abs(a.astype(int) - np.frombuffer(b, np.int8).astype(int)).max())
                                         for a, b in zip(got, exp)))
    manifest["check"] = {"expected_vs_simulator_levels_max": worst_sim, "pack_graph_vs_expected_max": worst_pack}
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "groups.npz"), **perm)
    print("%d packs in %s; expected vs the simulator's levels max %d, pack graph vs expected max %d (%.0f s)" % (
        len(manifest["packs"]), OUT, worst_sim, worst_pack, time.monotonic() - t0), flush=True)
    return 0 if worst_sim <= 1 and worst_pack == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

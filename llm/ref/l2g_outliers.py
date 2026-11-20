#!/usr/bin/env python3
"""l2g_outliers.py (2026-11-20, CPU only): fixes for the NPU's 8-bit activation outliers (SmolLM2-1.7B breaks, the
smaller models lose 5-25%), all free at run time in our own engine.

Target: EVERY layer step on the NPU in its 8-bit format (what M > 1 work - prompt chunks, drafted-token checks - needs):
int8 weights in G jobs per step (16 unless noted), int8 input with ONE static scale per step (all jobs read the same
input tensor), int8 outputs, position 0 on the CPU, fused qkv / gateup jobs, head fp32.
Rows:
  A   baseline: one output scale per step, groups by max |w| (= L2e "w8a8 pos0 CPU, 16 groups")
  B   per-job output scales: the same jobs as A (groups by max |w|), each with its own output scale and zero point
      from the calibrated per-channel ranges (an NPU job writes its own output tensor anyway). Grouping by output
      range instead was worse (smoke test 2026-11-20: it mixes small and large weights in one job), and extra
      headroom on the per-job ranges did not help.
  S   SmoothQuant (alpha 0.5) folded into the RMSNorm weights (qkv, gateup) and into the up_proj rows (down), + B
  R   rotation (QuaRot-style, exact in fp32): the residual stream rotated by an orthogonal Q folded into every weight
      that reads or writes it (embedding lookup, qkv, o, gateup, down; the norm weights folded in, the head reads the
      rotated-back residual), the attention output rotated per head (folded into v and o), the down_proj input
      rotated by a block Hadamard transform (online on the CPU, which computes SiLU x up anyway: a fast
      Walsh-Hadamard transform, ~20 us per token); one output scale per step
  RB  R + B;   RB4  R + B with 4 groups per step
Also printed: per-step input outlier ratio (largest channel max / median channel max) before and after each
transform, and the rotated model in fp32 (must equal fp32).
Usage: l2g_outliers.py MODEL_DIR [TOKENS]     writes npu-llm/runs/l2g-outliers-<model>-<stamp>.json
"""
import json
import os
import sys
import time

import numpy as np

MD = os.path.abspath(sys.argv[1])
os.environ["LLM_MODEL_DIR"] = MD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smollm import SmolLM, Quant, _w  # noqa: E402
import l2_quality  # noqa: E402
from l2_quality import logsoftmax, score  # noqa: E402

T = int(sys.argv[2]) if len(sys.argv) > 2 else 256
l2_quality.T = T
RUNS = "/home/jordan/Documents/AI_Projects/npu-llm/runs"
STEPS = ("qkv", "o", "gateup", "down")


def fwht(x, axis=-1):
    """Normalized fast Walsh-Hadamard transform along one axis (length a power of 2). H is symmetric, H @ H = I."""
    x = np.moveaxis(np.asarray(x, np.float32), axis, -1)
    shp, n = x.shape, x.shape[-1]
    y, h = x.reshape(-1, n), 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = np.stack([a + b, a - b], axis=2)
        h *= 2
    return np.moveaxis(y.reshape(shp) * np.float32(1 / np.sqrt(n)), -1, axis)


class Orth:
    """Orthogonal n x n Q: randomized Hadamard diag(s) H when n is a power of 2, else a dense random orthogonal matrix
    (folded into the weights offline, so its cost does not matter). right(A) = A @ Q on A's last axis,
    left_t(A) = Q.T @ A on its first axis, back(A) = A @ Q.T."""

    def __init__(self, n, rng):
        self.pow2 = n & (n - 1) == 0
        if self.pow2:
            self.s = rng.choice(np.array([-1, 1], np.float32), n)
        else:
            q, r = np.linalg.qr(rng.standard_normal((n, n)))
            self.Q = (q * np.sign(np.diag(r))).astype(np.float32)

    def right(self, A):
        return fwht(A * self.s) if self.pow2 else (A @ self.Q).astype(np.float32)

    def left_t(self, A):
        return fwht(A * self.s[:, None], axis=0) if self.pow2 else (self.Q.T @ A).astype(np.float32)

    def back(self, A):
        return fwht(A) * self.s if self.pow2 else (A @ self.Q.T).astype(np.float32)


class BlockH:
    """Block-diagonal randomized Hadamard on n = nb x b, b = the largest power of 2 dividing n (or given):
    right(A) = A @ blockdiag(diag(s_k) H_b) on A's last axis (same form for activations and weight columns).
    rows_t(A) = blockdiag(...)^T @ A on A's first axis. shared=True: the same signs in every block (per-head R)."""

    def __init__(self, n, rng, b=None, shared=False):
        self.n, self.b = n, b or (n & -n)
        s = rng.choice(np.array([-1, 1], np.float32), self.b if shared else n)
        self.s = np.tile(s, n // self.b) if shared else s

    def right(self, A):
        sh = A.shape
        return fwht((A * self.s).reshape(sh[:-1] + (sh[-1] // self.b, self.b))).reshape(sh)

    def rows_t(self, A):
        sh = A.shape
        B = (A * self.s.reshape((-1,) + (1,) * (A.ndim - 1))).reshape((sh[0] // self.b, self.b) + sh[1:])
        return fwht(B, axis=1).reshape(sh)


def fused(nd):
    nd["qkv"] = lambda nd=nd: np.concatenate([_w(nd["q"]), _w(nd["k"]), _w(nd["v"])])
    nd["gateup"] = lambda nd=nd: np.concatenate([_w(nd["gate"]), _w(nd["up"])])
    return nd


def install(m, layers, hooks=(None, None, None)):
    m.layers = layers
    m.post_embed, m.pre_head, m.down_in = hooks


def rotate(m, base, seed=0):
    rng = np.random.default_rng(seed)
    Q = Orth(m.D, rng)
    R = BlockH(m.H * m.hd, rng, b=m.hd, shared=True)          # per-head rotation of the attention output (and v)
    Rv = BlockH(m.KV * m.hd, rng, b=m.hd, shared=True)
    Rv.s = R.s[:m.KV * m.hd]                                   # the same R on the kv heads
    Hd = BlockH(m.cfg["intermediate_size"], rng)
    layers = []
    for d in base:
        g1, g2 = d["ln1"], d["ln2"]
        nd = dict(d)
        nd["q"] = lambda W=d["q"], g=g1: Q.right(_w(W) * g)
        nd["k"] = lambda W=d["k"], g=g1: Q.right(_w(W) * g)
        nd["v"] = lambda W=d["v"], g=g1: Rv.rows_t(Q.right(_w(W) * g))
        nd["o"] = lambda W=d["o"]: Q.left_t(R.right(_w(W)))
        nd["gate"] = lambda W=d["gate"], g=g2: Q.right(_w(W) * g)
        nd["up"] = lambda W=d["up"], g=g2: Q.right(_w(W) * g)
        nd["down"] = lambda W=d["down"]: Q.left_t(Hd.right(_w(W)))
        nd["ln1"], nd["ln2"] = np.ones_like(g1), np.ones_like(g2)
        layers.append(fused(nd))
    install(m, layers, (Q.right, Q.back, Hd.right))
    return "rotation: residual %s %d, per-head Hadamard %d, down input block Hadamard %d" % (
        "randomized Hadamard" if Q.pow2 else "dense random orthogonal", m.D, m.hd, Hd.b)


def smooth(m, base, chin, alpha=0.5):
    def sc(xmax, wmax):
        return (np.maximum(xmax, 1e-5) ** alpha / np.maximum(wmax, 1e-5) ** (1 - alpha)).astype(np.float32)
    layers = []
    for li, d in enumerate(base):
        nd = dict(d)
        s1 = sc(chin["%d.qkv" % li], np.max([np.abs(_w(d[n])).max(0) for n in "qkv"], 0))
        s2 = sc(chin["%d.gateup" % li], np.max([np.abs(_w(d[n])).max(0) for n in ("gate", "up")], 0))
        s3 = sc(chin["%d.down" % li], np.abs(_w(d["down"])).max(0))
        nd["ln1"], nd["ln2"] = d["ln1"] / s1, d["ln2"] / s2
        for n in "qkv":
            nd[n] = lambda W=d[n], s=s1: _w(W) * s
        nd["gate"] = lambda W=d["gate"], s=s2: _w(W) * s
        nd["up"] = lambda W=d["up"], s=s2, s3=s3: _w(W) * s / s3[:, None]
        nd["down"] = lambda W=d["down"], s=s3: _w(W) * s
        layers.append(fused(nd))
    install(m, layers)


def calibrate3(m, id_lists):
    """fp32 pass over the calibration texts, position 0 left out (it runs on the CPU): per-step input / output range,
    per-channel input |x| max, per-channel output min / max."""
    rec, chin, chlo, chhi = {}, {}, {}, {}

    class Rec(Quant):
        def linear(self, key, x, W):
            y = x @ _w(W).T
            if key == "head":
                return y
            xs, ys = (x[1:], y[1:]) if self.start == 0 else (x, y)
            if xs.shape[0] == 0:
                return y
            for k2, t in ((key + ".in", xs), (key + ".out", ys)):
                lo, hi = rec.get(k2, (np.inf, -np.inf))
                rec[k2] = (min(lo, float(t.min())), max(hi, float(t.max())))
            a, lo, hi = np.abs(xs).max(0), ys.min(0), ys.max(0)
            chin[key] = np.maximum(chin.get(key, a), a)
            chlo[key] = np.minimum(chlo.get(key, lo), lo)
            chhi[key] = np.maximum(chhi.get(key, hi), hi)
            return y
    for ids in id_lists:
        m.forward(ids, Rec(fused=True))
    return rec, chin, (chlo, chhi)


class QuantG(Quant):
    """The NPU format with explicit jobs: G jobs per step (groups of output channels), each with its own weight scale
    and (per_job_out) its own output scale and zero point; group_by = w (sorted by max |w|) | out (calibrated range)."""
    GRP = {}

    def __init__(self, *a, chout=None, group_by="w", per_job_out=False, margin=1.0, **k):
        super().__init__(*a, **k)
        self.chout, self.group_by, self.per_job_out, self.margin = chout, group_by, per_job_out, margin

    def wgroups(self, key, W):
        g = Quant.WGROUPS
        ck = "%s#%s%d" % (key, self.group_by, g)
        if ck in Quant.WQ:
            return Quant.WQ[ck], QuantG.GRP[ck]
        W = _w(W)
        r = np.abs(W).max(1)
        if self.group_by == "out":
            order = np.argsort(np.maximum(np.abs(self.chout[0][key]), np.abs(self.chout[1][key])))
        else:
            order = np.argsort(r)
        gid, s = np.empty(W.shape[0], np.int32), np.empty(W.shape[0], np.float32)
        for j, grp in enumerate(np.array_split(order, g if g > 0 else W.shape[0])):
            gid[grp], s[grp] = j, max(float(r[grp].max()), 1e-12) / 127.0
        wq = (np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8), s)
        if Quant.CACHE:
            Quant.WQ[ck], QuantG.GRP[ck] = wq, gid
        return wq, gid

    def out_q(self, key, y, gid):
        if not self.per_job_out:
            sy, zy = self.qparams(*self.rng(key + ".out", y))
            return (self.q8(y, sy, zy) - zy) * sy
        ng = int(gid.max()) + 1
        glo, ghi = np.zeros(ng), np.zeros(ng)
        np.minimum.at(glo, gid, self.chout[0][key])
        np.maximum.at(ghi, gid, self.chout[1][key])
        glo, ghi = glo * self.margin, ghi * self.margin   # headroom over the calibrated range (unseen text)
        s = np.maximum(ghi - glo, 1e-8) / 255.0
        zp = np.clip(np.round(-glo / s) - 128, -128, 127)
        sy, zy = s[gid], zp[gid]
        return (np.clip(np.round(y / sy) + zy, -128, 127) - zy) * sy

    def linear(self, key, x, W):
        if self.mode == "fp32" or (key == "head" and not self.quant_head):
            return x @ _w(W).T
        (Wq, sw), gid = self.wgroups(key, W)
        y0 = None
        if self.fp_first and self.start == 0:          # position 0: the CPU's job (int8 weights, fp activations)
            y0 = (x[:1] @ (Wq.astype(np.float32) * sw[:, None]).T).astype(np.float32)
            if x.shape[0] == 1:
                return y0
            x = x[1:]
        sx, zx = self.qparams(*self.rng(key + ".in", x))
        xq = self.q8(x, sx, zx) - zx
        ft = np.float64 if Quant.F64 else np.float32
        y = (xq.astype(ft) @ Wq.astype(ft).T) * (sx * sw.reshape(1, -1))
        y = self.out_q(key, y, gid).astype(np.float32)
        return y if y0 is None else np.concatenate([y0, y])


def main():
    from tokenizers import Tokenizer
    single = os.path.join(MD, "model.safetensors")
    big = not os.path.exists(single) or os.path.getsize(single) > 1 << 30
    Quant.CACHE, Quant.F64 = not big, not big
    t0 = time.monotonic()
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    m = SmolLM(MD)
    base = m.layers
    name = os.path.basename(MD)
    calib, evals = l2_quality.texts(tok)
    refs = {tn: logsoftmax(m.forward(ids, Quant("fp32"))) for tn, ids in evals.items()}
    out = {"model": name, "tokens_per_text": T, "rows": [], "outliers": {}}
    print("%s, %d tokens per text, every step on the NPU (8-bit), position 0 on the CPU, head fp32; fp32 %.0f s" % (
        name, T, time.monotonic() - t0), flush=True)

    def row(label, q, groups=16, fp=False):
        Quant.WGROUPS, Quant.WGROUPS_BY = groups, None
        Quant.WQ.clear()
        QuantG.GRP.clear()
        for tn, ids in evals.items():
            t = time.monotonic()
            s = score(refs[tn], logsoftmax(m.forward(ids, q)), np.asarray(ids))
            out["rows"].append({"variant": label, "groups": None if fp else groups, "text": tn,
                                **{k: float(v) for k, v in s.items()}})
            print("%-52s %-7s %8.2f %9.1f%% %10.4f   (%.0f s)" % (
                label, tn, s["ppl"], 100 * s["top1_vs_fp32"], s["kl_vs_fp32"], time.monotonic() - t), flush=True)

    def outliers(label, chin, rec):
        res = {}
        for st in STEPS:
            ks = [k for k in chin if k.endswith("." + st)]
            ratios = [float(chin[k].max() / max(float(np.median(chin[k])), 1e-12)) for k in ks]
            amax = [max(-rec[k + ".in"][0], rec[k + ".in"][1]) for k in ks]
            res[st] = {"ratio_median": float(np.median(ratios)), "ratio_max": max(ratios),
                       "in_absmax_median": float(np.median(amax)), "in_absmax_max": float(max(amax))}
        out["outliers"][label] = res
        print("%s: input outlier ratio (largest / median channel; median, max over layers), |x| max (median, max):"
              % label, flush=True)
        for st, r in res.items():
            print("   %-7s ratio %8.1f %8.1f   |x| %9.2f %9.2f" % (
                st, r["ratio_median"], r["ratio_max"], r["in_absmax_median"], r["in_absmax_max"]), flush=True)

    rec, chin, chout = calibrate3(m, calib)
    outliers("original", chin, rec)
    print("%-52s %-7s %8s %10s %10s" % ("variant", "text", "ppl", "top1=fp32", "KL"), flush=True)
    for tn, ids in evals.items():
        s = score(refs[tn], refs[tn], np.asarray(ids))
        out["rows"].append({"variant": "fp32", "groups": None, "text": tn, **{k: float(v) for k, v in s.items()}})
        print("%-52s %-7s %8.2f" % ("fp32", tn, s["ppl"]), flush=True)
    kw = dict(quant_head=False, fp_first=True)
    row("A baseline (out scale per step, groups by |w|)", QuantG("w8a8", True, rec, chout=chout, **kw))
    row("B per-job out scales (groups by |w|)",
        QuantG("w8a8", True, rec, chout=chout, per_job_out=True, **kw))

    smooth(m, base, chin)
    rec_s, chin_s, chout_s = calibrate3(m, calib)
    outliers("SmoothQuant 0.5", chin_s, rec_s)
    row("S SmoothQuant 0.5 + per-job out scales",
        QuantG("w8a8", True, rec_s, chout=chout_s, per_job_out=True, **kw))
    install(m, base)

    what = rotate(m, base)
    print(what, flush=True)
    out["rotation"] = what
    row("rotated model, fp32 (must equal fp32)", Quant("fp32"), fp=True)
    rec_r, chin_r, chout_r = calibrate3(m, calib)
    outliers("rotated", chin_r, rec_r)
    row("R rotation (out scale per step)", QuantG("w8a8", True, rec_r, chout=chout_r, **kw))
    row("RB rotation + per-job out scales",
        QuantG("w8a8", True, rec_r, chout=chout_r, per_job_out=True, **kw))
    row("RB4 rotation + per-job out scales, 4 groups",
        QuantG("w8a8", True, rec_r, chout=chout_r, per_job_out=True, **kw), groups=4)
    install(m, base)
    Quant.WQ.clear()
    path = os.path.join(RUNS, "l2g-outliers-%s-%s.json" % (name, time.strftime("%Y%m%d-%H%M%S")))
    json.dump(out, open(path, "w"), indent=1)
    print("wrote %s (%.0f min)" % (path, (time.monotonic() - t0) / 60), flush=True)


if __name__ == "__main__":
    sys.exit(main())

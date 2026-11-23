#!/usr/bin/env python3
"""test_jbglue.py (2026-11-23, CPU only): every kernel of libjbglue.so against the numpy code it replaces
(llm/ref/smollm.py, l2g_outliers.py, chunk_glue.py), at SmolLM2-135M and -1.7B shapes, 256 tokens; prints the worst
difference and numpy vs C time. Quantize / scatter / GEMV must be exact (levels), the float kernels within float32
noise. Usage: OMP_NUM_THREADS=4 GOMP_CPU_AFFINITY=2-5 test_jbglue.py"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "CORTEXA73")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/jordan/alta-npu/llm/ref")
import jbglue as J  # noqa: E402
sys.argv = [sys.argv[0], "/home/jordan/Documents/AI_Projects/npu-llm/models/SmolLM2-135M-Instruct"]
from l2g_outliers import fwht as np_fwht  # noqa: E402 (l2g_outliers reads a model dir from argv at import)

bad = []


def tm(f, n=3):
    t = time.monotonic()
    for _ in range(n):
        r = f()
    return r, (time.monotonic() - t) / n * 1e3


def report(name, ref_f, c_f, tol, exact=False):
    r, tn = tm(ref_f)
    c, tc = tm(c_f)
    r, c = np.asarray(r, np.float64), np.asarray(c, np.float64)
    d = float(np.abs(r - c).max())
    rel = d / max(float(np.abs(r).max()), 1e-30)
    ok = d == 0 if exact else rel <= tol
    if not ok:
        bad.append(name)
    print("%-34s max |diff| %.3g (rel %.2g) %s   numpy %8.2f ms  C %7.2f ms  x%.1f" % (
        name, d, rel, "ok" if ok else "FAIL", tn, tc, tn / max(tc, 1e-6)))


def shapes(D, H, KV, I, T=256, hd=64, eps=1e-5):
    rng = np.random.default_rng(D)
    tag = "D%d" % D
    x = rng.standard_normal((T, D)).astype(np.float32) * 3
    w = rng.uniform(0.5, 1.5, D).astype(np.float32)
    rms_np = lambda: (x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)).astype(np.float32) * w
    report(tag + " rmsnorm", rms_np, lambda: J.rmsnorm(x, w, eps), 1e-6)
    sx, zx = 0.031, -3
    q_np = lambda: np.ascontiguousarray(np.clip(np.round(x / np.float32(sx)) + zx, -128, 127).astype(np.int8).T)
    report(tag + " quant_planar", q_np, lambda: J.quant_planar(x, sx, zx), 0, exact=True)
    # dequant / scatter of 4 jobs (planar NPU buffer views), and a split step's accumulation
    OC = 3 * D
    perm = rng.permutation(OC).astype(np.int32)
    jobs = np.array_split(perm, 4)
    bufs = [rng.integers(-128, 128, (len(j), T)).astype(np.int8) for j in jobs]
    sc = [(float(rng.uniform(0.01, 0.1)), int(rng.integers(-20, 20))) for _ in jobs]

    def sc_np():
        y = np.zeros((T, OC), np.float32)
        for j, b, (s, z) in zip(jobs, bufs, sc):
            y[:, np.sort(j)] += (b.T.astype(np.float32) - z) * s
        return y

    def sc_c():
        y = np.zeros((T, OC), np.float32)
        for j, b, (s, z) in zip(jobs, bufs, sc):
            J.dequant_scatter(b.T, np.sort(j).astype(np.int32), s, z, y, True)
        return y
    report(tag + " dequant_scatter (4 jobs)", sc_np, sc_c, 0, exact=True)
    # position 0: int8 GEMV per job
    IC = D
    wq = [rng.integers(-127, 128, (len(j), IC)).astype(np.int8) for j in jobs]
    sw = [float(rng.uniform(1e-3, 1e-2)) for _ in jobs]
    x0 = x[0].copy()

    def gemv_np():
        y = np.zeros(OC, np.float32)
        for j, wj, s in zip(jobs, wq, sw):
            y[np.sort(j)] = x0 @ (wj.astype(np.float32) * s).T
        return y

    def gemv_c():
        y = np.zeros(OC, np.float32)
        for j, wj, s in zip(jobs, wq, sw):
            J.gemv_i8(x0, wj, s, np.sort(j).astype(np.int32), y, False)
        return y
    report(tag + " gemv_i8 (position 0)", gemv_np, gemv_c, 1e-5)
    # RoPE
    inv = 100000.0 ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    qh = rng.standard_normal((T, H, hd)).astype(np.float32)
    pos = np.arange(1, 1 + T)

    def rope_np():
        f = np.outer(pos, inv)
        cos, sin = np.cos(np.concatenate([f, f], -1))[:, None], np.sin(np.concatenate([f, f], -1))[:, None]
        h = hd // 2
        rot = np.concatenate([-qh[..., h:], qh[..., :h]], -1)
        return (qh * cos + rot * sin).astype(np.float32)

    def rope_c():
        a = qh.copy()
        J.rope(a, 1, inv)
        return a
    report(tag + " rope", rope_np, rope_c, 1e-6)
    # attention, causal, a whole chunk from position 0
    q = rng.standard_normal((T, H, hd)).astype(np.float32)
    k = rng.standard_normal((T, KV, hd)).astype(np.float32)
    v = rng.standard_normal((T, KV, hd)).astype(np.float32)

    def att_np():
        g = H // KV
        kk, vv = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)
        s = np.einsum("thd,shd->hts", q, kk) / np.sqrt(hd)
        p = np.arange(T)
        s = np.where((p[None, :] > p[:, None])[None], -1e30, s)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        return np.einsum("hts,shd->thd", s, vv).reshape(T, H * hd).astype(np.float32)
    report(tag + " attention (H %d, KV %d)" % (H, KV), att_np, lambda: J.attention(q, k, v, 0), 1e-5)
    gu = rng.standard_normal((T, 2 * I)).astype(np.float32) * 2
    silu_np = lambda: (gu[:, :I] / (1.0 + np.exp(-gu[:, :I])) * gu[:, I:]).astype(np.float32)
    report(tag + " silu_mul", silu_np, lambda: J.silu_mul(gu), 1e-6)
    b = I & -I
    s_ = rng.choice(np.array([-1, 1], np.float32), I)
    h = rng.standard_normal((T, I)).astype(np.float32)

    def fwht_np():
        sh = h.shape
        return np_fwht((h * s_).reshape(sh[:-1] + (sh[-1] // b, b))).reshape(sh)
    report(tag + " block FWHT (b %d)" % b, fwht_np, lambda: J.fwht(h.copy(), b, s_), 1e-5)
    if D & (D - 1) == 0:                            # the residual's randomized Hadamard (1.7B: 2048)
        sD = rng.choice(np.array([-1, 1], np.float32), D)
        report(tag + " Orth.back (fwht * s)", lambda: np_fwht(x) * sD, lambda: J.fwht(x.copy(), D, sD, True), 1e-5)


if __name__ == "__main__":
    print("libjbglue threads %d" % J.threads())
    shapes(576, 9, 3, 1536)                         # SmolLM2-135M
    shapes(2048, 32, 32, 8192)                      # SmolLM2-1.7B
    print("FAIL: " + ", ".join(bad) if bad else "ALL OK")


def fused(D, H, KV, I, T=256, hd=64):
    """the fused kernels must give exactly what the separate kernels give."""
    rng = np.random.default_rng(7 + D)
    tag = "D%d fused" % D
    x = rng.standard_normal((T + 1, D)).astype(np.float32) * 2
    s, z = 0.017, 2
    a = J.rmsnorm(x, None, 1e-5)
    ref = J.quant_planar(np.ascontiguousarray(a[1:]), s, z)
    got, a0 = J.norm_quant(x, 1e-5, s, z)
    ok = np.array_equal(ref, got) and np.array_equal(a[0], a0)
    print("%-34s %s" % (tag + " norm_quant", "exact" if ok else "DIFFERENT"))
    if not ok:
        bad.append(tag + " norm_quant")
    # FFN: gateup jobs (planar NPU buffers) -> SiLU * up -> FWHT -> planar int8
    perm = rng.permutation(2 * I).astype(np.int32)
    jobs = [np.sort(j).astype(np.int32) for j in np.array_split(perm, 4)]
    bufs = [rng.integers(-128, 128, (len(j), T)).astype(np.int8) for j in jobs]
    sc, zp = [0.02, 0.03, 0.025, 0.05], [1, -2, 0, 3]
    gu = np.empty((T, 2 * I), np.float32)
    J.scatter_jobs([b.T for b in bufs], jobs, sc, zp, gu)
    sg = rng.choice(np.array([-1, 1], np.float32), I)
    bI = I & -I
    h = J.silu_mul(gu)
    J.fwht(h, bI, sg)
    ref = J.quant_planar(h, 0.04, -1)
    got = J.ffn_fused([b.T for b in bufs], jobs, sc, zp, T, I, sg, bI, 0.04, -1)
    ok = np.array_equal(ref, got)
    print("%-34s %s" % (tag + " ffn_fused", "exact" if ok else "DIFFERENT (%d of %d)" % ((ref != got).sum(), ref.size)))
    if not ok:
        bad.append(tag + " ffn_fused")
    # qkv + RoPE
    OC = (H + 2 * KV) * hd
    perm = rng.permutation(OC).astype(np.int32)
    jobs = [np.sort(j).astype(np.int32) for j in np.array_split(perm, 4)]
    bufs = [rng.integers(-128, 128, (len(j), T)).astype(np.int8) for j in jobs]
    y = np.empty((T, OC), np.float32)
    J.scatter_jobs([b.T for b in bufs], jobs, sc, zp, y)
    inv = 100000.0 ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    qr = np.ascontiguousarray(y[:, :H * hd]).reshape(T, H, hd)
    kr = np.ascontiguousarray(y[:, H * hd:(H + KV) * hd]).reshape(T, KV, hd)
    vr = np.ascontiguousarray(y[:, (H + KV) * hd:]).reshape(T, KV, hd)
    J.rope(qr, 1, inv)                               # tokens 1..T are positions 1..T
    J.rope(kr, 1, inv)
    q = np.zeros((T + 1, H, hd), np.float32)
    k = np.zeros((T + 1, KV, hd), np.float32)
    v = np.zeros((T + 1, KV, hd), np.float32)
    J.qkv_rope([b.T for b in bufs], jobs, sc, zp, q, k, v, 0, inv)
    ok = np.array_equal(q[1:], qr) and np.array_equal(k[1:], kr) and np.array_equal(v[1:], vr)
    print("%-34s %s" % (tag + " qkv_rope", "exact" if ok else "DIFFERENT"))
    if not ok:
        bad.append(tag + " qkv_rope")
    t0 = time.monotonic()
    for _ in range(5):
        J.ffn_fused([b.T for b in bufs[:0]] or [bb.T for bb in [rng.integers(-128, 128, (len(j), T)).astype(np.int8)
                                                               for j in np.array_split(np.arange(2 * I, dtype=np.int32), 4)]],
                    [np.ascontiguousarray(j) for j in np.array_split(np.arange(2 * I, dtype=np.int32), 4)],
                    sc, zp, T, I, sg, bI, 0.04, -1)
    print("%-34s %.2f ms per call (incl. making test buffers)" % (tag + " ffn_fused time", (time.monotonic() - t0) / 5 * 1e3))


if __name__ == "__main__" and os.environ.get("JBG_FUSED", "1") == "1":
    fused(576, 9, 3, 1536)
    fused(2048, 32, 32, 8192)
    print("FUSED FAIL: " + ", ".join(bad) if bad else "FUSED ALL OK")

    sys.exit(1 if bad else 0)

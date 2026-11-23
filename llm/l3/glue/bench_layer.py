#!/usr/bin/env python3
"""bench_layer.py (2026-11-23, CPU only): the C glue of one 256-token chunk at real shapes without the NPU (random
int8 job outputs stand in for it), per part, x layers. Answers "what does the CPU side cost per chunk" for a model.
Usage: OMP_NUM_THREADS=4 GOMP_CPU_AFFINITY=2-5 taskset -c 2-5 bench_layer.py [135m|1p7b] [ROUNDS] [fused]"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jbglue as J  # noqa: E402

CFG = {"135m": dict(D=576, H=9, KV=3, I=1536, L=30, split=1),
       "1p7b": dict(D=2048, H=32, KV=32, I=8192, L=24, split=4)}
c = CFG[sys.argv[1] if len(sys.argv) > 1 else "1p7b"]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
D, H, KV, I, L, K = c["D"], c["H"], c["KV"], c["I"], c["L"], c["split"]
hd, T, JOBS = 64, 256, 4
rng = np.random.default_rng(0)


def step(OC, IC, pieces=1):
    """an NPU step: per job its rows (sorted), int8 weights, planar output buffer [n, T] (random levels)."""
    jobs = []
    for p in range(pieces):
        for r in np.array_split(rng.permutation(OC), JOBS):
            r = np.sort(r).astype(np.int32)
            jobs.append(dict(rows=r, w=rng.integers(-127, 128, (len(r), IC // pieces)).astype(np.int8),
                             col=p * (IC // pieces), out=rng.integers(-128, 128, (len(r), T)).astype(np.int8)))
    return jobs


steps = {"qkv": step((H + 2 * KV) * hd, D), "o": step(D, H * hd), "gateup": step(2 * I, D), "down": step(D, I, K)}
inv = 100000.0 ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
s_I = rng.choice(np.array([-1, 1], np.float32), I)
b_I = I & -I
tm = {}


def tick(name, t):
    tm[name] = tm.get(name, 0.0) + time.monotonic() - t
    return time.monotonic()


def linear(name, a, OC, add_to=None):
    js = steps[name]
    y = add_to if add_to is not None else np.empty((T + 1, OC), np.float32)
    acc = add_to is not None
    t = time.monotonic()
    J.gemv_jobs(np.ascontiguousarray(a[0]), [j["w"] for j in js], [j["col"] for j in js], [0.01] * len(js),
                [j["rows"] for j in js], y[0], acc)
    t = tick("pos0 gemv", t)
    xp = J.quant_planar(np.ascontiguousarray(a[1:]), 0.03, 0)
    t = tick("quantize", t)
    J.scatter_jobs([j["out"].T for j in js], [j["rows"] for j in js], [0.01] * len(js), [0] * len(js), y[1:], acc)
    tick("scatter", t)
    return y


FUSED = len(sys.argv) > 3 and sys.argv[3] == "fused"


def pos0(name, a0, OC, y0=None, acc=False):
    js = steps[name]
    y0 = np.empty(OC, np.float32) if y0 is None else y0
    J.gemv_jobs(a0, [j["w"] for j in js], [j["col"] for j in js], [0.01] * len(js), [j["rows"] for j in js], y0, acc)
    return y0


def outs(name):
    js = steps[name]
    return [j["out"].T for j in js], [j["rows"] for j in js], [0.01] * len(js), [0] * len(js)


def layer_fused(x):
    t = time.monotonic()
    ap, a0 = J.norm_quant(x, 1e-5, 0.03, 0)            # rmsnorm + quantize, row 0 in float
    t = tick("norm+quant", t)
    y0 = pos0("qkv", a0, (H + 2 * KV) * hd)
    t = tick("pos0 gemv", t)
    q = np.empty((T + 1, H, hd), np.float32)
    k = np.empty((T + 1, KV, hd), np.float32)
    v = np.empty((T + 1, KV, hd), np.float32)
    J.qkv_rope(*outs("qkv"), q, k, v, 0, inv)
    q[0], k[0], v[0] = (y0[:H * hd].reshape(H, hd), y0[H * hd:(H + KV) * hd].reshape(KV, hd),
                        y0[(H + KV) * hd:].reshape(KV, hd))   # position 0: RoPE is the identity
    t = tick("qkv+rope", t)
    att = J.attention(q, k, v, 0)
    t = tick("attention", t)
    pos0("o", np.ascontiguousarray(att[0]), D, x[0], True)
    t = tick("pos0 gemv", t)
    J.quant_planar(att[1:], 0.03, 0)
    t = tick("quantize", t)
    J.scatter_jobs(*outs("o"), x[1:], True)
    t = tick("scatter", t)
    ap, a0 = J.norm_quant(x, 1e-5, 0.03, 0)
    t = tick("norm+quant", t)
    gu0 = pos0("gateup", a0, 2 * I)
    t = tick("pos0 gemv", t)
    h0 = J.silu_mul(gu0[None])
    J.fwht(h0, b_I, s_I)
    hp = J.ffn_fused(*outs("gateup"), T, I, s_I, b_I, 0.03, 0)
    t = tick("ffn fused", t)
    pos0("down", h0[0], D, x[0], True)
    t = tick("pos0 gemv", t)
    J.scatter_jobs(*outs("down"), x[1:], True)
    tick("scatter", t)


x = rng.standard_normal((T + 1, D)).astype(np.float32)
t_all = time.monotonic()
if FUSED:
    for _ in range(ROUNDS):
        for li in range(L):
            layer_fused(x)
            if not np.isfinite(x).all():
                x[:] = rng.standard_normal(x.shape)
    ROUNDS_DONE = True
for _ in range(0 if FUSED else ROUNDS):
    for li in range(L):
        t = time.monotonic()
        a = J.rmsnorm(x, None, 1e-5)
        tick("norm", t)
        qkv = linear("qkv", a, (H + 2 * KV) * hd)
        t = time.monotonic()
        q = np.ascontiguousarray(qkv[:, :H * hd]).reshape(T + 1, H, hd)
        k = np.ascontiguousarray(qkv[:, H * hd:(H + KV) * hd]).reshape(T + 1, KV, hd)
        v = np.ascontiguousarray(qkv[:, (H + KV) * hd:]).reshape(T + 1, KV, hd)
        t = tick("split qkv", t)
        J.rope(q, 0, inv)
        J.rope(k, 0, inv)
        t = tick("rope", t)
        att = J.attention(q, k, v, 0)
        tick("attention", t)
        if not np.isfinite(x).all():
            x[:] = rng.standard_normal(x.shape)        # random job outputs grow the residual: keep it finite
        linear("o", att, D, add_to=x)
        t = time.monotonic()
        a = J.rmsnorm(x, None, 1e-5)
        tick("norm", t)
        gu = linear("gateup", a, 2 * I)
        t = time.monotonic()
        h = J.silu_mul(gu)
        J.fwht(h, b_I, s_I)
        tick("silu + fwht", t)
        linear("down", h, D, add_to=x)
wall = (time.monotonic() - t_all) / ROUNDS
print("%s, %d layers, 256-token chunk, C glue only%s (threads %d): %.0f ms per chunk" % (
    sys.argv[1] if len(sys.argv) > 1 else "1p7b", L, ", fused" if FUSED else "", J.threads(), wall * 1e3))
for k_, v_ in sorted(tm.items(), key=lambda kv: -kv[1]):
    print("  %-12s %7.1f ms" % (k_, v_ / ROUNDS * 1e3))

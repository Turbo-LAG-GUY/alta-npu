#!/usr/bin/env python3
"""compare_libs.py (2026-11-24): run the glue kernels on fixed random inputs (1.7B shapes) with one libjbglue build and
save the outputs; with two saved files, compare them bit for bit. Used to prove a speed change kept every bit.
Usage: JB_GLUE_LIB=<lib.so> compare_libs.py run OUT.npz      compare_libs.py cmp A.npz B.npz"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(out):
    import jbglue as J
    rng = np.random.default_rng(11)
    T, D, H, KV, I, hd = 256, 2048, 32, 32, 8192, 64
    sc, zp = [0.02, 0.03, 0.025, 0.05], [1, -2, 0, 3]
    res = {}
    OC = (H + 2 * KV) * hd
    perm = rng.permutation(OC).astype(np.int32)
    jobs = [np.sort(j).astype(np.int32) for j in np.array_split(perm, 4)]
    bufs = [rng.integers(-128, 128, (len(j), T)).astype(np.int8) for j in jobs]
    y = np.empty((T, OC), np.float32)
    J.scatter_jobs([b.T for b in bufs], jobs, sc, zp, y)
    res["scatter_partition"] = y
    split = [np.sort(rng.choice(D, D * 3 // 4, replace=False)).astype(np.int32) for _ in range(4)]
    sb = [rng.integers(-128, 128, (len(j), T)).astype(np.int8) for j in split]
    y = rng.standard_normal((T, D)).astype(np.float32)
    J.scatter_jobs([b.T for b in sb], split, sc, zp, y, accumulate=True)
    res["scatter_split_acc"] = y
    q = np.zeros((T + 1, H, hd), np.float32)
    k = np.zeros((T + 1, KV, hd), np.float32)
    v = np.zeros((T + 1, KV, hd), np.float32)
    inv = 100000.0 ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    J.qkv_rope([b.T for b in bufs], jobs, sc, zp, q, k, v, 0, inv)
    res["qkv_q"], res["qkv_k"], res["qkv_v"] = q, k, v
    perm = rng.permutation(2 * I).astype(np.int32)
    gj = [np.sort(j).astype(np.int32) for j in np.array_split(perm, 4)]
    gb = [rng.integers(-128, 128, (len(j), T)).astype(np.int8) for j in gj]
    sg = rng.choice(np.array([-1, 1], np.float32), I)
    res["ffn"] = J.ffn_fused([b.T for b in gb], gj, sc, zp, T, I, sg, I & -I, 0.04, -1)
    np.savez(out, **res)
    print("saved %s (%s)" % (out, os.environ.get("JB_GLUE_LIB", "default lib")))


def cmp(a, b):
    A, B = np.load(a), np.load(b)
    ok = True
    for name in A.files:
        same = A[name].tobytes() == B[name].tobytes()
        ok &= same
        print("%-20s %s" % (name, "bit-identical" if same else "DIFFERENT (%d of %d)" % (
            (A[name] != B[name]).sum(), A[name].size)))
    print("LIBS IDENTICAL" if ok else "LIBS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[2]) if sys.argv[1] == "run" else cmp(sys.argv[2], sys.argv[3]))

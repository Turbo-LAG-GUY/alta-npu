#!/usr/bin/env python3
"""test_head.py (2026-11-24): jbg_head_bf16 (the tied head on bf16 embeddings) vs a float64 numpy reference, on
synthetic shapes (V % 4 != 0, D % 8 != 0 tails) and on a model's real embedding table from its model.jba, with timing:
the last token only (nl 1, memory-bound: the table read once) and every token (nl 256).
Usage: test_head.py [MODEL_JBA]      (e.g. npu-llm/models/jb/aux-1p7b/model.jba)"""
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jbglue as G


def f32(E):
    return (E.astype(np.uint32) << 16).view(np.float32)


def check(x, E, tol=2e-5):
    got = G.head_bf16(x, E)
    ref = x.astype(np.float64) @ f32(E).astype(np.float64).T
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
    return got, err, err < tol


def embed_of(path):
    m = np.memmap(path, np.uint8, "r")
    o = 4
    while o + 80 <= m.size:
        name = bytes(m[o:o + 48]).rstrip(b"\0").decode()
        dt, nd, *dims = struct.unpack("<II4I", bytes(m[o + 48:o + 72]))
        nb = struct.unpack("<Q", bytes(m[o + 72:o + 80]))[0]
        if name == "embed":
            return np.frombuffer(m, np.uint16, nb // 2, o + 80).reshape(dims[0], dims[1])
        o += 80 + nb
    raise SystemExit("no embed record in " + path)


def main():
    rng = np.random.default_rng(1)
    ok = True
    for nl, V, D in [(1, 1003, 64), (3, 1001, 76), (37, 515, 128), (2, 4, 8)]:
        E = (rng.standard_normal((V, D)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        x = rng.standard_normal((nl, D)).astype(np.float32)
        _, err, good = check(x, E)
        ok &= good
        print("synthetic nl %3d V %5d D %4d: max rel err %.2e %s" % (nl, V, D, err, "OK" if good else "FAIL"))
    if len(sys.argv) > 1:
        E = np.ascontiguousarray(embed_of(sys.argv[1]))
        V, D = E.shape
        for nl in (1, 256):
            x = rng.standard_normal((nl, D)).astype(np.float32)
            G.head_bf16(x[:1], E)                           # warm: the table resident
            ts = []
            for _ in range(3 if nl > 1 else 10):
                t = time.monotonic()
                got = G.head_bf16(x, E)
                ts.append(time.monotonic() - t)
            k = min(nl, 8)
            ref = x[:k].astype(np.float64) @ f32(E).astype(np.float64).T
            err = np.abs(got[:k] - ref).max() / np.abs(ref).max()
            ok &= err < 2e-5
            gb = E.nbytes * -(-nl // 32) / 1e9
            print("real V %d D %d nl %3d: %.1f ms (median of %d; %.1f GB/s of table, %.1f GMAC/s), max rel err %.2e"
                  % (V, D, nl, 1e3 * np.median(ts), len(ts), gb / np.median(ts), nl * V * D / 1e9 / np.median(ts),
                     err))
    print("HEAD ALL OK" if ok else "HEAD FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

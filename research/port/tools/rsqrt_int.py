#!/usr/bin/env python3
"""rsqrt_int.py (step 6g, 2026-11-14, CPU only): TFLite's integer int8 RSQRT (elementwise.cc RsqrtEvalQuantized:
GetInvSqrtQuantizedMultiplierExp, a gemmlowp fixed-point Newton-Raphson, then two MultiplyByQuantizedMultiplier),
written out in Python integers, and an exhaustive check against the interpreter. The float LUTPopulate formula fitted
first matched 10240 random entries but missed one entry of the layernorm model's RSQRT (stored 143: CPU 46, float 45).

  rsqrt_int.py [--trials N]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eltwise_fit as E  # noqa: E402
import reduce_models as R  # noqa: E402

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


def wrap32(x):
    x &= 0xffffffff
    return x - (1 << 32) if x & 0x80000000 else x


def srdhm(a, b):
    if a == I32_MIN and b == I32_MIN:
        return I32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    v = ab + nudge
    q = abs(v) >> 31
    return q if v >= 0 else -q  # C++ division truncates toward zero


def rdbpot(x, e):
    mask = (1 << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> e) + (1 if rem > thr else 0)


def sat_shift_left(x, e):
    thr = (1 << (31 - e)) - 1
    if x > thr:
        return I32_MAX
    if x < -thr:
        return I32_MIN
    return wrap32(x << e)


def mbqm(x, m, shift):
    left, right = max(shift, 0), max(-shift, 0)
    return rdbpot(srdhm(wrap32(x * (1 << left)), m), right)


def inv_sqrt_qm_exp(value, reverse_shift=-1):
    """TFLite GetInvSqrtQuantizedMultiplierExp"""
    if value <= 1:
        return I32_MAX, 0
    shift = 11
    x_in = value
    while x_in >= (1 << 29):
        x_in //= 4
        shift += 1
    clz = 32 - x_in.bit_length()
    max_left_shift_bits = clz - 1
    left_shift_bit_pairs = max_left_shift_bits // 2 - 1
    shift -= left_shift_bit_pairs
    x_in <<= 2 * left_shift_bit_pairs
    fixed_input = x_in >> 1                          # F3 raw
    half_input = rdbpot(fixed_input, 1)              # SaturatingRoundingMultiplyByPOT<-1>
    half_three = (1 << 28) + (1 << 27)               # 1.5 in F3
    x = 1 << 28                                      # F3 one
    for _ in range(5):
        x3 = sat_shift_left(srdhm(srdhm(x, x), x), 6)            # Rescale<3>(F9)
        t = wrap32(srdhm(half_three, x) - srdhm(half_input, x3))  # F6 - F6
        x = sat_shift_left(t, 3)                                   # Rescale<3>(F6)
    x = srdhm(x, 1518500250)                         # * sqrt(2)/2 (F0)
    if shift < 0:
        x = wrap32(x << -shift)
        shift = 0
    return x, shift * reverse_shift


def rsqrt_int(xs, iq, oq, lo, hi, variant):
    si, zi = iq
    so, zo = oq
    f32 = np.float32
    if variant == "f32_inner":
        mult = 1.0 / float(np.sqrt(f32(si)) * f32(so))
    else:
        mult = 1.0 / (np.sqrt(float(f32(si))) * float(f32(so)))
    m, sh = R.quantize_multiplier(mult)
    out = []
    for v in xs:
        value = int(v) - zi
        if value == 0:
            out.append(hi)
            continue
        im, ish = inv_sqrt_qm_exp(value)
        data = mbqm(1, im, ish + 20)
        o = mbqm(data, m, sh - 20) + zo
        out.append(min(max(o, lo), hi))
    return np.array(out, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    a = ap.parse_args()
    rng = np.random.default_rng(1114)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "rsqrt_int.tflite")
    cases = [((0.019582165405154228, -128), (0.0034443186596035957, -128))]   # the layernorm model's RSQRT
    for _ in range(a.trials):
        cases.append((E.rand_q(rng, True, 0.0005, 0.05), E.rand_q(rng, True, 0.01, 0.5)))
    tot = {}
    for iq, oq in cases:
        E.build(tmp, "rsqrt", 1, True, [iq], oq, [(1, 1, 1, 256)], (1, 1, 1, 256))
        x = np.arange(-128, 128)
        valid = x >= iq[1]
        xin = np.where(valid, x, iq[1]).astype(np.int8).reshape(1, 1, 1, 256)
        got = E.run(tmp, [xin], False).reshape(256)
        fl = E.cand_rsqrt(xin.astype(np.int64).reshape(256), iq, oq, -128, 127)["lutpopulate"]
        res = {"lutpopulate_float": fl}
        for v in ("f64", "f32_inner"):
            res["int_" + v] = rsqrt_int(xin.reshape(256), iq, oq, -128, 127, v)
        for k, v in res.items():
            e = tot.setdefault(k, [0, 0])
            e[0] += int((v[valid] == got[valid]).sum())
            e[1] += int(valid.sum())
    for k, (g, n) in tot.items():
        print("%-20s %s" % (k, "EXACT (%d entries, %d cases)" % (n, len(cases)) if g == n else "%d/%d" % (g, n)))


if __name__ == "__main__":
    main()

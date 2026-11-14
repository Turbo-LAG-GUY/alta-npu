import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Documents/AI_Projects/npu-shader/models"))
import reduce_models as R
from ai_edge_litert.interpreter import Interpreter
f32, q = np.float32, np.longdouble          # aarch64 long double = IEEE quad: exact for these products/sums
assert np.finfo(q).nmant >= 100
def xnn(S, n, iq, oq, lo, hi, fma_zp, fma_main):
    s_eff = f32(f32(f32(iq[0]) / f32(oq[0])) * f32(f32(1.0) / f32(n)))
    nz = f32(n * iq[1])
    zp = f32(q(oq[1]) - q(nz) * q(s_eff)) if fma_zp else f32(f32(oq[1]) - f32(nz * s_eff))
    Sf = S.astype(f32)
    v = (q(Sf) * q(s_eff) + q(zp)).astype(f32) if fma_main else (Sf * s_eff).astype(f32) + zp
    return np.clip(np.round(v), lo, hi)       # np.round = half to even = nearbyint
rng = np.random.default_rng(11)
cases = [
    ("ch-diffq-u8", (1, 1, 64, 256), [3], False, (0.05, 120), (0.03, 128)),
    ("ch-sameq-u8", (1, 1, 64, 256), [3], False, (0.05, 120), (0.05, 120)),
    ("sp-diffq-u8", (1, 7, 7, 1024), [1, 2], False, (0.0235, 0), (0.0117, 5)),
    ("sp-sameq-u8", (1, 7, 7, 1024), [1, 2], False, (0.0235, 0), (0.0235, 0)),
    ("ch-diffq-i8", (1, 1, 64, 256), [3], True, (0.05, -8), (0.03, 0)),
    ("ch-sameq-i8", (1, 1, 64, 256), [3], True, (0.05, -8), (0.05, -8)),
    ("sp-diffq-i8", (1, 7, 7, 1024), [1, 2], True, (0.0235, -128), (0.0117, -100)),
    ("ch-odd-u8", (1, 2, 5, 768), [3], False, (0.021, 77), (0.0123, 131)),
    ("ch-sameq-i8b", (1, 4, 4, 192), [3], True, (0.1, 3), (0.1, 3)),
    ("ch-sameq-u8c", (1, 4, 4, 192), [3], False, (0.1, 131), (0.1, 131)),
    ("ch-i8-c", (1, 1, 50, 384), [3], True, (0.07, 5), (0.02, -3)),
]
tmp = tempfile.mkdtemp(prefix="fitx2_", dir=".")
tot = {}
for name, shape, axes, int8, iq, oq in cases:
    p = os.path.join(tmp, name + ".tflite"); R.build_mean(p, shape, axes, int8, iq, oq)
    it = Interpreter(model_path=p, num_threads=1); it.allocate_tensors()
    lo, hi = (-128, 127) if int8 else (0, 255); dt = np.int8 if int8 else np.uint8
    n = int(np.prod([shape[a] for a in axes])); sc = {}
    for t in range(60):
        x = rng.integers(lo, hi + 1, size=shape).astype(dt)
        it.set_tensor(it.get_input_details()[0]["index"], x); it.invoke()
        got = it.get_tensor(it.get_output_details()[0]["index"]).astype(np.int64)
        S = x.astype(np.int64).sum(axis=tuple(axes), keepdims=True)
        for fz in (0, 1):
            for fm in (0, 1):
                k = "zp_%s/main_%s" % ("fma" if fz else "mul", "fma" if fm else "mul")
                v = xnn(S, n, iq, oq, lo, hi, fz, fm)
                e = sc.setdefault(k, [0, 0]); e[0] += int((v == got).sum()); e[1] += got.size
    for k, (a, b) in sc.items():
        t = tot.setdefault(k, [0, 0]); t[0] += a; t[1] += b
    print("%-13s %s" % (name, "  ".join("%s %s" % (k, "EXACT" if a == b else "%d/%d" % (a, b)) for k, (a, b) in sc.items())))
print("ALL:", "  ".join("%s %d/%d" % (k, a, b) for k, (a, b) in tot.items()))

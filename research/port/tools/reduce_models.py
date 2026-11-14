#!/usr/bin/env python3
"""reduce_models.py (step 6e, CPU only, nothing here touches the NPU): single-op TFLite MEAN models for the shader-core
reductions, and a fit of TFLite's quantized MEAN arithmetic against the CPU interpreter.

  reduce_models.py make OUT.tflite --shape 1 1 64 256 --axes 3 [--int8] [--in-q S ZP] [--out-q S ZP] [--conv]
      MEAN over the given axes (keep_dims), 8-bit per-tensor quantization. --conv puts a 1x1 CONV_2D (identity-ish
      weights) in front, so in Teflon the MEAN reads an NN output (channel planes) instead of the graph input.
  reduce_models.py fit [--trials N]
      builds a set of MEAN models in a temp dir, runs each on the CPU with random inputs and reports which candidate
      formula reproduces the interpreter's bytes exactly.
"""
import argparse
import os
import sys
import tempfile

import numpy as np


def build_mean(path, shape, axes, int8=False, in_q=(0.05, 120), out_q=(0.03, 128), conv=False):
    import flatbuffers
    from ai_edge_litert import schema_py_generated as S

    T8 = S.TensorType.INT8 if int8 else S.TensorType.UINT8
    buffers = [S.BufferT()]
    tensors = []

    def add_buffer(arr):
        b = S.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(arr).tobytes(), np.uint8)
        buffers.append(b)
        return len(buffers) - 1

    def tensor(name, shp, ttype, q=None, buf=0):
        t = S.TensorT()
        t.name = name
        t.shape = list(shp)
        t.type = ttype
        t.buffer = buf
        if q is not None:
            qp = S.QuantizationParametersT()
            qp.scale = [float(q[0])]
            qp.zeroPoint = [int(q[1])]
            t.quantization = qp
        tensors.append(t)
        return len(tensors) - 1

    codes, ops = [], []

    def opcode(builtin):
        for i, c in enumerate(codes):
            if c.builtinCode == builtin:
                return i
        c = S.OperatorCodeT()
        c.builtinCode = builtin
        c.deprecatedBuiltinCode = builtin
        c.version = 2 if builtin == S.BuiltinOperator.MEAN else 1
        codes.append(c)
        return len(codes) - 1

    x = first = tensor("input", shape, T8, in_q)
    if conv:
        # 1x1 conv, weights = 1 at the diagonal (scale 1/127 or 1/255 range), output keeps the input quantization
        c = shape[3]
        wq = np.zeros((c, 1, 1, c), np.int8 if int8 else np.uint8)
        wzp = 0 if int8 else 128
        wq[...] = wzp
        for i in range(c):
            wq[i, 0, 0, i] = wzp + 100
        ws = 1.0 / 100.0
        wt = tensor("conv_w", wq.shape, T8, (ws, wzp), add_buffer(wq))
        bt = tensor("conv_b", (c,), S.TensorType.INT32, (in_q[0] * ws, 0), add_buffer(np.zeros(c, np.int32)))
        y = tensor("conv_out", shape, T8, in_q)
        o = S.Conv2DOptionsT()
        o.padding = S.Padding.SAME
        o.strideW = o.strideH = o.dilationWFactor = o.dilationHFactor = 1
        o.fusedActivationFunction = S.ActivationFunctionType.NONE
        op = S.OperatorT()
        op.opcodeIndex = opcode(S.BuiltinOperator.CONV_2D)
        op.inputs, op.outputs = [x, wt, bt], [y]
        op.builtinOptionsType, op.builtinOptions = S.BuiltinOptions.Conv2DOptions, o
        ops.append(op)
        x = y

    oshape = [1 if i in axes else d for i, d in enumerate(shape)]
    at = tensor("axes", (len(axes),), S.TensorType.INT32, None, add_buffer(np.array(axes, np.int32)))
    out = tensor("mean", oshape, T8, out_q)
    ro = S.ReducerOptionsT()
    ro.keepDims = True
    op = S.OperatorT()
    op.opcodeIndex = opcode(S.BuiltinOperator.MEAN)
    op.inputs, op.outputs = [x, at], [out]
    op.builtinOptionsType, op.builtinOptions = S.BuiltinOptions.ReducerOptions, ro
    ops.append(op)

    sub = S.SubGraphT()
    sub.tensors, sub.inputs, sub.outputs, sub.operators, sub.name = tensors, [first], [out], ops, "mean"
    m = S.ModelT()
    m.version = 3
    m.operatorCodes, m.subgraphs, m.buffers = codes, [sub], buffers
    m.description = "alta step 6e MEAN test"
    b = flatbuffers.Builder(1 << 16)
    b.Finish(m.Pack(b), b"TFL3")
    with open(path, "wb") as f:
        f.write(bytes(b.Output()))


def cpu_run(path, x, default_delegates=True):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    kw = {} if default_delegates else {"experimental_op_resolver_type": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES}
    it = Interpreter(model_path=path, num_threads=1, **kw)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x)
    it.invoke()
    return it.get_tensor(o["index"])


# --- candidate formulas (int64 numpy) ---------------------------------------------------------------------------

def quantize_multiplier(real):
    """TFLite QuantizeMultiplier: real = m * 2^shift, m in [0.5, 1) as Q31."""
    if real == 0:
        return 0, 0
    q, shift = np.frexp(real)
    qf = int(round(float(q) * (1 << 31)))
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    return qf, int(shift)


def mul_by_qm(x, m, shift):
    """TFLite MultiplyByQuantizedMultiplier (gemmlowp SaturatingRoundingDoublingHighMul + RoundingDivideByPOT)."""
    x = x.astype(np.int64)
    left = max(shift, 0)
    right = max(-shift, 0)
    a = x * (1 << left)
    ab = a * m
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    hi = (ab + nudge) // (1 << 31)
    hi = np.where((ab + nudge) < 0, -((-(ab + nudge)) // (1 << 31)), hi)  # C++ division truncates toward zero
    if right == 0:
        return hi
    mask = (1 << right) - 1
    rem = hi & mask
    thr = (mask >> 1) + (hi < 0)
    return (hi >> right) + (rem > thr)


def candidates(x, axes, in_q, out_q, lo, hi):
    """x: int64 values as stored (int8 or uint8). Returns {name: output} (same shape as keep_dims)."""
    n = int(np.prod([x.shape[a] for a in axes]))
    s = x.sum(axis=tuple(axes), keepdims=True).astype(np.int64)
    si, zi = in_q
    so, zo = out_q
    out = {}
    # A: TFLite reference QuantizedMeanOrSum, float32 math, round half away from zero
    scale = np.float32(si) / np.float32(so)
    bias = -np.float32(zi) * scale
    fm = s.astype(np.float32) / np.float32(n)
    r = fm * scale + bias
    out["float32_ref"] = np.clip(np.where(r >= 0, np.floor(r + 0.5), np.ceil(r - 0.5)) + zo, lo, hi)
    # A2: same in float64
    r64 = s / n * (si / so) - zi * (si / so)
    out["float64_ref"] = np.clip(np.where(r64 >= 0, np.floor(r64 + 0.5), np.ceil(r64 - 0.5)) + zo, lo, hi)
    # B: integer: (sum - n*zp_in) * QM(si / (n*so)) + zp_out
    m, sh = quantize_multiplier(si / (n * so))
    out["int_qm_n"] = np.clip(mul_by_qm(s - n * zi, m, sh) + zo, lo, hi)
    # B2: integer: mean first (rounded integer division), then QM(si/so)
    acc = s - n * zi
    mean_r = np.where(acc >= 0, (acc + n // 2) // n, -((-acc + n // 2) // n))
    m2, sh2 = quantize_multiplier(si / so)
    out["int_div_then_qm"] = np.clip(mul_by_qm(mean_r, m2, sh2) + zo, lo, hi)
    # C: same quantization: rounded and truncated integer mean
    out["same_q_round"] = np.clip(np.where(s >= 0, (s + n // 2) // n, -((-s + n // 2) // n)), lo, hi)
    out["same_q_trunc"] = np.clip(np.where(s >= 0, s // n, -((-s) // n)), lo, hi)
    return out


def fit(trials):
    rng = np.random.default_rng(6)
    cases = [
        ("ch-diffq-u8", (1, 1, 64, 256), [3], False, (0.05, 120), (0.03, 128)),
        ("ch-sameq-u8", (1, 1, 64, 256), [3], False, (0.05, 120), (0.05, 120)),
        ("ch-diffq-u8-hw", (1, 8, 8, 64), [3], False, (0.1, 3), (0.004, 10)),
        ("sp-diffq-u8", (1, 7, 7, 1024), [1, 2], False, (0.0235, 0), (0.0117, 5)),
        ("sp-sameq-u8", (1, 7, 7, 1024), [1, 2], False, (0.0235, 0), (0.0235, 0)),
        ("ch-diffq-i8", (1, 1, 64, 256), [3], True, (0.05, -8), (0.03, 0)),
        ("ch-sameq-i8", (1, 1, 64, 256), [3], True, (0.05, -8), (0.05, -8)),
        ("sp-diffq-i8", (1, 7, 7, 1024), [1, 2], True, (0.0235, -128), (0.0117, -100)),
    ]
    tmp = tempfile.mkdtemp(prefix="reduce_fit_")
    for name, shape, axes, int8, iq, oq in cases:
        path = os.path.join(tmp, name + ".tflite")
        build_mean(path, shape, axes, int8, iq, oq)
        lo, hi = (-128, 127) if int8 else (0, 255)
        dt = np.int8 if int8 else np.uint8
        score = {}
        same_default = True
        for t in range(trials):
            x = rng.integers(lo, hi + 1, size=shape).astype(dt)
            if t == 0:
                x[...] = hi
            if t == 1:
                x[...] = lo
            got = cpu_run(path, x).astype(np.int64)
            if t < 3:
                same_default &= np.array_equal(got, cpu_run(path, x, default_delegates=False).astype(np.int64))
            for k, v in candidates(x.astype(np.int64), axes, iq, oq, lo, hi).items():
                e = score.setdefault(k, [0, 0])
                e[0] += int((v == got).sum())
                e[1] += got.size
        best = [k for k, (a, b) in score.items() if a == b]
        print("%-16s exact: %-40s  default delegates = builtin: %s   (%s)" % (
            name, ",".join(best) or "NONE", same_default,
            " ".join("%s %d/%d" % (k, a, b) for k, (a, b) in score.items() if a != b)))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    mk = sp.add_parser("make")
    mk.add_argument("out")
    mk.add_argument("--shape", type=int, nargs=4, required=True)
    mk.add_argument("--axes", type=int, nargs="+", required=True)
    mk.add_argument("--int8", action="store_true")
    mk.add_argument("--in-q", type=float, nargs=2, default=(0.05, 120))
    mk.add_argument("--out-q", type=float, nargs=2, default=(0.03, 128))
    mk.add_argument("--conv", action="store_true")
    ft = sp.add_parser("fit")
    ft.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "make":
        build_mean(a.out, a.shape, a.axes, a.int8, (a.in_q[0], int(a.in_q[1])), (a.out_q[0], int(a.out_q[1])), a.conv)
        print("wrote", a.out)
    else:
        fit(a.trials)


if __name__ == "__main__":
    sys.exit(main())

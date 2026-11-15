#!/usr/bin/env python3
"""layernorm_model.py (step 6g, 2026-11-14, CPU only, nothing here touches the NPU): an int8 layer normalization over
the last axis of a [1, T, C] tensor, decomposed the way TFLite's converter emits Keras LayerNormalization
(tf.nn.moments + tf.nn.batch_normalization):

    m   = MEAN(x, axis -1, keep_dims)        [1,T,1]
    sd  = SQUARED_DIFFERENCE(x, m)           [1,T,C]
    v   = MEAN(sd, axis -1, keep_dims)       [1,T,1]
    ve  = ADD(v, eps)                        [1,T,1]   eps: constant scalar
    r   = RSQRT(ve)                          [1,T,1]
    rg  = MUL(r, gamma)                      [1,T,C]   gamma: constant [C]
    xs  = MUL(x, rg)                         [1,T,C]
    ms  = MUL(m, rg)                         [1,T,C]
    bms = SUB(beta, ms)                      [1,T,C]   beta: constant [C]
    y   = ADD(xs, bms)                       [1,T,C]

Every tensor is int8 with per-tensor asymmetric quantization from min/max calibration of a float reference (range
widened to include 0, as the converter does). Writes npu-shader/models/layernorm/NAME.tflite and NAME_reference.npz
("input" without the batch dimension, "output" = the CPU interpreter's output, default delegates = XNNPACK), plus the
float output for a sanity figure.

  layernorm_model.py [--T 64] [--C 384] [--name ln_64x384_i8] [--partial OP]
      --partial OP: stop the graph after that op's output (m, sd, v, ve, r, rg, xs, ms, bms) for single-step tests
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "layernorm")
ORDER = ["m", "sd", "v", "ve", "r", "rg", "xs", "ms", "bms", "y"]


def qparams(lo, hi):
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if hi - lo < 1e-12:
        hi = lo + 1e-6
    s = (hi - lo) / 255.0
    z = int(np.clip(np.round(-128 - lo / s), -128, 127))
    return s, z


def quant(x, q):
    return np.clip(np.round(x / q[0]) + q[1], -128, 127).astype(np.int8)


def float_graph(x, gamma, beta, eps):
    t = {"x": x}
    t["m"] = x.mean(axis=-1, keepdims=True)
    t["sd"] = (x - t["m"]) ** 2
    t["v"] = t["sd"].mean(axis=-1, keepdims=True)
    t["ve"] = t["v"] + eps
    t["r"] = 1.0 / np.sqrt(t["ve"])
    t["rg"] = t["r"] * gamma
    t["xs"] = x * t["rg"]
    t["ms"] = t["m"] * t["rg"]
    t["bms"] = beta - t["ms"]
    t["y"] = t["xs"] + t["bms"]
    return t


def build(path, T, C, q, gamma_q, beta_q, eps_q, upto):
    import flatbuffers
    from ai_edge_litert import schema_py_generated as S

    I8, I32 = S.TensorType.INT8, S.TensorType.INT32
    buffers = [S.BufferT()]
    tensors, ops, codes = [], [], []

    def buf(arr):
        b = S.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(arr).tobytes(), np.uint8)
        buffers.append(b)
        return len(buffers) - 1

    def tensor(name, shp, ttype, qp=None, data=None):
        t = S.TensorT()
        t.name, t.shape, t.type = name, list(shp), ttype
        t.buffer = buf(data) if data is not None else 0
        if qp is not None:
            p = S.QuantizationParametersT()
            p.scale, p.zeroPoint = [float(qp[0])], [int(qp[1])]
            t.quantization = p
        tensors.append(t)
        return len(tensors) - 1

    def opcode(bc, version):
        for i, c in enumerate(codes):
            if c.builtinCode == bc:
                return i
        c = S.OperatorCodeT()
        c.builtinCode = c.deprecatedBuiltinCode = bc
        c.version = version
        codes.append(c)
        return len(codes) - 1

    def op(bc, version, ins, outs, otype=None, opts=None):
        o = S.OperatorT()
        o.opcodeIndex, o.inputs, o.outputs = opcode(bc, version), ins, outs
        if otype is not None:
            o.builtinOptionsType, o.builtinOptions = otype, opts
        ops.append(o)

    def act_none(cls):
        o = cls()
        o.fusedActivationFunction = S.ActivationFunctionType.NONE
        return o

    B = S.BuiltinOperator
    ro = S.ReducerOptionsT()
    ro.keepDims = True
    tid = {}
    tid["x"] = tensor("x", (1, T, C), I8, q["x"])
    axes = tensor("axes", (1,), I32, None, np.array([2], np.int32))
    shapes = {"m": (1, T, 1), "sd": (1, T, C), "v": (1, T, 1), "ve": (1, T, 1), "r": (1, T, 1), "rg": (1, T, C),
              "xs": (1, T, C), "ms": (1, T, C), "bms": (1, T, C), "y": (1, T, C)}
    for name in ORDER[: ORDER.index(upto) + 1]:
        tid[name] = tensor(name, shapes[name], I8, q[name])
    gamma = tensor("gamma", (C,), I8, gamma_q[0], gamma_q[1])
    beta = tensor("beta", (C,), I8, beta_q[0], beta_q[1])
    eps = tensor("eps", (), I8, eps_q[0], eps_q[1])
    steps = [
        ("m", lambda: op(B.MEAN, 2, [tid["x"], axes], [tid["m"]], S.BuiltinOptions.ReducerOptions, ro)),
        ("sd", lambda: op(B.SQUARED_DIFFERENCE, 2, [tid["x"], tid["m"]], [tid["sd"]],
                          S.BuiltinOptions.SquaredDifferenceOptions, S.SquaredDifferenceOptionsT())),
        ("v", lambda: op(B.MEAN, 2, [tid["sd"], axes], [tid["v"]], S.BuiltinOptions.ReducerOptions, ro)),
        ("ve", lambda: op(B.ADD, 2, [tid["v"], eps], [tid["ve"]], S.BuiltinOptions.AddOptions, act_none(S.AddOptionsT))),
        ("r", lambda: op(B.RSQRT, 2, [tid["ve"]], [tid["r"]])),
        ("rg", lambda: op(B.MUL, 2, [tid["r"], gamma], [tid["rg"]], S.BuiltinOptions.MulOptions, act_none(S.MulOptionsT))),
        ("xs", lambda: op(B.MUL, 2, [tid["x"], tid["rg"]], [tid["xs"]], S.BuiltinOptions.MulOptions, act_none(S.MulOptionsT))),
        ("ms", lambda: op(B.MUL, 2, [tid["m"], tid["rg"]], [tid["ms"]], S.BuiltinOptions.MulOptions, act_none(S.MulOptionsT))),
        ("bms", lambda: op(B.SUB, 2, [beta, tid["ms"]], [tid["bms"]], S.BuiltinOptions.SubOptions, act_none(S.SubOptionsT))),
        ("y", lambda: op(B.ADD, 2, [tid["xs"], tid["bms"]], [tid["y"]], S.BuiltinOptions.AddOptions, act_none(S.AddOptionsT))),
    ]
    for name, emit in steps:
        emit()
        if name == upto:
            break
    sub = S.SubGraphT()
    sub.tensors, sub.inputs, sub.outputs, sub.operators, sub.name = tensors, [tid["x"]], [tid[upto]], ops, "layernorm"
    m = S.ModelT()
    m.version, m.operatorCodes, m.subgraphs, m.buffers = 3, codes, [sub], buffers
    m.description = "alta step 6g int8 layernorm (Keras decomposition)"
    fb = flatbuffers.Builder(1 << 16)
    fb.Finish(m.Pack(fb), b"TFL3")
    open(path, "wb").write(bytes(fb.Output()))


def cpu_run(path, x):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=1)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--C", type=int, default=384)
    ap.add_argument("--name")
    ap.add_argument("--partial", default="y", choices=ORDER)
    ap.add_argument("--eps", type=float, default=1e-3)
    a = ap.parse_args()
    T, C = a.T, a.C
    name = a.name or ("ln_%dx%d_i8" % (T, C) + ("" if a.partial == "y" else "_upto_" + a.partial))
    rng = np.random.default_rng(1114 + T * 7 + C)
    # activations like a transformer's: per-channel offsets and spreads, a few large channels
    off = rng.normal(0, 0.5, C)
    spread = np.exp(rng.normal(0, 0.4, C))
    spread[rng.choice(C, max(1, C // 64), replace=False)] *= 4
    def sample(n):
        return (rng.normal(0, 1, (n, T, C)) * spread + off).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, C).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, C).astype(np.float32)
    cal = float_graph(sample(32), gamma, beta, a.eps)
    q = {k: qparams(v.min(), v.max()) for k, v in cal.items()}
    gq = qparams(gamma.min(), gamma.max())
    bq = qparams(beta.min(), beta.max())
    eq = qparams(0.0, a.eps)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".tflite")
    build(path, T, C, q, (gq, quant(gamma, gq)), (bq, quant(beta, bq)), (eq, quant(np.float32(a.eps), eq)), a.partial)
    xf = sample(1)
    x = quant(xf, q["x"])
    y = cpu_run(path, x)
    yf = float_graph(xf, gamma, beta, a.eps)[a.partial]
    yd = (y.astype(np.float32) - q[a.partial][1]) * q[a.partial][0]
    np.savez(os.path.join(OUT, name + "_reference.npz"), input=x[0], output=y[0], float_output=yf[0])
    err = np.abs(yd - yf)
    print("%s: %s -> %s, int8 vs float: mean abs err %.4f (output scale %.4f), max %.3f, corr %.4f" % (
        name, list(x.shape), list(y.shape), err.mean(), q[a.partial][0], err.max(),
        np.corrcoef(yd.ravel(), yf.ravel())[0, 1]))
    for k in ORDER[: ORDER.index(a.partial) + 1]:
        print("   %-4s scale %.6f zp %4d" % (k, q[k][0], q[k][1]))


if __name__ == "__main__":
    main()

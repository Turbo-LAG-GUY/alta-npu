#!/usr/bin/env python3
"""int8nn_models.py (2026-11-14, port step 6i, CPU only, nothing here touches the NPU): int8 models for the NN unit
and their uint8 twins.

etnaviv stores int8 tensors +128 (etnaviv_ml.c uploads x + 128, etna_tensor_zero_point() = zp + 128), so an int8 conv
or ADD should become exactly the hardware job of the uint8 model whose bytes and zero points are all +128 (same
scales, same int32 bias). The twin's CPU output is the int8 output + 128 (the integer arithmetic is the same; checked
below). Offline the two must give the same NN/TP configs and compressed weights (tools/nn_config_compare.sh-style
compare, int8 vs twin); every uint8 NN path used here is proven on the NPU (MobileNet, the layout convs).

Before the 6i fix the v7 weight encoder wrote int8 weight bytes unshifted next to the shifted zero point (a weight of 0
became 0 - 128) and the v7 ADD passed the raw int8 zero points to compute_bias_add().

Writes npu-shader/models/int8nn/NAME.tflite + NAME_reference.npz ("input" without batch, "output" = the CPU
interpreter's output, XNNPACK by default) for NAME_i8 and NAME_u8.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "int8nn")

# name, kind, input shape NHWC, out channels, kernel, stride, relu, per_channel
MODELS = [
    ("pw64", "conv", (1, 8, 8, 64), 64, 1, 1, False, False),    # 1x1, random weights and bias
    ("pw64r", "conv", (1, 8, 8, 64), 64, 1, 1, True, False),    # same with a fused RELU
    ("c3", "conv", (1, 16, 16, 16), 32, 3, 1, False, False),    # 3x3 SAME
    ("c3s2", "conv", (1, 16, 16, 16), 32, 3, 2, False, False),  # 3x3 SAME stride 2
    ("dw3", "dw", (1, 16, 16, 32), 32, 3, 1, False, False),     # depthwise 3x3 SAME
    ("add", "add", (1, 8, 8, 64), 64, 1, 1, False, False),      # x -> 1x1 conv -> y; ADD(x, y), different q
    ("pw64pc", "conv", (1, 8, 8, 64), 64, 1, 1, False, True),   # per-channel weights (TFLite's int8 default), no twin
]
IN_Q = (0.05, 5)   # int8 input scale, zero point (twin: zp + 128)


def conv_ref(x, w, b, stride, depthwise):
    """Float reference only for choosing output scales: x [H,W,C] real, w [O,kh,kw,I] (dw: [1,kh,kw,C]) real."""
    H, W, C = x.shape
    k = w.shape[1]
    p = (k - 1) // 2
    xp = np.pad(x, ((p, p), (p, p), (0, 0)))
    oh, ow = (H + stride - 1) // stride, (W + stride - 1) // stride
    O = C if depthwise else w.shape[0]
    y = np.zeros((oh, ow, O))
    for i in range(oh):
        for j in range(ow):
            patch = xp[i * stride:i * stride + k, j * stride:j * stride + k, :]
            if depthwise:
                y[i, j] = (patch * w[0]).sum(axis=(0, 1))
            else:
                y[i, j] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2]))
    return y + b


def qrange(y, int8):
    lo, hi = min(float(y.min()), 0.0), max(float(y.max()), 0.0)
    s = (hi - lo) / 255.0
    zp = int(round(-lo / s)) + (-128 if int8 else 0)
    return s, zp


class Graph:
    def __init__(self):
        from ai_edge_litert import schema_py_generated as S
        self.S = S
        self.buffers = [S.BufferT()]
        self.tensors, self.ops, self.codes = [], [], {}

    def buffer(self, arr):
        b = self.S.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(arr).tobytes(), np.uint8)
        self.buffers.append(b)
        return len(self.buffers) - 1

    def tensor(self, name, shp, ttype, scale, zp, data=None, qdim=0):
        S = self.S
        t = S.TensorT()
        t.name, t.shape, t.type = name, list(shp), ttype
        t.buffer = self.buffer(data) if data is not None else 0
        qp = S.QuantizationParametersT()
        qp.scale = [float(v) for v in np.atleast_1d(scale)]
        qp.zeroPoint = [int(v) for v in np.atleast_1d(zp)]
        qp.quantizedDimension = qdim
        t.quantization = qp
        self.tensors.append(t)
        return len(self.tensors) - 1

    def op(self, builtin, version, ins, outs, otype, opts):
        S = self.S
        if builtin not in self.codes:
            c = S.OperatorCodeT()
            c.builtinCode = c.deprecatedBuiltinCode = builtin
            c.version = version
            self.codes[builtin] = (len(self.codes), c)
        o = S.OperatorT()
        o.opcodeIndex = self.codes[builtin][0]
        o.inputs, o.outputs = ins, outs
        o.builtinOptionsType, o.builtinOptions = otype, opts
        self.ops.append(o)

    def save(self, path, inputs, outputs):
        import flatbuffers
        S = self.S
        sub = S.SubGraphT()
        sub.tensors, sub.inputs, sub.outputs, sub.operators, sub.name = self.tensors, inputs, outputs, self.ops, "g"
        m = S.ModelT()
        m.version = 3
        m.operatorCodes = [c for _, c in sorted(self.codes.values(), key=lambda v: v[0])]
        m.subgraphs, m.buffers = [sub], self.buffers
        m.description = "alta step 6i int8 NN test"
        b = flatbuffers.Builder(1 << 20)
        b.Finish(m.Pack(b), b"TFL3")
        with open(path, "wb") as f:
            f.write(bytes(b.Output()))


def add_conv(g, x, xq, shape, oc, k, stride, relu, depthwise, wq, wscale, wzp, bias, yq, int8, name):
    S = g.S
    T8 = S.TensorType.INT8 if int8 else S.TensorType.UINT8
    per_channel = np.ndim(wscale) > 0
    wt = g.tensor(name + "_w", wq.shape, T8, wscale, wzp, wq, qdim=3 if depthwise else 0)
    bscale = xq[0] * np.asarray(wscale)
    bt = g.tensor(name + "_b", (oc,), S.TensorType.INT32, bscale, np.zeros_like(np.atleast_1d(bscale), int), bias)
    oh, ow = (shape[1] + stride - 1) // stride, (shape[2] + stride - 1) // stride
    y = g.tensor(name + "_out", (1, oh, ow, oc), T8, yq[0], yq[1])
    act = S.ActivationFunctionType.RELU if relu else S.ActivationFunctionType.NONE
    if depthwise:
        o = S.DepthwiseConv2DOptionsT()
        o.depthMultiplier = 1
        otype, builtin = S.BuiltinOptions.DepthwiseConv2DOptions, S.BuiltinOperator.DEPTHWISE_CONV_2D
    else:
        o = S.Conv2DOptionsT()
        otype, builtin = S.BuiltinOptions.Conv2DOptions, S.BuiltinOperator.CONV_2D
    o.padding = S.Padding.SAME
    o.strideW = o.strideH = stride
    o.dilationWFactor = o.dilationHFactor = 1
    o.fusedActivationFunction = act
    g.op(builtin, 3 if int8 else 1, [x, wt, bt], [y], otype, o)
    return y


def build(path, spec, p, int8):
    """p: the int8 parameters; the twin (int8=False) shifts every byte and zero point by +128."""
    name, kind, shape, oc, k, stride, relu, pc = spec
    S = None
    g = Graph()
    S = g.S
    T8 = S.TensorType.INT8 if int8 else S.TensorType.UINT8
    sh = 0 if int8 else 128
    dt = np.int8 if int8 else np.uint8
    xq = (IN_Q[0], IN_Q[1] + sh)
    x = g.tensor("input", shape, T8, xq[0], xq[1])
    wq = (p["w"].astype(np.int16) + sh).astype(dt)
    wzp = p["wzp"] + sh
    yq = (p["yq"][0], p["yq"][1] + sh)
    if kind == "add":
        cq = (p["cq"][0], p["cq"][1] + sh)
        c = add_conv(g, x, xq, shape, oc, 1, 1, False, False, wq, p["ws"], wzp, p["b"], cq, int8, "conv")
        z = g.tensor("add_out", shape, T8, yq[0], yq[1])
        o = S.AddOptionsT()
        o.fusedActivationFunction = S.ActivationFunctionType.NONE
        g.op(S.BuiltinOperator.ADD, 2 if int8 else 1, [x, c], [z], S.BuiltinOptions.AddOptions, o)
        out = z
    else:
        out = add_conv(g, x, xq, shape, oc, k, stride, relu, kind == "dw", wq, p["ws"], wzp, p["b"], yq, int8, "conv")
    g.save(path, [x], [out])


def params(spec, rng):
    name, kind, shape, oc, k, stride, relu, pc = spec
    ic = shape[3]
    depthwise = kind == "dw"
    wshape = (1, k, k, ic) if depthwise else (oc, k, k, ic)
    w = rng.integers(-127, 128, size=wshape).astype(np.int8)
    if pc:
        ws = rng.uniform(0.002, 0.02, size=oc)
    else:
        ws = 0.01
    wzp = np.zeros(oc, int) if pc else 0
    bias = rng.integers(-2000, 2001, size=oc).astype(np.int32)
    xs = rng.integers(-128, 128, size=shape[1:]).astype(np.int8)
    xr = IN_Q[0] * (xs.astype(np.float64) - IN_Q[1])
    wr = w.astype(np.float64) * (np.asarray(ws)[:, None, None, None] if pc else ws)
    yr = conv_ref(xr, wr, bias * IN_Q[0] * np.asarray(ws), 1 if kind == "add" else stride, depthwise)
    p = {"w": w, "ws": ws, "wzp": wzp, "b": bias, "x": xs}
    if kind == "add":
        p["cq"] = qrange(yr, True)
        p["yq"] = qrange(yr + xr, True)
    else:
        p["yq"] = qrange(np.maximum(yr, 0) if relu else yr, True)
    return p


def cpu_run(path, x):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path, num_threads=1)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x[None])
    it.invoke()
    return it.get_tensor(o["index"])[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(61114)
    for spec in MODELS:
        name, pc = spec[0], spec[7]
        p = params(spec, rng)
        pi = os.path.join(OUT, name + "_i8.tflite")
        build(pi, spec, p, True)
        yi = cpu_run(pi, p["x"])
        np.savez(os.path.join(OUT, name + "_i8_reference.npz"), input=p["x"], output=yi)
        rails = np.mean((yi == -128) | (yi == 127)) * 100
        line = "%-7s i8 out %s range %d..%d (%.1f%% at the rails)" % (name, list(yi.shape), yi.min(), yi.max(), rails)
        if not pc:
            pu = os.path.join(OUT, name + "_u8.tflite")
            build(pu, spec, p, False)
            xu = (p["x"].astype(np.int16) + 128).astype(np.uint8)
            yu = cpu_run(pu, xu)
            np.savez(os.path.join(OUT, name + "_u8_reference.npz"), input=xu, output=yu)
            d = np.abs(yu.astype(np.int16) - 128 - yi.astype(np.int16))
            line += "; u8 twin - 128 vs i8: max %d, %d differ" % (d.max(), int((d > 0).sum()))
        print(line)


if __name__ == "__main__":
    sys.exit(main())

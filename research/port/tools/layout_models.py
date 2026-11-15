#!/usr/bin/env python3
"""layout_models.py (2026-11-14, CPU only, nothing here touches the NPU): single 1x1 CONV_2D models whose output is a
Teflon partition output, for the partition-output detranspose fix (etnaviv_ml.c layouts_differ: C > 1 && H*W > 1;
before, H > 1 && W > 1 && C > 1 skipped [1,1,T,C] and [1,H,1,C], whose bytes then reached the CPU in channel planes).

The conv has identity weights (as reduce_models.py --conv) but a different output scale, so the NN really requantizes;
with random inputs a channel-plane misread shows up as a mean difference of tens of levels, NN rounding as <= 1.

Writes into npu-shader/models/layout/: NAME.tflite and NAME_reference.npz ("input" without the batch dimension, as
teflon_check.py expects, "output" = the CPU interpreter's output with its default delegates = XNNPACK).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "layout")

MODELS = [
    # name, shape (NHWC), int8, in_q, out_q
    ("conv_1x64x384_u8", (1, 1, 64, 384), False, (0.05, 120), (0.04, 128)),  # [1,T,C] as Teflon sees it: H = 1
    ("conv_64x1x384_u8", (1, 64, 1, 384), False, (0.05, 120), (0.04, 128)),  # W = 1
    ("conv_1x50x96_u8", (1, 1, 50, 96), False, (0.05, 120), (0.04, 128)),    # T % 4 != 0: TP even with transpose
    ("conv_1x32x768_i8", (1, 1, 32, 768), True, (0.07, 5), (0.05, -3)),      # int8 (the conv that saturated in 6e)
    ("conv_8x8x64_u8", (1, 8, 8, 64), False, (0.05, 120), (0.04, 128)),      # 3D control: detransposed before too
    ("conv_64x8x64_u8", (1, 64, 8, 64), False, (0.05, 120), (0.04, 128)),    # 3D with H > W, W < 64 (6h tiling fix)
]


def build_conv(path, shape, int8, in_q, out_q):
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
        t.name, t.shape, t.type, t.buffer = name, list(shp), ttype, buf
        if q is not None:
            qp = S.QuantizationParametersT()
            qp.scale, qp.zeroPoint = [float(q[0])], [int(q[1])]
            t.quantization = qp
        tensors.append(t)
        return len(tensors) - 1

    c = shape[3]
    wzp = 0 if int8 else 128
    wq = np.full((c, 1, 1, c), wzp, np.int8 if int8 else np.uint8)
    for i in range(c):
        wq[i, 0, 0, i] = wzp + 100
    ws = 1.0 / 100.0
    x = tensor("input", shape, T8, in_q)
    wt = tensor("conv_w", wq.shape, T8, (ws, wzp), add_buffer(wq))
    bt = tensor("conv_b", (c,), S.TensorType.INT32, (in_q[0] * ws, 0), add_buffer(np.zeros(c, np.int32)))
    y = tensor("conv_out", shape, T8, out_q)

    code = S.OperatorCodeT()
    code.builtinCode = code.deprecatedBuiltinCode = S.BuiltinOperator.CONV_2D
    code.version = 1
    o = S.Conv2DOptionsT()
    o.padding = S.Padding.SAME
    o.strideW = o.strideH = o.dilationWFactor = o.dilationHFactor = 1
    o.fusedActivationFunction = S.ActivationFunctionType.NONE
    op = S.OperatorT()
    op.opcodeIndex = 0
    op.inputs, op.outputs = [x, wt, bt], [y]
    op.builtinOptionsType, op.builtinOptions = S.BuiltinOptions.Conv2DOptions, o

    sub = S.SubGraphT()
    sub.tensors, sub.inputs, sub.outputs, sub.operators, sub.name = tensors, [x], [y], [op], "conv"
    m = S.ModelT()
    m.version = 3
    m.operatorCodes, m.subgraphs, m.buffers = [code], [sub], buffers
    m.description = "alta detranspose-fix layout test"
    b = flatbuffers.Builder(1 << 16)
    b.Finish(m.Pack(b), b"TFL3")
    with open(path, "wb") as f:
        f.write(bytes(b.Output()))


def cpu_run(path, x):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=1)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x)
    it.invoke()
    return it.get_tensor(o["index"])


def planar_misread(y):
    """What the CPU sees when the NN output stays in channel planes: bytes [C][H][W] read as [H][W][C]."""
    n, h, w, c = y.shape
    return np.ascontiguousarray(y[0].transpose(2, 0, 1)).reshape(1, h, w, c)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1114)
    for name, shape, int8, iq, oq in MODELS:
        path = os.path.join(OUT, name + ".tflite")
        build_conv(path, shape, int8, iq, oq)
        lo, hi = (-128, 127) if int8 else (0, 255)
        x = rng.integers(lo, hi + 1, size=shape).astype(np.int8 if int8 else np.uint8)
        y = cpu_run(path, x)
        np.savez(os.path.join(OUT, name + "_reference.npz"), input=x[0], output=y[0])
        mis = np.abs(planar_misread(y).astype(np.int32) - y.astype(np.int32)).mean()
        print("%-18s %s range %d..%d, a planar misread would differ by mean %.1f levels"
              % (name, list(shape), int(y.min()), int(y.max()), mis))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""A tiny uint8 model where an NPU layer's output is also read by a CPU layer (limit 2 of the AXI SRAM).
CPU only. conv1 -> t1 -> conv2 -> conv3 -> out_a; t1 -> TANH (not delegated, runs on the CPU) -> out_b.
Teflon makes one partition of the three convs whose outputs are t1 and out_a, and t1 is also read inside it.
usage: make_branch_model.py OUT.tflite [--exact]
--exact: the CPU layer is MAXIMUM(t1, t1), an exact copy of t1, listed as the first output (for teflon_check.py,
which checks output 0)."""
import sys
import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as S

H = W = 32
C = 16
rng = np.random.default_rng(7)
buffers, tensors = [S.BufferT()], []


def add_buffer(arr):
    b = S.BufferT()
    b.data = np.frombuffer(arr.tobytes(), np.uint8)
    buffers.append(b)
    return len(buffers) - 1


def tensor(name, shape, ttype, scale, zero, buf=0):
    t = S.TensorT()
    t.name, t.shape, t.type, t.buffer = name, list(shape), ttype, buf
    q = S.QuantizationParametersT()
    q.scale, q.zeroPoint = [float(scale)], [int(zero)]
    t.quantization = q
    tensors.append(t)
    return len(tensors) - 1


U8, I32 = S.TensorType.UINT8, S.TensorType.INT32
ops = []


def conv(name, src, s_in, s_out):
    wq = np.clip(np.rint(rng.normal(128, 12, (C, 3, 3, C))), 0, 255).astype(np.uint8)
    bq = np.zeros(C, np.int32)
    wt = tensor(name + "_w", wq.shape, U8, 1 / 128, 128, add_buffer(wq))
    bt = tensor(name + "_b", bq.shape, I32, s_in / 128, 0, add_buffer(bq))
    ot = tensor(name, (1, H, W, C), U8, s_out, 128)
    o = S.Conv2DOptionsT()
    o.padding, o.strideW, o.strideH, o.dilationWFactor, o.dilationHFactor = S.Padding.SAME, 1, 1, 1, 1
    o.fusedActivationFunction = S.ActivationFunctionType.NONE
    op = S.OperatorT()
    op.opcodeIndex, op.inputs, op.outputs = 0, [src, wt, bt], [ot]
    op.builtinOptionsType, op.builtinOptions = S.BuiltinOptions.Conv2DOptions, o
    ops.append(op)
    return ot


x = tensor("input", (1, H, W, C), U8, 1 / 128, 128)
t1 = conv("t1", x, 1 / 128, 0.05)
t2 = conv("t2", t1, 0.05, 0.1)
out_a = conv("out_a", t2, 0.1, 0.2)
EXACT = "--exact" in sys.argv
out_b = tensor("out_b", (1, H, W, C), U8, 0.05 if EXACT else 1 / 128, 128)
op = S.OperatorT()
op.opcodeIndex, op.inputs, op.outputs = 1, [t1, t1] if EXACT else [t1], [out_b]
ops.append(op)

codes = []
for bc in (S.BuiltinOperator.CONV_2D, S.BuiltinOperator.MAXIMUM if EXACT else S.BuiltinOperator.TANH):
    c = S.OperatorCodeT()
    c.builtinCode = c.deprecatedBuiltinCode = bc
    c.version = 1
    codes.append(c)
sub = S.SubGraphT()
sub.tensors, sub.inputs, sub.outputs, sub.operators, sub.name = tensors, [x], [out_b, out_a] if EXACT else [out_a, out_b], ops, "branch"
m = S.ModelT()
m.version, m.operatorCodes, m.subgraphs, m.buffers = 3, codes, [sub], buffers
m.description = "sram-limits: NPU output also read by a CPU layer"
b = flatbuffers.Builder(1 << 16)
b.Finish(m.Pack(b), b"TFL3")
open(sys.argv[1], "wb").write(bytes(b.Output()))

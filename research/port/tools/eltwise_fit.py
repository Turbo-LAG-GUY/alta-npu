#!/usr/bin/env python3
"""eltwise_fit.py (step 6g, 2026-11-14, CPU only, nothing here touches the NPU): which CPU arithmetic the layernorm
ops RSQRT, MUL and SQUARED_DIFFERENCE use for 8-bit tensors, so the shader core can reproduce it exactly (as
mean_fit_xnnpack.py did for MEAN).

Every op is run EXHAUSTIVELY: unary ops on all 256 input values, binary ops on all 65536 pairs (A[i][j] = i, B = j as
two [1,1,256,256] inputs), each with random quantization parameters, through the interpreter with its default
delegates (XNNPACK: what runs beside Teflon) and with the builtin kernels only, and compared with candidate formulas.

  eltwise_fit.py [--trials N] [--ops rsqrt,mul,sqdiff] [--uint8]
"""
import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reduce_models as R  # noqa: E402  (quantize_multiplier, mul_by_qm)

f32 = np.float32


def build(path, op, n_inputs, int8, in_qs, out_q, in_shapes, out_shape):
    import flatbuffers
    from ai_edge_litert import schema_py_generated as S

    T8 = S.TensorType.INT8 if int8 else S.TensorType.UINT8
    tensors = []

    def tensor(name, shp, q):
        t = S.TensorT()
        t.name, t.shape, t.type, t.buffer = name, list(shp), T8, 0
        qp = S.QuantizationParametersT()
        qp.scale, qp.zeroPoint = [float(q[0])], [int(q[1])]
        t.quantization = qp
        tensors.append(t)
        return len(tensors) - 1

    ins = [tensor("in%d" % i, in_shapes[i], in_qs[i]) for i in range(n_inputs)]
    out = tensor("out", out_shape, out_q)
    code = S.OperatorCodeT()
    b = {"rsqrt": S.BuiltinOperator.RSQRT, "mul": S.BuiltinOperator.MUL,
         "sqdiff": S.BuiltinOperator.SQUARED_DIFFERENCE, "add": S.BuiltinOperator.ADD,
         "sub": S.BuiltinOperator.SUB}[op]
    code.builtinCode = code.deprecatedBuiltinCode = b
    code.version = 2 if int8 else 1
    o = S.OperatorT()
    o.opcodeIndex, o.inputs, o.outputs = 0, ins, [out]
    if op == "mul":
        mo = S.MulOptionsT()
        mo.fusedActivationFunction = S.ActivationFunctionType.NONE
        o.builtinOptionsType, o.builtinOptions = S.BuiltinOptions.MulOptions, mo
    elif op == "add":
        ao = S.AddOptionsT()
        ao.fusedActivationFunction = S.ActivationFunctionType.NONE
        o.builtinOptionsType, o.builtinOptions = S.BuiltinOptions.AddOptions, ao
    elif op == "sub":
        so_ = S.SubOptionsT()
        so_.fusedActivationFunction = S.ActivationFunctionType.NONE
        o.builtinOptionsType, o.builtinOptions = S.BuiltinOptions.SubOptions, so_
    elif op == "sqdiff":
        o.builtinOptionsType, o.builtinOptions = S.BuiltinOptions.SquaredDifferenceOptions, S.SquaredDifferenceOptionsT()
    sub = S.SubGraphT()
    sub.tensors, sub.inputs, sub.outputs, sub.operators = tensors, ins, [out], [o]
    m = S.ModelT()
    m.version, m.operatorCodes, m.subgraphs, m.buffers = 3, [code], [sub], [S.BufferT()]
    fb = flatbuffers.Builder(1 << 12)
    fb.Finish(m.Pack(fb), b"TFL3")
    open(path, "wb").write(bytes(fb.Output()))


def run(path, xs, builtin_only):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    kw = {"experimental_op_resolver_type": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES} if builtin_only else {}
    it = Interpreter(model_path=path, num_threads=1, **kw)
    it.allocate_tensors()
    for d, x in zip(it.get_input_details(), xs):
        it.set_tensor(d["index"], x)
    try:
        it.invoke()
    except RuntimeError as e:
        return str(e).splitlines()[0]
    return it.get_tensor(it.get_output_details()[0]["index"]).astype(np.int64)


def rne(v):
    return np.round(v)  # numpy rounds half to even = nearbyint/lrint


def clampq(v, lo, hi):
    return np.clip(v, lo, hi).astype(np.int64)


# --- candidates ---------------------------------------------------------------------------------------------------

def cand_rsqrt(x, iq, oq, lo, hi):
    si, zi = iq
    so, zo = oq
    d = (x - zi)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        # float LUT: dequantize, 1/sqrt in float32, requantize (XNNPACK-style unary LUT)
        v = f32(1.0) / np.sqrt(d.astype(f32) * f32(si))
        out["f32_div_so"] = v / f32(so)
        out["f32_mul_inv_so"] = v * (f32(1.0) / f32(so))
        v64 = 1.0 / np.sqrt(d * si)
        out["f64"] = v64 / so
    res = {}
    for k, v in out.items():
        q = np.where(np.isfinite(v), rne(np.nan_to_num(v, posinf=1e9, neginf=-1e9)) + zo, hi)
        res[k] = clampq(q, lo, hi)
    # TFLite LUTPopulate: dequantized = si * (x - zi); y = 1/sqrt(dequantized); rescaled = round_away(y * (1/so));
    # q = (int32)(rescaled + zo) (aarch64: NaN -> 0, +-inf saturate); clamp
    with np.errstate(divide="ignore", invalid="ignore"):
        deq = f32(si) * d.astype(f32)
        y = f32(1.0) / np.sqrt(deq)
        r = y * (f32(1.0) / f32(so))
        ra = np.where(r >= 0, np.floor(r + f32(0.5)), np.ceil(r - f32(0.5))).astype(f32)  # std::round in float
        v = ra + f32(zo)
    qi = np.where(np.isnan(v), 0, np.where(v > 2**31 - 1, 2**31 - 1, np.where(v < -2**31, -2**31, np.nan_to_num(v))))
    res["lutpopulate"] = clampq(qi.astype(np.int64), lo, hi)
    return res


def cand_mul(a, b, qa, qb, qo, lo, hi):
    acc = (a - qa[1]) * (b - qb[1])
    res = {}
    # XNNPACK qs8/qu8 vmul fp32 requantization: scale = (sa * sb) / so in float32
    s1 = f32(f32(qa[0]) * f32(qb[0])) / f32(qo[0])
    res["fp32_(ab)/o"] = clampq(rne(acc.astype(f32) * s1) + qo[1], lo, hi)
    s2 = f32(qa[0]) * (f32(qb[0]) / f32(qo[0]))
    res["fp32_a(b/o)"] = clampq(rne(acc.astype(f32) * s2) + qo[1], lo, hi)
    s3 = f32(qa[0] * qb[0] / qo[0])
    res["fp32_from_f64"] = clampq(rne(acc.astype(f32) * s3) + qo[1], lo, hi)
    # TFLite builtin: integer QuantizeMultiplier(sa*sb/so in double)
    m, sh = R.quantize_multiplier(qa[0] * qb[0] / qo[0])
    res["int_qm"] = clampq(R.mul_by_qm(acc, m, sh) + qo[1], lo, hi)
    return res


def mul_by_qm_smaller_than_one_exp(x, m, shift):
    """TFLite MultiplyByQuantizedMultiplierSmallerThanOneExp: RoundingDivideByPOT(SRDHM(x, m), -shift)."""
    return R.mul_by_qm(x, m, shift)  # same arithmetic for shift <= 0


def cand_sqdiff(a, b, qa, qb, qo, lo, hi, int8):
    res = {}
    # float: (sa*(a-za) - sb*(b-zb))^2 / so
    da = (a - qa[1]).astype(f32) * f32(qa[0])
    db = (b - qb[1]).astype(f32) * f32(qb[0])
    d = da - db
    res["f32_sq_div_so"] = clampq(rne((d * d) / f32(qo[0])) + qo[1], lo, hi)
    res["f32_sq_mul_inv_so"] = clampq(rne((d * d) * (f32(1) / f32(qo[0]))) + qo[1], lo, hi)
    # TFLite builtin squared_difference.cc (int8/uint8): left_shift 7 (int8) / 20 (?), rescale both inputs to
    # 2*max(sa, sb), integer square, requantize
    for ls in (7, 20):
        twice_max = 2.0 * max(qa[0], qb[0])
        ma, sha = R.quantize_multiplier(qa[0] / twice_max)
        mb, shb = R.quantize_multiplier(qb[0] / twice_max)
        mo, sho = R.quantize_multiplier(twice_max * twice_max / ((1 << (2 * ls)) * qo[0]))
        xa = mul_by_qm_smaller_than_one_exp((a - qa[1]) * (1 << ls), ma, sha)
        xb = mul_by_qm_smaller_than_one_exp((b - qb[1]) * (1 << ls), mb, shb)
        dd = xa - xb
        res["int_ls%d" % ls] = clampq(R.mul_by_qm(dd * dd, mo, sho) + qo[1], lo, hi)
    return res


def xnn_add_params(sa, sb, so):
    """XNNPACK xnn_init_qs8_add_minmax_scalar_params: multipliers with 20 bits, shift from the larger scale ratio."""
    import struct
    fa, fb = f32(f32(sa) / f32(so)), f32(f32(sb) / f32(so))
    mx = max(abs(fa), abs(fb))
    bits = struct.unpack("<I", struct.pack("<f", mx))[0]
    shift = 20 - ((bits >> 23) - 127)
    def scaled(x):
        b2 = struct.unpack("<I", struct.pack("<f", x))[0] + (shift << 23)
        return int(np.round(struct.unpack("<f", struct.pack("<I", b2 & 0xffffffff))[0]))
    return scaled(fa), scaled(fb), shift


def cand_add(a, b, qa, qb, qo, lo, hi, sub):
    res = {}
    sgn = -1 if sub else 1
    fa = (a - qa[1]).astype(f32) * f32(qa[0])
    fb = (b - qb[1]).astype(f32) * f32(qb[0])
    res["f32"] = clampq(rne((fa + sgn * fb) / f32(qo[0])) + qo[1], lo, hi)
    ma, mb, shift = xnn_add_params(qa[0], qb[0], qo[0])
    mb *= sgn
    bias = (1 << (shift - 1)) - ma * qa[1] - mb * qb[1]
    acc = bias + a * ma + b * mb
    res["xnn_vadd"] = clampq((acc >> shift) + qo[1], lo, hi)
    # TFLite builtin add.cc/sub.cc (8-bit): left_shift 20, twice_max = 2*max(sa, sb)
    ls = 20
    twice_max = 2.0 * max(qa[0], qb[0])
    m1, s1 = R.quantize_multiplier(qa[0] / twice_max)
    m2, s2 = R.quantize_multiplier(qb[0] / twice_max)
    mo, sho = R.quantize_multiplier(twice_max / ((1 << ls) * qo[0]))
    x1 = R.mul_by_qm((a - qa[1]) * (1 << ls), m1, s1)
    x2 = R.mul_by_qm((b - qb[1]) * (1 << ls), m2, s2)
    res["int_ls20"] = clampq(R.mul_by_qm(x1 + sgn * x2, mo, sho) + qo[1], lo, hi)
    return res


def rand_q(rng, int8, lo_s=0.005, hi_s=0.2):
    s = float(np.exp(rng.uniform(np.log(lo_s), np.log(hi_s))))
    z = int(rng.integers(-128, 128)) if int8 else int(rng.integers(0, 256))
    return (s, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--ops", default="rsqrt,mul,sqdiff")
    ap.add_argument("--uint8", action="store_true")
    ap.add_argument("--bcast", action="store_true", help="binary ops: B as [1,1,1,256] broadcast over the rows")
    ap.add_argument("--rsqrt-full", action="store_true", help="RSQRT: all 256 inputs, also those below the zero point")
    a = ap.parse_args()
    int8 = not a.uint8
    lo, hi = (-128, 127) if int8 else (0, 255)
    dt = np.int8 if int8 else np.uint8
    rng = np.random.default_rng(1114)
    tmp = tempfile.mkdtemp(prefix="eltfit_", dir=os.environ.get("TMPDIR", "/tmp"))
    grid_a = np.repeat(np.arange(lo, hi + 1), 256).reshape(1, 1, 256, 256)
    grid_b = np.tile(np.arange(lo, hi + 1), 256).reshape(1, 1, 256, 256)
    for op in a.ops.split(","):
        tot, agree, errors = {}, [0, 0], []
        for t in range(a.trials):
            p = os.path.join(tmp, "%s_%d.tflite" % (op, t))
            if op == "rsqrt":
                iq = rand_q(rng, int8, 0.0005, 0.05)
                # inputs below the zero point are invalid for RSQRT (the builtin kernel rejects them); test those
                # >= zp only, but the full range in a second model
                oq = rand_q(rng, int8, 0.01, 0.5)
                build(p, op, 1, int8, [iq], oq, [(1, 1, 1, 256)], (1, 1, 1, 256))
                x = np.arange(lo, hi + 1).reshape(1, 1, 1, 256)
                valid = (x >= iq[1]) | a.rsqrt_full
                xs = [np.where(valid, x, iq[1]).astype(dt)]
                cands = cand_rsqrt(xs[0].astype(np.int64), iq, oq, lo, hi)
                desc = "in %.5f/%d out %.4f/%d" % (iq[0], iq[1], oq[0], oq[1])
            else:
                qa, qb, qo = rand_q(rng, int8), rand_q(rng, int8), rand_q(rng, int8, 0.005, 2.0)
                bshape = (1, 1, 1, 256) if a.bcast else (1, 1, 256, 256)
                build(p, op, 2, int8, [qa, qb], qo, [(1, 1, 256, 256), bshape], (1, 1, 256, 256))
                xs = [grid_a.astype(dt), (grid_b[:, :, :1, :] if a.bcast else grid_b).astype(dt)]
                A, B = grid_a.astype(np.int64), grid_b.astype(np.int64)
                if op == "mul":
                    cands = cand_mul(A, B, qa, qb, qo, lo, hi)
                elif op == "sqdiff":
                    cands = cand_sqdiff(A, B, qa, qb, qo, lo, hi, int8)
                else:
                    cands = cand_add(A, B, qa, qb, qo, lo, hi, op == "sub")
                desc = "a %.4f/%d b %.4f/%d out %.4f/%d" % (qa[0], qa[1], qb[0], qb[1], qo[0], qo[1])
            got = run(p, xs, False)
            bi = run(p, xs, True)
            if isinstance(got, str) or isinstance(bi, str):
                errors.append("%s: default=%s builtin=%s" % (desc, got if isinstance(got, str) else "ok",
                                                             bi if isinstance(bi, str) else "ok"))
                continue
            agree[0] += int((got == bi).sum())
            agree[1] += got.size
            line = []
            for k, v in cands.items():
                e = tot.setdefault(k, [0, 0, 0, 0])
                e[0] += int((v == got).sum()); e[1] += got.size
                e[2] += int((v == bi).sum()); e[3] += bi.size
                line.append("%s %s/%s" % (k, "=" if (v == got).all() else int((v != got).sum()),
                                          "=" if (v == bi).all() else int((v != bi).sum())))
            print("%-7s %-40s default==builtin %s | %s" % (op, desc, "yes" if (got == bi).all() else
                  "NO (%d differ)" % int((got != bi).sum()), "  ".join(line)))
        print("%s: default vs builtin agree on %d/%d" % (op, agree[0], agree[1]))
        for k, (g1, n1, g2, n2) in tot.items():
            print("  %-18s vs default (XNNPACK path) %s   vs builtin %s" % (
                k, "EXACT" if g1 == n1 else "%d/%d" % (g1, n1), "EXACT" if g2 == n2 else "%d/%d" % (g2, n2)))
        for e in errors:
            print("  error:", e)


if __name__ == "__main__":
    main()

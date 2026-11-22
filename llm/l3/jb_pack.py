#!/usr/bin/env python3
"""jb_pack.py (2026-11-19, LLM engine L3b, CPU only): write "job packs" for jb_run (NN jobs through the Gallium ML API
directly, no TFLite).

A pack = the tensors and 1x1-conv operations of one subgraph (= one NPU submit), plus a test input and the expected
output(s). Format (little-endian), file PACK.jbp:
  "JBP1" u32 n_tensors u32 n_ops
  tensor: i32 index, u32 dims[4] (N H W C), f32 scale, i32 zero_point, u8 is_signed, u8 is_constant, u8 type_size,
          u8 is_output, u32 n_bytes, n_bytes of data (constants only)
  op:     u32 type, i32 in, i32 a, i32 b, i32 out; type 1 = 1x1 conv (a = weight, b = bias or -1), type 2 = TRANSPOSE
          (a = perm as 4 bytes p0..p3). Graphs are planar like the L1c models: a channels-first input [1, C, H, W] and
          an explicit transpose to channels-last, and an explicit transpose back in front of every output, which the
          driver turns into bypasses (no TP job; without them it adds an implicit TP transpose per input).
  then the graph input (u32 n_bytes + bytes) and for each output tensor (in tensor order) u32 n_bytes + expected bytes.

Modes:
  tflite MODEL.tflite REF.npz OUT.jbp     the CONV_2D of an L1c model, weights and reference unchanged (byte-identity
                                          check against Teflon's offline dumps of the same model)
  groups MODEL.tflite REF.npz G OUT.jbp   the same conv split into G jobs over contiguous output-channel groups, each
                                          with its own weight scale (channel groups, L2e) and its own output
  chain MODEL.tflite REF.npz OUT.jbp      the conv followed by a second, dependent conv OC -> IC (synthetic weights):
                                          two NN jobs in one submit, the second reading the first's output
  altinput PACK.jbp OUT.jbp [SEED]         PACK with its input values permuted and the matching expected outputs
                                          (numpy reference of the graph, checked against PACK's own): jb_run --alt
  synthetic IC OC G OUT.jbp               random int8 weights IC -> OC (the NPU's share of a split step), G jobs over
                                          contiguous output-channel groups, each with its own scale; one token
  synth IC OCJ G H W LAYOUT OUT.jbp       (2026-11-22, AXI SRAM test) G jobs of OCJ output channels each on an H x W
                                          token image, random int8 weights / input; LAYOUT planar = the usual planar
                                          input + bypass transpose, nhwc = a channels-last graph input read by every
                                          conv (the driver adds the input transpose job; ETNA_ML_SHARE_INPUT_TRANSPOSE=1
                                          makes it one, ETNA_ML_AXI_SRAM=all can put its output in the AXI SRAM)
Expected outputs of groups/chain come from a float reference of the int8 arithmetic (<= 1 level off TFLite).
"""
import struct
import sys

import numpy as np


def load_conv(model, ref):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=model)
    it.allocate_tensors()
    det = {d["index"]: d for d in it.get_tensor_details()}
    conv = [o for o in it._get_ops_details() if o["op_name"] == "CONV_2D"]
    assert len(conv) == 1, "expected one CONV_2D"
    i, w, b = conv[0]["inputs"]
    o = conv[0]["outputs"][0]
    r = np.load(ref)
    q = lambda k: det[k]["quantization"]
    return dict(x=r["input"].reshape(-1).astype(np.int8), y=r["output"].reshape(-1).astype(np.int8),
                w=it.get_tensor(w).reshape(det[w]["shape"][0], -1).astype(np.int8), b=it.get_tensor(b).astype(np.int32),
                xq=q(i), wq=q(w), yq=q(o), in_shape=list(det[i]["shape"]), out_shape=list(det[o]["shape"]))


def ref_conv(x, xq, w, wq, b, yq):
    acc = (x.astype(np.int64) - xq[1]) @ (w.astype(np.int64) - wq[1]).T + b
    return np.clip(np.round(acc * (xq[0] * wq[0] / yq[0])) + yq[1], -128, 127).astype(np.int8)


class Pack:
    def __init__(self):
        self.t, self.ops = [], []

    def tensor(self, dims, q, signed=True, data=None, out=False, size=1):
        idx = len(self.t)
        self.t.append((idx, list(dims), float(q[0]), int(q[1]), signed, data is not None, size, out,
                       b"" if data is None else np.ascontiguousarray(data).tobytes()))
        return idx

    def conv(self, i, w, b, o):
        self.ops.append((1, i, w, b, o))

    def transpose(self, i, o, perm):
        self.ops.append((2, i, perm[0] | perm[1] << 8 | perm[2] << 16 | perm[3] << 24, -1, o))

    def planar_in(self, H, W, C, q):
        xp = self.tensor([1, C, H, W], q)
        x = self.tensor([1, H, W, C], q)
        self.transpose(xp, x, (0, 2, 3, 1))
        return x

    def planar_out(self, y, H, W, C, q):
        yp = self.tensor([1, C, H, W], q, out=True)
        self.transpose(y, yp, (0, 3, 1, 2))

    def write(self, path, x, expected):
        with open(path, "wb") as f:
            f.write(b"JBP1" + struct.pack("<II", len(self.t), len(self.ops)))
            for idx, dims, s, zp, sg, const, size, out, data in self.t:
                f.write(struct.pack("<i4IfiBBBBI", idx, *dims, s, zp, sg, const, size, out, len(data)) + data)
            for op in self.ops:
                f.write(struct.pack("<Iiiii", *op))
            f.write(struct.pack("<I", x.nbytes) + x.tobytes())
            for e in expected:
                f.write(struct.pack("<I", e.nbytes) + e.tobytes())
        print("%s: %d tensors, %d ops, %d weight bytes" % (path, len(self.t), len(self.ops),
                                                            sum(len(t[8]) for t in self.t)))


def synthetic(IC, OC, G, out):
    rng = np.random.default_rng(IC * 100003 + OC)
    xq = (0.05, 0)
    x = rng.integers(-128, 128, size=IC).astype(np.int8)
    p = Pack()
    xi = p.planar_in(1, 1, IC, xq)
    exp = []
    for grp in np.array_split(np.arange(OC), G):
        n = len(grp)
        w = rng.integers(-127, 128, size=(n, IC)).astype(np.int8)
        sw = float(rng.uniform(0.5, 2.0)) / 127.0 / np.sqrt(IC)
        b = rng.integers(-3000, 3000, size=n).astype(np.int32)
        acc = (x.astype(np.float64) - xq[1]) @ w.astype(np.float64).T + b
        sy = float(np.abs(acc * xq[0] * sw).max() / 120.0)
        wi = p.tensor([n, 1, 1, IC], (sw, 0), data=w)
        bi = p.tensor([1, 1, 1, n], (xq[0] * sw, 0), data=b, size=4)
        yi = p.tensor([1, 1, 1, n], (sy, 0))
        p.conv(xi, wi, bi, yi)
        p.planar_out(yi, 1, 1, n, (sy, 0))
        exp.append(ref_conv(x.reshape(1, -1), xq, w, (sw, 0), b, (sy, 0)).reshape(-1))
    p.write(out, x, exp)


def synth(IC, OCJ, G, H, W, layout, out):
    """G jobs x OCJ output channels on an H x W token image (expected outputs from run_graph, the int8 reference)."""
    rng = np.random.default_rng(IC * 1009 + OCJ * 17 + G)
    xq = (0.05, 0)
    p = Pack()
    if layout == "planar":
        xi = p.planar_in(H, W, IC, xq)
    elif layout == "nhwc":
        xi = p.tensor([1, H, W, IC], xq)
    else:
        sys.exit("layout: planar | nhwc")
    for g in range(G):
        w = rng.integers(-127, 128, size=(OCJ, IC)).astype(np.int8)
        sw = float(rng.uniform(0.5, 2.0)) / 127.0 / np.sqrt(IC)
        sy = 4.0 * 74.0 * 73.0 * np.sqrt(IC) * xq[0] * sw / 120.0  # 4 sigma of the accumulator (uniform x, w) = 120 levels
        wi = p.tensor([OCJ, 1, 1, IC], (sw, 0), data=w)
        bi = p.tensor([1, 1, 1, OCJ], (xq[0] * sw, 0), data=np.zeros(OCJ, np.int32), size=4)
        yi = p.tensor([1, H, W, OCJ], (sy, 0))
        p.conv(xi, wi, bi, yi)
        p.planar_out(yi, H, W, OCJ, (sy, 0))
    x = rng.integers(-128, 128, size=IC * H * W).astype(np.int8)  # bytes in the graph input's own layout
    exp = run_graph(p_tensors(p), p.ops, x.tobytes())
    sat = np.mean(np.concatenate([np.abs(e.astype(int)) >= 127 for e in exp]))
    print("synth %s: saturated outputs %.2f%%" % (layout, 100 * sat))
    p.write(out, x, exp)


def p_tensors(p):
    """Pack.t records -> the dicts read_pack returns (for run_graph)."""
    return [dict(idx=idx, dims=dims, q=(s, zp), signed=sg, const=const, out=o, data=data)
            for idx, dims, s, zp, sg, const, size, o, data in p.t]


TREC = "<i4IfiBBBBI"   # one tensor record before its data (36 bytes)


def read_pack(path):
    """-> tensors (dicts), ops (tuples), input bytes, expected outputs (bytes, tensor order), header + body bytes."""
    b = open(path, "rb").read()
    assert b[:4] == b"JBP1", path
    nt, nops = struct.unpack_from("<II", b, 4)
    o, T = 12, []
    for _ in range(nt):
        idx, d0, d1, d2, d3, s, zp, sg, const, size, out, n = struct.unpack_from(TREC, b, o)
        o += struct.calcsize(TREC)
        T.append(dict(idx=idx, dims=[d0, d1, d2, d3], q=(s, zp), signed=sg, const=const, out=out, data=b[o:o + n]))
        o += n
    ops = [struct.unpack_from("<Iiiii", b, o + 20 * k) for k in range(nops)]
    o += 20 * nops
    body = b[:o]
    n = struct.unpack_from("<I", b, o)[0]
    x = b[o + 4:o + 4 + n]
    o += 4 + n
    exp = []
    for t in T:
        if t["out"]:
            n = struct.unpack_from("<I", b, o)[0]
            exp.append(b[o + 4:o + 4 + n])
            o += 4 + n
    return T, ops, x, exp, body


def run_graph(T, ops, x):
    """numpy reference of a pack's graph (planar transposes + 1x1 convs, signed int8) -> outputs in tensor order."""
    written = {op[4] for op in ops}
    inp = [t for t in T if not t["const"] and t["idx"] not in written]
    assert len(inp) == 1, "one graph input expected"
    val = {inp[0]["idx"]: np.frombuffer(x, np.int8).reshape(inp[0]["dims"])}
    for typ, i, a, b, o in ops:
        if typ == 2:
            perm = [(a >> (8 * k)) & 0xff for k in range(4)]
            val[o] = np.ascontiguousarray(val[i].transpose(perm)).reshape(T[o]["dims"])
            continue
        xi = val[i]                                    # [1, H, W, IC]
        H, W, IC = xi.shape[1], xi.shape[2], xi.shape[3]
        w = np.frombuffer(T[a]["data"], np.int8).reshape(T[a]["dims"][0], -1)
        bias = np.frombuffer(T[b]["data"], np.int32) if b >= 0 else np.zeros(w.shape[0], np.int32)
        y = ref_conv(xi.reshape(H * W, IC), T[i]["q"], w, T[a]["q"], bias, T[o]["q"])
        val[o] = y.reshape(1, H, W, -1)
    return [np.ascontiguousarray(val[t["idx"]]).reshape(-1).astype(np.int8) for t in T if t["out"]]


def altinput(src, out, seed=1):
    """The same pack with its input values permuted (same distribution, different outputs) and the matching
    expected outputs: jb_run --alt alternates the two and checks every invoke (stale output reads)."""
    T, ops, x, exp, body = read_pack(src)
    check = run_graph(T, ops, x)
    d0 = max(int(np.abs(c.astype(int) - np.frombuffer(e, np.int8).astype(int)).max()) for c, e in zip(check, exp))
    if d0 > 1:
        sys.exit("%s: the reference gives %d levels off the stored expected output" % (src, d0))
    x2 = np.random.default_rng(seed).permutation(np.frombuffer(x, np.int8)).astype(np.int8)
    exp2 = run_graph(T, ops, x2.tobytes())
    changed = np.mean(np.concatenate([np.abs(a.astype(int) - np.frombuffer(e, np.int8).astype(int)) > 1
                                      for a, e in zip(exp2, exp)]))
    with open(out, "wb") as f:
        f.write(body + struct.pack("<I", x2.nbytes) + x2.tobytes())
        for e in exp2:
            f.write(struct.pack("<I", e.nbytes) + e.tobytes())
    print("%s: reference vs stored expected max %d; %.1f%% of the output bytes differ by more than 1 from %s" % (
        out, d0, 100 * changed, src))


def main():
    mode = sys.argv[1]
    if mode == "altinput":
        return altinput(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    if mode == "synth":
        a = sys.argv[2:]
        return synth(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), a[5], a[6])
    if mode == "synthetic":
        return synthetic(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    c = load_conv(sys.argv[2], sys.argv[3])
    OC, IC = c["w"].shape
    H, W = c["in_shape"][1], c["in_shape"][2]
    p = Pack()
    xi = p.planar_in(H, W, IC, c["xq"])
    if mode == "tflite":
        wi = p.tensor([OC, 1, 1, IC], c["wq"], data=c["w"])
        bi = p.tensor([1, 1, 1, OC], (c["xq"][0] * c["wq"][0], 0), data=c["b"], size=4)
        yi = p.tensor([1, H, W, OC], c["yq"])
        p.conv(xi, wi, bi, yi)
        p.planar_out(yi, H, W, OC, c["yq"])
        p.write(sys.argv[4], c["x"], [c["y"]])
    elif mode == "groups":
        G = int(sys.argv[4])
        wf = c["w"].astype(np.float64) * c["wq"][0]
        bf = c["b"].astype(np.float64) * c["xq"][0] * c["wq"][0]
        exp = []
        for grp in np.array_split(np.arange(OC), G):
            sw = np.abs(wf[grp]).max() / 127.0
            wq = np.clip(np.round(wf[grp] / sw), -127, 127).astype(np.int8)
            bq = np.round(bf[grp] / (c["xq"][0] * sw)).astype(np.int32)
            wi = p.tensor([len(grp), 1, 1, IC], (sw, 0), data=wq)
            bi = p.tensor([1, 1, 1, len(grp)], (c["xq"][0] * sw, 0), data=bq, size=4)
            yi = p.tensor([1, H, W, len(grp)], c["yq"])
            p.conv(xi, wi, bi, yi)
            p.planar_out(yi, H, W, len(grp), c["yq"])
            exp.append(ref_conv(c["x"].reshape(1, -1), c["xq"], wq, (sw, 0), bq, c["yq"]).reshape(-1))
        p.write(sys.argv[5], c["x"], exp)
    elif mode == "chain":
        wi = p.tensor([OC, 1, 1, IC], c["wq"], data=c["w"])
        bi = p.tensor([1, 1, 1, OC], (c["xq"][0] * c["wq"][0], 0), data=c["b"], size=4)
        mi = p.tensor([1, H, W, OC], c["yq"])
        p.conv(xi, wi, bi, mi)
        rng = np.random.default_rng(7)
        w2 = rng.integers(-127, 128, size=(IC, OC), dtype=np.int64).astype(np.int8)
        sw2 = 1.0 / 127.0 / np.sqrt(OC)
        b2 = rng.integers(-2000, 2000, size=IC).astype(np.int32)
        h = ref_conv(c["x"].reshape(1, -1), c["xq"], c["w"], c["wq"], c["b"], c["yq"]).reshape(-1)
        acc = (h.astype(np.float64) - c["yq"][1]) @ (w2.astype(np.float64)).T + b2
        sy2 = max(np.abs(acc * c["yq"][0] * sw2).max() / 127.0, 1e-6)
        y2 = ref_conv(h.reshape(1, -1), c["yq"], w2, (sw2, 0), b2, (sy2, 0)).reshape(-1)
        w2i = p.tensor([IC, 1, 1, OC], (sw2, 0), data=w2)
        b2i = p.tensor([1, 1, 1, IC], (c["yq"][0] * sw2, 0), data=b2, size=4)
        yi = p.tensor([1, H, W, IC], (sy2, 0))
        p.conv(mi, w2i, b2i, yi)
        p.planar_out(yi, H, W, IC, (sy2, 0))
        p.write(sys.argv[4], c["x"], [y2])
    else:
        sys.exit("mode: tflite | groups | chain")


if __name__ == "__main__":
    main()

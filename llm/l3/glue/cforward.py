"""cforward.py (2026-11-23): the rotated SmolLM forward of one prompt chunk with the CPU glue in C (libjbglue.so via
jbglue.py), the linear steps through chunk_glue's backends (ref = int8 reference on the CPU, npu = jb_engine --serve).
Mirrors llm/ref/smollm.py SmolLM.forward with l2g_outliers.rotate's hooks (residual Q, per-head R folded into v / o,
block Hadamard on the down input; RMSNorm weights folded, so ln1 / ln2 are ones). Start of a prompt only (no KV
cache yet): position 0 on the CPU, positions 1.. through the backend. Timings per part in PackQuantC.t / forward_c."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jbglue as J  # noqa: E402

PARTS = ("norm", "linear_quant", "linear_backend", "linear_scatter", "linear_pos0", "rope", "attention", "ffn_glue",
         "residual", "embed_head")


class PackQuantC:
    """chunk_glue.PackQuant with the quantize / scatter / position-0 work in C."""
    mode, fused, quant_head = "w8a8", True, False

    def __init__(self, steps, order, backend, check=False):
        self.steps, self.order, self.be, self.check = steps, order, backend, check
        self.start = 0
        self.t = {p: 0.0 for p in PARTS}
        self.worst_in = self.worst_out = 0
        self.n_in_diff = 0
        self.rows = {}                                     # job row maps as int32, per step

    def _rows(self, key, st):
        if key not in self.rows:
            self.rows[key] = [np.ascontiguousarray(i, np.int32) for i in st.idx]
        return self.rows[key]

    @property
    def t_be(self):
        return self.t["linear_backend"]

    @property
    def t_q(self):
        return self.t["linear_quant"] + self.t["linear_scatter"] + self.t["linear_pos0"]

    def linear(self, key, x, W=None, add_to=None):
        """y = the step on x [T, IC] (position 0 on the CPU); add_to = an array [T, OC] to add y into instead (the
        residual: no y buffer at all)."""
        st = self.steps[key]
        rows = self._rows(key, st)
        T = x.shape[0]
        y = add_to if add_to is not None else np.empty((T, st.oc), np.float32)
        acc = add_to is not None
        t0 = time.monotonic()
        o = 1 if self.start == 0 else 0
        if o:                                              # position 0: int8 GEMV from the packs' weights, all jobs
            J.gemv_jobs(np.ascontiguousarray(x[0]), st.w, [c.start for c in st.cols], st.sw, rows, y[0], acc)
        t1 = time.monotonic()
        xs = x[o:] if x[o:].flags.c_contiguous else np.ascontiguousarray(x[o:])
        assert xs.shape[0] == st.tokens, "chunk must be %d tokens after position 0" % st.tokens
        xp = J.quant_planar(xs, st.sx, st.zx)              # [IC, T] planar int8
        xq = xp.T                                          # [T, IC] view for the backends
        t2 = time.monotonic()
        lv = self.be.run(self.order[key], st, xq)
        t3 = time.monotonic()
        if self.check:
            d = np.abs(xq.astype(int) - st.rec_in.astype(int))
            self.worst_in = max(self.worst_in, int(d.max()))
            self.n_in_diff += int((d > 0).sum())
            if d.max() == 0:
                self.worst_out = max(self.worst_out, max(int(np.abs(a.astype(int) - b.astype(int)).max())
                                                         for a, b in zip(lv, st.rec_out)))
        J.scatter_jobs(lv, rows, st.sy, st.zy, y[o:], acc)
        t4 = time.monotonic()
        self.t["linear_pos0"] += t1 - t0
        self.t["linear_quant"] += t2 - t1
        self.t["linear_backend"] += t3 - t2
        self.t["linear_scatter"] += t4 - t3
        return y


    def run_planar(self, key, xp, x0):
        """the step on a planar int8 input xp [IC, T] (tokens 1..T, already quantized with the step's scale) and the
        float row 0 x0: returns (the jobs' 8-bit outputs [T, n] each, position 0's output y0 [OC])."""
        st = self.steps[key]
        rows = self._rows(key, st)
        t0 = time.monotonic()
        y0 = np.empty(st.oc, np.float32)
        J.gemv_jobs(np.ascontiguousarray(x0), st.w, [c.start for c in st.cols], st.sw, rows, y0, False)
        t1 = time.monotonic()
        xq = xp.T
        lv = self.be.run(self.order[key], st, xq)
        t2 = time.monotonic()
        if self.check:
            d = np.abs(xq.astype(int) - st.rec_in.astype(int))
            self.worst_in = max(self.worst_in, int(d.max()))
            self.n_in_diff += int((d > 0).sum())
            if d.max() == 0:
                self.worst_out = max(self.worst_out, max(int(np.abs(a.astype(int) - b.astype(int)).max())
                                                         for a, b in zip(lv, st.rec_out)))
        self.t["linear_pos0"] += t1 - t0
        self.t["linear_backend"] += t2 - t1
        return lv, y0, st, rows


    def step_async(self, key, write_input, y0=None, acc0=False):
        """one step through the backend's async API: write_input(view) quantizes the step's input straight into the
        engine's input slot and returns the float row 0; submit; position 0's int8 GEMV on the CPU WHILE the NPU
        runs; collect the jobs' outputs (views, valid until the next submit). y0 / acc0: where position 0 goes."""
        st = self.steps[key]
        rows = self._rows(key, st)
        T = st.tokens
        t0 = time.monotonic()
        xp = self.be.input_view(st, T)
        x0 = write_input(xp)
        t1 = time.monotonic()
        if self.check:
            d = np.abs(xp.T.astype(int) - st.rec_in.astype(int))
            self.worst_in = max(self.worst_in, int(d.max()))
            self.n_in_diff += int((d > 0).sum())
            same = d.max() == 0
        self.be.submit(self.order[key], st)
        t2 = time.monotonic()
        y0 = np.empty(st.oc, np.float32) if y0 is None else y0
        J.gemv_jobs(np.ascontiguousarray(x0), st.w, [c.start for c in st.cols], st.sw, rows, y0, acc0)
        t3 = time.monotonic()
        lv = self.be.collect(st, T)
        t4 = time.monotonic()
        if self.check and same:
            self.worst_out = max(self.worst_out, max(int(np.abs(a.astype(int) - b.astype(int)).max())
                                                     for a, b in zip(lv, st.rec_out)))
        self.t["linear_quant"] += t1 - t0
        self.t["linear_pos0"] += t3 - t2                   # overlapped with the NPU on the npu backend
        self.t["linear_backend"] += (t2 - t1) + (t4 - t3)  # what is left of the NPU time after position 0
        return lv, y0, st, rows


def forward_c_async(m, ids, q8, record=None):
    """forward_c_fused through the backends' async API: every step's input quantized straight into the engine's
    input slot, position 0 computed while the NPU runs, the outputs read in place."""
    tm = q8.t
    q8.start = 0
    Q = m.post_embed.__self__
    Hd = m.down_in.__self__
    H, KV, hd = m.H, m.KV, m.hd
    T1 = len(ids)
    t = time.monotonic()
    x = m.embed.rows(np.asarray(ids)).astype(np.float32)
    x = J.fwht(np.ascontiguousarray(x), x.shape[1], Q.s) if Q.pow2 else np.ascontiguousarray(x @ Q.Q, np.float32)
    tm["embed_head"] += time.monotonic() - t
    for li in range(m.L):
        sq, so = q8.steps["%d.qkv" % li], q8.steps["%d.o" % li]
        sg, sd = q8.steps["%d.gateup" % li], q8.steps["%d.down" % li]
        lv, y0, st, rows = q8.step_async("%d.qkv" % li, lambda o: J.norm_quant(x, m.eps, sq.sx, sq.zx, out=o)[1])
        t = time.monotonic()
        q = np.empty((T1, H, hd), np.float32)
        k = np.empty((T1, KV, hd), np.float32)
        v = np.empty((T1, KV, hd), np.float32)
        J.qkv_rope(lv, rows, st.sy, st.zy, q, k, v, 0, m.inv_freq, bool(m.use_rope[li]))
        q[0] = y0[:H * hd].reshape(H, hd)                  # position 0: RoPE at position 0 is the identity
        k[0] = y0[H * hd:(H + KV) * hd].reshape(KV, hd)
        v[0] = y0[(H + KV) * hd:].reshape(KV, hd)
        tm["linear_scatter"] += time.monotonic() - t
        if record is not None:
            record["k0"].append(k[0].reshape(-1).copy())
            record["v0"].append(v[0].reshape(-1).copy())
        t = time.monotonic()
        att = J.attention(q, k, v, 0)
        tm["attention"] += time.monotonic() - t

        def w_o(o, att=att):
            J.quant_planar(att[1:], so.sx, so.zx, out=o)
            return att[0]
        lv, _, st, rows = q8.step_async("%d.o" % li, w_o, y0=x[0], acc0=True)
        t = time.monotonic()
        J.scatter_jobs(lv, rows, st.sy, st.zy, x[1:], True)
        tm["linear_scatter"] += time.monotonic() - t
        lv, gu0, stg, rowsg = q8.step_async("%d.gateup" % li, lambda o: J.norm_quant(x, m.eps, sg.sx, sg.zx, out=o)[1])
        t = time.monotonic()
        h0 = J.silu_mul(gu0[None])
        J.fwht(h0, Hd.b, Hd.s)
        tm["ffn_glue"] += time.monotonic() - t

        def w_d(o, lv=lv, st=stg, rows=rowsg):
            J.ffn_fused(lv, rows, st.sy, st.zy, T1 - 1, sd.ic, Hd.s, Hd.b, sd.sx, sd.zx, out=o)
            return h0[0]
        lv, _, st, rows = q8.step_async("%d.down" % li, w_d, y0=x[0], acc0=True)
        t = time.monotonic()
        J.scatter_jobs(lv, rows, st.sy, st.zy, x[1:], True)
        tm["linear_scatter"] += time.monotonic() - t
        if record is not None:
            record["x"].append(x.copy())
    t = time.monotonic()
    x = J.fwht(x, x.shape[1], Q.s, True) if Q.pow2 else np.ascontiguousarray(x @ Q.Q.T, np.float32)
    x = J.rmsnorm(x, m.norm, m.eps)
    V = m.embed.shape[0]
    if m.embed.shape[0] * m.embed.shape[1] * 4 > 256 << 20:
        lg = np.concatenate([x @ m.embed.rows(np.arange(a, min(a + 8192, V))).T for a in range(0, V, 8192)], 1)
    else:
        lg = x @ m.head().T
    tm["embed_head"] += time.monotonic() - t
    return lg


def forward_c_fused(m, ids, q8):
    """forward_c with the fused kernels: RMSNorm straight into the planar input, qkv scattered into q / k / v with
    RoPE, the FFN glue (SiLU * up, block FWHT, down's input quantization) in one pass; only the residual, q / k / v
    and the attention output are float arrays."""
    tm = q8.t
    T1 = len(ids)
    q8.start = 0
    Q = m.post_embed.__self__
    Hd = m.down_in.__self__
    H, KV, hd = m.H, m.KV, m.hd
    t = time.monotonic()
    x = m.embed.rows(np.asarray(ids)).astype(np.float32)
    x = J.fwht(np.ascontiguousarray(x), x.shape[1], Q.s) if Q.pow2 else np.ascontiguousarray(x @ Q.Q, np.float32)
    tm["embed_head"] += time.monotonic() - t
    for li in range(m.L):
        s_qkv = q8.steps["%d.qkv" % li]
        t = time.monotonic()
        ap, a0 = J.norm_quant(x, m.eps, s_qkv.sx, s_qkv.zx)
        tm["norm"] += time.monotonic() - t
        lv, y0, st, rows = q8.run_planar("%d.qkv" % li, ap, a0)
        t = time.monotonic()
        q = np.empty((T1, H, hd), np.float32)
        k = np.empty((T1, KV, hd), np.float32)
        v = np.empty((T1, KV, hd), np.float32)
        J.qkv_rope(lv, rows, st.sy, st.zy, q, k, v, 0, m.inv_freq, bool(m.use_rope[li]))
        q[0] = y0[:H * hd].reshape(H, hd)                  # position 0: RoPE at position 0 is the identity
        k[0] = y0[H * hd:(H + KV) * hd].reshape(KV, hd)
        v[0] = y0[(H + KV) * hd:].reshape(KV, hd)
        tm["linear_scatter"] += time.monotonic() - t
        t = time.monotonic()
        att = J.attention(q, k, v, 0)
        tm["attention"] += time.monotonic() - t
        q8.linear("%d.o" % li, att, add_to=x)
        s_gu = q8.steps["%d.gateup" % li]
        t = time.monotonic()
        ap, a0 = J.norm_quant(x, m.eps, s_gu.sx, s_gu.zx)
        tm["norm"] += time.monotonic() - t
        lv, gu0, st, rows = q8.run_planar("%d.gateup" % li, ap, a0)
        s_dn = q8.steps["%d.down" % li]
        t = time.monotonic()
        h0 = J.silu_mul(gu0[None])
        J.fwht(h0, Hd.b, Hd.s)
        hp = J.ffn_fused(lv, rows, st.sy, st.zy, T1 - 1, s_dn.ic, Hd.s, Hd.b, s_dn.sx, s_dn.zx)
        tm["ffn_glue"] += time.monotonic() - t
        lv, y0, st, rows = q8.run_planar("%d.down" % li, hp, h0[0])
        t = time.monotonic()
        x[0] += y0
        J.scatter_jobs(lv, rows, st.sy, st.zy, x[1:], True)
        tm["linear_scatter"] += time.monotonic() - t
    t = time.monotonic()
    x = J.fwht(x, x.shape[1], Q.s, True) if Q.pow2 else np.ascontiguousarray(x @ Q.Q.T, np.float32)
    x = J.rmsnorm(x, m.norm, m.eps)
    V = m.embed.shape[0]
    if m.embed.shape[0] * m.embed.shape[1] * 4 > 256 << 20:
        lg = np.concatenate([x @ m.embed.rows(np.arange(a, min(a + 8192, V))).T for a in range(0, V, 8192)], 1)
    else:
        lg = x @ m.head().T
    tm["embed_head"] += time.monotonic() - t
    return lg


def forward_c(m, ids, q8):
    """ids [T] -> logits [T, V] (all positions, for scoring; a prompt needs only the last row)."""
    tm = q8.t
    T = len(ids)
    pos0 = 0
    q8.start = 0
    Q = m.post_embed.__self__                              # l2g_outliers.Orth
    Hd = m.down_in.__self__                                # l2g_outliers.BlockH
    t = time.monotonic()
    x = m.embed.rows(np.asarray(ids)).astype(np.float32)
    if Q.pow2:
        x = J.fwht(np.ascontiguousarray(x), x.shape[1], Q.s)          # Q.right = fwht(x * s)
    else:
        x = np.ascontiguousarray(x @ Q.Q, np.float32)
    tm["embed_head"] += time.monotonic() - t
    H, KV, hd = m.H, m.KV, m.hd
    for li, d in enumerate(m.layers):
        t = time.monotonic()
        a = J.rmsnorm(x, None, m.eps)                      # ln1 folded into the weights (ones)
        tm["norm"] += time.monotonic() - t
        qkv = q8.linear("%d.qkv" % li, a)
        t = time.monotonic()
        q = np.ascontiguousarray(qkv[:, :H * hd]).reshape(T, H, hd)
        k = np.ascontiguousarray(qkv[:, H * hd:(H + KV) * hd]).reshape(T, KV, hd)
        v = np.ascontiguousarray(qkv[:, (H + KV) * hd:]).reshape(T, KV, hd)
        if m.use_rope[li]:
            J.rope(q, pos0, m.inv_freq)
            J.rope(k, pos0, m.inv_freq)
        tm["rope"] += time.monotonic() - t
        t = time.monotonic()
        att = J.attention(q, k, v, pos0)
        tm["attention"] += time.monotonic() - t
        q8.linear("%d.o" % li, att, add_to=x)                # x += o(att), straight into the residual
        t = time.monotonic()
        a = J.rmsnorm(x, None, m.eps)
        tm["norm"] += time.monotonic() - t
        gu = q8.linear("%d.gateup" % li, a)
        t = time.monotonic()
        h = J.silu_mul(gu)
        J.fwht(h, Hd.b, Hd.s)                              # BlockH.right: blockwise fwht(h * s)
        tm["ffn_glue"] += time.monotonic() - t
        q8.linear("%d.down" % li, h, add_to=x)               # x += down(h)
    t = time.monotonic()
    if Q.pow2:
        x = J.fwht(x, x.shape[1], Q.s, True)                          # Q.back = fwht(x) * s
    else:
        x = np.ascontiguousarray(x @ Q.Q.T, np.float32)
    x = J.rmsnorm(x, m.norm, m.eps)
    V = m.embed.shape[0]
    if m.embed.shape[0] * m.embed.shape[1] * 4 > 256 << 20:
        lg = np.concatenate([x @ m.embed.rows(np.arange(a, min(a + 8192, V))).T for a in range(0, V, 8192)], 1)
    else:
        lg = x @ m.head().T
    tm["embed_head"] += time.monotonic() - t
    return lg

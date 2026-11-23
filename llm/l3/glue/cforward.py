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

"""jbglue.py (2026-11-23): ctypes bindings of npu-llm/bin/libjbglue.so (llm/l3/glue/jb_glue.c), the C glue of a
prompt chunk. Arrays must be C-contiguous float32 / int8 / int32 numpy arrays (checked); strides are in elements."""
import ctypes as C
import os

import numpy as np

LIB = os.environ.get("JB_GLUE_LIB", "/home/jordan/Documents/AI_Projects/npu-llm/bin/libjbglue.so")
_l = C.CDLL(LIB)
_f, _i8, _i32, _d = C.POINTER(C.c_float), C.POINTER(C.c_int8), C.POINTER(C.c_int32), C.POINTER(C.c_double)
_I, _L, _F = C.c_int, C.c_long, C.c_float
_l.jbg_rmsnorm.argtypes = [_f, _f, _f, _I, _I, _F]
_l.jbg_quant_planar.argtypes = [_f, _I, _I, _F, _I, _i8]
_l.jbg_dequant_scatter.argtypes = [_i8, _L, _L, _I, _I, _i32, _F, _I, _f, _I, _I]
_l.jbg_gemv_i8.argtypes = [_f, _i8, _I, _I, _F, _i32, _f, _I]
_l.jbg_rope.argtypes = [_f, _I, _I, _I, _I, _d]
_l.jbg_attention.argtypes = [_f, _f, _f, _f, _I, _I, _I, _I, _I, _I]
_l.jbg_silu_mul.argtypes = [_f, _I, _I, _f]
_l.jbg_fwht.argtypes = [_f, _I, _I, _I, _f, _I]
_l.jbg_threads.restype = _I
_vp = C.c_void_p
_l.jbg_scatter_jobs.argtypes = [_I, C.POINTER(_vp), C.POINTER(_L), C.POINTER(_L), C.POINTER(_I), C.POINTER(_vp),
                                C.POINTER(_F), C.POINTER(_I), _f, _I, _I, _I]
_l.jbg_gemv_jobs.argtypes = [_I, _f, C.POINTER(_vp), C.POINTER(_I), C.POINTER(_I), C.POINTER(_I), C.POINTER(_F),
                             C.POINTER(_vp), _f, _I, _I]


def _p(a, dt, ptr):
    assert a.dtype == dt and a.flags.c_contiguous, (a.dtype, a.flags)
    return a.ctypes.data_as(ptr)


def threads():
    return _l.jbg_threads()


def rmsnorm(x, w, eps):
    out = np.empty_like(x)
    _l.jbg_rmsnorm(_p(x, np.float32, _f), None if w is None else _p(w, np.float32, _f), _p(out, np.float32, _f),
                   x.shape[0], x.shape[1], eps)
    return out


def quant_planar(x, s, zp):
    T, IC = x.shape
    out = np.empty((IC, T), np.int8)
    _l.jbg_quant_planar(_p(x, np.float32, _f), T, IC, float(np.float32(s)), int(zp), _p(out, np.int8, _i8))
    return out


def dequant_scatter(lv, rows, s, zp, y, accumulate):
    """lv: a job's 8-bit outputs as a [T, n] int8 array view (any strides, e.g. the NPU's planar buffer .T)."""
    T, n = lv.shape
    ts, js = lv.strides
    base = lv.ctypes.data
    _l.jbg_dequant_scatter(C.cast(base, _i8), ts, js, T, n, _p(rows, np.int32, _i32), float(s), int(zp),
                           _p(y, np.float32, _f), y.shape[1], int(accumulate))


def gemv_i8(x, w, sw, rows, y, accumulate):
    _l.jbg_gemv_i8(_p(x, np.float32, _f), _p(w, np.int8, _i8), w.shape[0], w.shape[1], float(sw),
                   _p(rows, np.int32, _i32), _p(y, np.float32, _f), int(accumulate))


def rope(x, pos0, inv_freq):
    T, heads, hd = x.shape
    _l.jbg_rope(_p(x, np.float32, _f), T, heads, hd, int(pos0), _p(inv_freq, np.float64, _d))


def attention(q, k, v, q0):
    T, H, hd = q.shape
    S, KV, _ = k.shape
    out = np.empty((T, H * hd), np.float32)
    _l.jbg_attention(_p(q, np.float32, _f), _p(k, np.float32, _f), _p(v, np.float32, _f), _p(out, np.float32, _f),
                     T, S, int(q0), H, KV, hd)
    return out


def silu_mul(gu):
    T, I2 = gu.shape
    h = np.empty((T, I2 // 2), np.float32)
    _l.jbg_silu_mul(_p(gu, np.float32, _f), T, I2 // 2, _p(h, np.float32, _f))
    return h


def fwht(x, b, s=None, post_signs=False):
    """in place on x [T, n]: blockwise normalized FWHT of x * s (or FWHT(x) * s with post_signs)."""
    _l.jbg_fwht(_p(x, np.float32, _f), x.shape[0], x.shape[1], b, None if s is None else _p(s, np.float32, _f),
                int(post_signs))
    return x


def scatter_jobs(lvs, rows, s, zp, y, accumulate=False):
    """y [T, OC] = the jobs' 8-bit outputs (each a [T, n] array view, any strides) dequantized into their rows (summed
    for a split step). y is overwritten (no clearing needed)."""
    nj = len(lvs)
    keep = (lvs, rows)                                   # keep the arrays alive during the call
    LV = (_vp * nj)(*[a.ctypes.data for a in lvs])
    TS = (_L * nj)(*[a.strides[0] for a in lvs])
    JS = (_L * nj)(*[a.strides[1] for a in lvs])
    N = (_I * nj)(*[a.shape[1] for a in lvs])
    for r in rows:
        assert r.dtype == np.int32 and r.flags.c_contiguous
    R = (_vp * nj)(*[r.ctypes.data for r in rows])
    S = (_F * nj)(*[float(v) for v in s])
    Z = (_I * nj)(*[int(v) for v in zp])
    _l.jbg_scatter_jobs(nj, LV, TS, JS, N, R, S, Z, _p(y, np.float32, _f), y.shape[0], y.shape[1], int(accumulate))
    del keep


def gemv_jobs(x, ws, cols, sws, rows, y, accumulate=False):
    """position 0: y [OC] = sum over jobs of sw * w [n, ic] . x[col : col + ic] at the job's rows (y overwritten)."""
    nj = len(ws)
    for w in ws:
        assert w.dtype == np.int8 and w.flags.c_contiguous
    W = (_vp * nj)(*[w.ctypes.data for w in ws])
    N = (_I * nj)(*[w.shape[0] for w in ws])
    IC = (_I * nj)(*[w.shape[1] for w in ws])
    COL = (_I * nj)(*[int(c) for c in cols])
    SW = (_F * nj)(*[float(v) for v in sws])
    R = (_vp * nj)(*[r.ctypes.data for r in rows])
    _l.jbg_gemv_jobs(nj, _p(x, np.float32, _f), W, N, IC, COL, SW, R, _p(y, np.float32, _f), y.shape[0],
                     int(accumulate))

#!/usr/bin/env python3
"""l2j_down_split.py (2026-11-22, CPU only): split the down step's INPUT channels into K pieces (each piece its own
NPU jobs, 8-bit partial outputs with per-job scales, the CPU adds the K partials into the residual). Why: on the NPU
the 1.7B down (8192 inputs, 2 MiB per 256-token chunk) can't cache its input; a 2048-input piece tiles like the o
step (tile-ladder 2026-11-22: o 28.7 -> 14.8 ms per 4 layers). Cost: K roundings of partial sums instead of one.
Rows (l2h_groups row 4 setup: rotation, 4 jobs per step, per-job output scales, position 0 on the CPU, head fp32):
  K = 1 (today), then the K values in L2J_K (default 2,4): down = sum of K pieces, each 4 jobs over its own output
  groups (by max |w| of the piece), its own calibrated per-job output ranges; the input is quantized once, as today.
Grouping (2026-11-23, after the 1.7B run: weight-grouped pieces cost KL .182 -> .251): L2J_GROUP = w (default: equal
groups by max |w| of the piece) | range (groups by the calibrated output range of each channel, uneven sizes chosen by
dynamic programming to minimise sum over jobs of rows x range^2, i.e. the outlier channels get small jobs of their own);
L2J_JOBS = jobs per piece (default 4). With L2J_GROUP=range the K = 1 row uses the same grouping on the whole down step.
Usage: l2j_down_split.py MODEL_DIR [TOKENS]     writes npu-llm/runs/l2j-down-split-<model>-<stamp>.json
"""
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "CORTEXA73")
import numpy as np  # noqa: E402

MD = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l2g_outliers as G  # noqa: E402  (reads MODEL_DIR and TOKENS from argv)
import l2_quality  # noqa: E402
from l2_quality import logsoftmax, score  # noqa: E402
from smollm import SmolLM, Quant, _w  # noqa: E402

RUNS = "/home/jordan/Documents/AI_Projects/npu-llm/runs"
JOBS_LIST = [int(j) for j in os.environ.get("L2J_JOBS", "4").split(",")]
JOBS = JOBS_LIST[0]
MIN_ROWS = 8                                                     # rows per job at least (one per NN core)
GROUP = os.environ.get("L2J_GROUP", "w")
KS = [int(k) for k in os.environ.get("L2J_KS", "1," + os.environ.get("L2J_K", "2,4")).split(",")]


def dp_groups(r, G):
    """channels -> G groups minimising sum of rows x (group's max r)^2 over contiguous runs of the channels sorted by r
    (the 8-bit step of a job is its range / 255, every row pays its job's step). Returns a group id per channel."""
    order = np.argsort(r)
    rs = r[order].astype(np.float64) ** 2
    N = len(rs)
    INF = np.inf
    best = np.full(N + 1, INF)
    best[0] = 0.0
    cut = np.zeros((G + 1, N + 1), np.int64)
    for g in range(1, G + 1):
        nb = np.full(N + 1, INF)
        for j in range(1, N + 1):                              # group ends at j: rows i..j-1, cost (j - i) * rs[j-1]
            i = np.arange(0, j)
            c = best[:j] + (j - i) * rs[j - 1]
            c[j - i < MIN_ROWS] = INF                          # no job smaller than MIN_ROWS
            k = int(np.argmin(c))
            nb[j], cut[g, j] = c[k], k
        best = nb
    gid = np.empty(N, np.int32)
    j = N
    for g in range(G, 0, -1):
        i = cut[g, j]
        gid[order[i:j]] = g - 1
        j = i
    return gid


def piece_ranges(m, ids_list, K):
    """per-channel min / max of each of the K partial down outputs (fp32, rotated), position 0 left out."""
    rng = {}

    class Rec(Quant):
        def linear(self, key, x, W):
            Wf = _w(W)
            if key.endswith(".down"):
                n = Wf.shape[1] // K
                for k in range(K):
                    y = x[1:, k * n:(k + 1) * n] @ Wf[:, k * n:(k + 1) * n].T
                    lo, hi = y.min(0), y.max(0)
                    o = rng.get((key, k))
                    rng[(key, k)] = (lo, hi) if o is None else (np.minimum(o[0], lo), np.maximum(o[1], hi))
            return x @ Wf.T

    for ids in ids_list:
        m.forward(ids, Rec("fp32", True))
    return rng


class SplitDown(G.QuantG):
    def __init__(self, K, prng, *a, jobs=None, **k):
        super().__init__(*a, **k)
        self.K, self.prng, self.gcache = K, prng, {}
        self.jobs = jobs or JOBS

    def linear(self, key, x, W):
        if not key.endswith(".down") or (self.K == 1 and GROUP == "w"):
            return super().linear(key, x, W)
        Wf = _w(W)
        y0 = x[:1] @ Wf.T                                        # position 0: CPU
        xs = x[1:]
        sx, zx = self.qparams(*self.rng(key + ".in", xs))       # the down input quantized once, as today
        xq = (self.q8(xs, sx, zx) - zx).astype(np.float64)
        n = Wf.shape[1] // self.K
        y = np.zeros((xs.shape[0], Wf.shape[0]), np.float64)
        for k in range(self.K):
            Wk = Wf[:, k * n:(k + 1) * n]
            r = np.abs(Wk).max(1)
            lo, hi = self.prng[(key, k)]                         # per-job 8-bit output, calibrated partial ranges
            sw = np.empty(Wk.shape[0])
            if GROUP == "range":
                gid = self.gcache.get((key, k))
                if gid is None:
                    gid = self.gcache[(key, k)] = dp_groups(np.maximum(np.abs(lo), np.abs(hi)), self.jobs)
                for j in range(self.jobs):
                    grp = np.nonzero(gid == j)[0]
                    sw[grp] = max(float(r[grp].max()), 1e-12) / 127.0
            else:
                gid = np.empty(Wk.shape[0], np.int32)
                for j, grp in enumerate(np.array_split(np.argsort(r), self.jobs)):
                    sw[grp], gid[grp] = max(float(r[grp].max()), 1e-12) / 127.0, j
            Wq = np.clip(np.round(Wk / sw[:, None]), -127, 127)
            yk = (xq[:, k * n:(k + 1) * n] @ Wq.T) * (sx * sw[None, :])
            glo, ghi = np.zeros(self.jobs), np.zeros(self.jobs)
            np.minimum.at(glo, gid, lo)
            np.maximum.at(ghi, gid, hi)
            s = np.maximum(ghi - glo, 1e-8) / 255.0
            zp = np.clip(np.round(-glo / s) - 128, -128, 127)
            yk = (np.clip(np.round(yk / s[gid]) + zp[gid], -128, 127) - zp[gid]) * s[gid]
            y += yk                                              # the CPU adds the partials
        return np.concatenate([y0.astype(np.float32), y.astype(np.float32)])


def main():
    from tokenizers import Tokenizer
    single = os.path.join(MD, "model.safetensors")
    big = not os.path.exists(single) or os.path.getsize(single) > 1 << 30
    Quant.CACHE, Quant.F64 = not big, not big
    t0 = time.monotonic()
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    m = SmolLM(MD)
    name = os.path.basename(MD)
    calib, evals = l2_quality.texts(tok)
    refs = {tn: logsoftmax(m.forward(ids, Quant("fp32"))) for tn, ids in evals.items()}
    what = G.rotate(m, m.layers)
    rec, chin, chout = G.calibrate3(m, calib)
    prngs = {K: piece_ranges(m, calib, K) for K in KS if K > 1}
    prngs[1] = {(k, 0): (chout[0][k], chout[1][k]) for k in chout[0] if k.endswith(".down")}   # whole down = 1 piece
    print("%s, %d tokens per text; %s; calibration + fp32 %.0f s" % (name, G.T, what, time.monotonic() - t0), flush=True)
    print("%-36s %-7s %8s %10s %10s" % ("down input split (4 jobs per piece)", "text", "ppl", "top1=fp32", "KL"),
          flush=True)
    out = {"model": name, "tokens_per_text": G.T, "rotation": what, "rows": []}
    for tn, ids in evals.items():
        s = score(refs[tn], refs[tn], np.asarray(ids))
        out["rows"].append({"K": None, "text": tn, **{k: float(v) for k, v in s.items()}})
        print("%-36s %-7s %8.2f" % ("fp32", tn, s["ppl"]), flush=True)
    for K, jobs in [(K, j) for j in JOBS_LIST for K in KS if not (K == 1 and GROUP == "w" and j != JOBS_LIST[0])]:
        Quant.WGROUPS, Quant.WGROUPS_BY = 4, None                  # the other steps: 4 jobs, as today
        Quant.WQ.clear()
        G.QuantG.GRP.clear()
        q = SplitDown(K, prngs.get(K), "w8a8", True, rec, chout=chout, per_job_out=True, quant_head=False,
                      fp_first=True, jobs=jobs)
        for tn, ids in evals.items():
            t = time.monotonic()
            s = score(refs[tn], logsoftmax(m.forward(ids, q)), np.asarray(ids))
            out["rows"].append({"K": K, "jobs": jobs, "text": tn, **{k: float(v) for k, v in s.items()}})
            print("%-36s %-7s %8.2f %9.1f%% %10.4f   (%.0f s)" % (
                ("today (one piece)" if GROUP == "w" else "1 piece, %d range jobs" % jobs) if K == 1 else
                "%d pieces x %d %s jobs" % (K, jobs, "range" if GROUP == "range" else "w"), tn, s["ppl"],
                100 * s["top1_vs_fp32"], s["kl_vs_fp32"], time.monotonic() - t), flush=True)
    out["group"], out["jobs_per_piece"] = GROUP, JOBS
    path = os.path.join(RUNS, "l2j-down-split-%s-%s.json" % (name, time.strftime("%Y%m%d-%H%M%S")))
    json.dump(out, open(path, "w"), indent=1)
    print("wrote %s (%.0f min)" % (path, (time.monotonic() - t0) / 60), flush=True)


if __name__ == "__main__":
    sys.exit(main())

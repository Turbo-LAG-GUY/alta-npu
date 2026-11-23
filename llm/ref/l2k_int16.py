#!/usr/bin/env python3
"""l2k_int16.py (2026-11-23, CPU only): which 1.7B layer step carries the remaining 8-bit gap? The A311D's NN cores
also multiply int16 x int16 (hwdb: 8 INT16 cores, NN_INT16_ALU 1; no int16 x int8 mix, no float), a path Mesa never
drives. Before any reverse engineering: move ONE step kind at a time to 16 bits (input, weights and outputs, as the
int16 cores would) and see how much of the gap to fp32 closes.
Base = today's 1.7B layout (llm/l3/real_packs.py 4 4range): 4 jobs per step grouped by max |w|, per-job 8-bit output
scales, static input scale per step, position 0 on the CPU, head fp32; down = 4 input pieces x 4 range-grouped jobs
(calibrated partial scales; llm/ref/l2j_down_split.py, sim KL .145 / .086).
Rows (L2K_ROWS, default base,qkv,o,gateup,down,all): base = all 8-bit; <kind> = that step kind in 16 bits; all = every
step in 16 bits (upper bound for the int16 path).
Usage: l2k_int16.py MODEL_DIR [TOKENS]     writes npu-llm/runs/l2k-int16-<model>-<stamp>.json
"""
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "CORTEXA73")
os.environ["L2J_GROUP"] = "range"                       # today's down layout: range-grouped jobs per piece
os.environ["L2J_JOBS"] = "4"
os.environ["L2J_KS"] = "4"
import numpy as np  # noqa: E402

MD = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l2g_outliers as G  # noqa: E402  (reads MODEL_DIR and TOKENS from argv)
import l2_quality  # noqa: E402
import l2j_down_split as S  # noqa: E402
from l2_quality import logsoftmax, score  # noqa: E402
from smollm import SmolLM, Quant, _w  # noqa: E402

RUNS = "/home/jordan/Documents/AI_Projects/npu-llm/runs"
ROWS = os.environ.get("L2K_ROWS", "base,qkv,o,gateup,down,all").split(",")
KINDS = ("qkv", "o", "gateup", "down")
K, JOBS = 4, 4


def aq(lo, hi, bits):
    """asymmetric activation quantization: scale, zero point for [lo, hi] (and 0) with 2^bits levels."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    s = max(hi - lo, 1e-8) / (2 ** bits - 1)
    zmin = -2 ** (bits - 1)
    return s, float(np.clip(np.round(-lo / s) + zmin, zmin, -zmin - 1))


def qa(x, s, z, bits):
    zmin = -2 ** (bits - 1)
    return np.clip(np.round(x / s) + z, zmin, -zmin - 1)


def out_levels(y, gid, lo, hi, bits):
    """per-job output quantization (calibrated per-channel ranges lo / hi, grouped by gid)."""
    ng = int(gid.max()) + 1
    glo, ghi = np.zeros(ng), np.zeros(ng)
    np.minimum.at(glo, gid, lo)
    np.maximum.at(ghi, gid, hi)
    zmin = -2 ** (bits - 1)
    s = np.maximum(ghi - glo, 1e-8) / (2 ** bits - 1)
    z = np.clip(np.round(-glo / s) + zmin, zmin, -zmin - 1)
    return (np.clip(np.round(y / s[gid]) + z[gid], zmin, -zmin - 1) - z[gid]) * s[gid]


class Wide(S.SplitDown):
    """today's layout, with the step kinds in `wide` in 16 bits."""

    def __init__(self, wide, *a, **k):
        super().__init__(*a, **k)
        self.wide = set(wide)
        self.w16 = {}

    def linear(self, key, x, W):
        kind = key.split(".")[-1]
        if key == "head" or self.mode == "fp32" or kind not in self.wide:
            return super().linear(key, x, W)
        if kind == "down":
            return self.split16(key, x, W)
        return self.plain16(key, x, W)

    def plain16(self, key, x, W):
        """QuantG.linear (groups by max |w|, per-job outputs) with 16-bit input, weights and outputs."""
        (_, _), gid = self.wgroups(key, W)
        hit = self.w16.get(key)
        if hit is None:
            Wf = _w(W)
            sw = np.empty(Wf.shape[0], np.float64)
            for g in range(int(gid.max()) + 1):
                rows = np.nonzero(gid == g)[0]
                sw[rows] = max(float(np.abs(Wf[rows]).max()), 1e-12) / 32767.0
            hit = (np.clip(np.round(Wf / sw[:, None]), -32767, 32767), sw)
            if Quant.CACHE:                                   # small models only (1.7B: GBs; re-quantized per call)
                self.w16[key] = hit
        Wq, sw = hit
        y0 = x[:1] @ (Wq * sw[:, None]).T                     # position 0: CPU
        xs = x[1:]
        sx, zx = aq(*self.rng(key + ".in", xs), 16)
        xq = qa(xs, sx, zx, 16) - zx
        y = (xq.astype(np.float64) @ Wq.T) * (sx * sw[None, :])
        y = out_levels(y, gid, self.chout[0][key], self.chout[1][key], 16)
        return np.concatenate([y0.astype(np.float32), y.astype(np.float32)])

    def split16(self, key, x, W):
        """SplitDown (4 pieces x 4 range-grouped jobs) with 16-bit input, weights and partial outputs."""
        Wf = _w(W)
        xs = x[1:]
        sx, zx = aq(*self.rng(key + ".in", xs), 16)
        xq = (qa(xs, sx, zx, 16) - zx).astype(np.float64)
        n = Wf.shape[1] // K
        y = np.zeros((xs.shape[0], Wf.shape[0]), np.float64)
        y0 = np.zeros((1, Wf.shape[0]), np.float64)
        for k in range(K):
            Wk = Wf[:, k * n:(k + 1) * n]
            r = np.abs(Wk).max(1)
            lo, hi = self.prng[(key, k)]
            gid = self.gcache.get((key, k))
            if gid is None:
                gid = self.gcache[(key, k)] = S.dp_groups(np.maximum(np.abs(lo), np.abs(hi)), JOBS)
            sw = np.empty(Wk.shape[0])
            for j in range(JOBS):
                grp = np.nonzero(gid == j)[0]
                sw[grp] = max(float(r[grp].max()), 1e-12) / 32767.0
            Wq = np.clip(np.round(Wk / sw[:, None]), -32767, 32767)
            y0 += x[:1, k * n:(k + 1) * n] @ (Wq * sw[:, None]).T
            yk = (xq[:, k * n:(k + 1) * n] @ Wq.T) * (sx * sw[None, :])
            y += out_levels(yk, gid, lo, hi, 16)
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
    prng = S.piece_ranges(m, calib, K)
    print("%s, %d tokens per text; %s; calibration + fp32 %.0f s" % (name, G.T, what, time.monotonic() - t0), flush=True)
    print("%-36s %-7s %8s %10s %10s" % ("16-bit steps (rest: today's 8-bit)", "text", "ppl", "top1=fp32", "KL"),
          flush=True)
    out = {"model": name, "tokens_per_text": G.T, "rotation": what, "rows": []}
    for tn, ids in evals.items():
        s = score(refs[tn], refs[tn], np.asarray(ids))
        out["rows"].append({"row": "fp32", "text": tn, **{k: float(v) for k, v in s.items()}})
        print("%-36s %-7s %8.2f" % ("fp32", tn, s["ppl"]), flush=True)
    for row in ROWS:
        wide = () if row == "base" else KINDS if row == "all" else (row,)
        Quant.WGROUPS, Quant.WGROUPS_BY = JOBS, None
        Quant.WQ.clear()
        G.QuantG.GRP.clear()
        q = Wide(wide, K, prng, "w8a8", True, rec, chout=chout, per_job_out=True, quant_head=False, fp_first=True,
                 jobs=JOBS)
        for tn, ids in evals.items():
            t = time.monotonic()
            s = score(refs[tn], logsoftmax(m.forward(ids, q)), np.asarray(ids))
            out["rows"].append({"row": row, "text": tn, **{k: float(v) for k, v in s.items()}})
            print("%-36s %-7s %8.2f %9.1f%% %10.4f   (%.0f s)" % (
                "base: all 8-bit (today)" if row == "base" else "all steps 16-bit" if row == "all" else
                "%s in 16 bits" % row, tn, s["ppl"], 100 * s["top1_vs_fp32"], s["kl_vs_fp32"], time.monotonic() - t),
                flush=True)
    path = os.path.join(RUNS, "l2k-int16-%s-%s.json" % (name, time.strftime("%Y%m%d-%H%M%S")))
    json.dump(out, open(path, "w"), indent=1)
    print("wrote %s (%.0f min)" % (path, (time.monotonic() - t0) / 60), flush=True)


if __name__ == "__main__":
    sys.exit(main())

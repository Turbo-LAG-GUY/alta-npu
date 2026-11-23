#!/usr/bin/env python3
"""l2i_npu_mlp.py (2026-11-22, CPU only): can the FFN glue run on the NPU without losing quality?
Idea 2 of the "NPU drives itself" plan: gateup -> SiLU(gate) * up -> down in ONE NPU submit, nothing back to the CPU.
The NPU can only do that in 8 bits, with one scale per tensor:
  - gateup jobs PAIRED: job j holds gate channels G_j and the matching up channels G_j (one output scale per job),
    groups by max |w| over the pair;
  - SiLU on the TP unit's lookup table (8-bit in, 8-bit out, own calibrated scale per job);
  - the multiply on the shader core (mesa-compute step 6g, exact vs TFLite: round(a * b * sa * sb / so) + zo);
  - NO down-input Hadamard (1.7B: a full 8192 transform = a 64 MB conv): each job's h scale is folded into the down
    weight columns, its zero point into the bias, so down reads the concatenated 8-bit h levels as one input.
Rows (same rotation otherwise as l2h_groups row 4: residual Q, per-head R, 4 jobs per step, per-job output scales,
position 0 on the CPU, head fp32):
  cpu     today: fp SiLU * up on the CPU, block Hadamard, down input quantized with one static scale
  pairs   cpu, but gateup jobs paired (the grouping the NPU path needs), to separate the two effects
  nohad   pairs, no Hadamard: down input = per-job quantized h (scales folded into down), fp SiLU * up
  npu     nohad + SiLU and the multiply rounded to 8 bits like the NPU does it
  hadconv npu, but the block Hadamard kept as an 8-bit NPU conv over the per-job h levels (job scales folded into its
          weights, 4 weight groups); its output scale = down's static input scale, so down is unchanged from today
Usage: l2i_npu_mlp.py MODEL_DIR [TOKENS]     writes npu-llm/runs/l2i-npu-mlp-<model>-<stamp>.json
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
ROWS = os.environ.get("L2I_ROWS", "cpu,pairs,nohad,npu,hadconv").split(",")
JOBS = int(os.environ.get("L2I_JOBS", "4"))
PAIR_BY = os.environ.get("L2I_PAIR_BY", "w")   # w: max |w| of the pair; h: calibrated |h| range


def silu(x):
    return x / (1.0 + np.exp(-x))


def qp(lo, hi):
    """asymmetric int8 scale / zero point covering [lo, hi] (and 0)."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    s = max(hi - lo, 1e-8) / 255.0
    return s, float(np.clip(np.round(-lo / s) - 128, -128, 127))


def q8(x, s, z):
    return np.clip(np.round(x / s) + z, -128, 127)


class NpuMLP(G.QuantG):
    """QuantG with paired gateup groups and, per row, the FFN glue the NPU would do."""

    def __init__(self, row, I, ffn_rng, *a, **k):
        super().__init__(*a, **k)
        self.row, self.I, self.ffn_rng = row, I, ffn_rng
        self.gu = {}

    def wgroups(self, key, W):
        if not key.endswith("gateup") or self.row == "cpu":
            return super().wgroups(key, W)
        ck = "%s#pairs%d" % (key, JOBS)
        if ck in Quant.WQ:
            return Quant.WQ[ck], G.QuantG.GRP[ck]
        W = _w(W)
        I = self.I
        r = np.abs(W).max(1)
        if PAIR_BY == "h":                                      # pairs sorted by their calibrated h range
            lo, hi = self.ffn_rng["%s.h" % key.split(".")[0]]
            order = np.argsort(np.maximum(np.abs(lo), np.abs(hi)))
        else:
            order = np.argsort(np.maximum(r[:I], r[I:]))       # channel pairs (gate i, up i) sorted together
        gid, s = np.empty(2 * I, np.int32), np.empty(2 * I, np.float32)
        for j, grp in enumerate(np.array_split(order, JOBS)):
            both = np.concatenate([grp, grp + I])
            gid[both], s[both] = j, max(float(r[both].max()), 1e-12) / 127.0
        wq = (np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8), s)
        Quant.WQ[ck], G.QuantG.GRP[ck] = wq, gid
        return wq, gid

    def linear(self, key, x, W):
        if key.endswith(".down") and self.row in ("nohad", "npu"):
            return self.down_folded(key, x, W)
        y = super().linear(key, x, W)
        if key.endswith(".gateup"):
            self.gu[key.split(".")[0]] = y                     # dequantized 8-bit gate | up (position 0 in fp)
        return y

    def ffn(self, li, h):
        """ffn_hook: h = silu(gate) * up from the 8-bit gateup output, as the NPU path would compute it."""
        self.cur_li = li
        if self.row not in ("npu", "hadconv"):
            return h
        gu = self.gu[str(li)]
        I = self.I
        gt, up = gu[:, :I], gu[:, I:]
        _, gid = self.wgroups("%d.gateup" % li, None)
        g = gid[:I]
        out = np.empty_like(h)
        out[:1] = h[:1]                                         # position 0: CPU
        for j in range(JOBS):
            c = np.where(g == j)[0]
            lo, hi = self.ffn_rng["%d.silu" % li]
            ss, zs = qp(lo[c].min(), hi[c].max())
            sl = (q8(silu(gt[1:, c]), ss, zs) - zs) * ss       # SiLU on the TP lookup table, 8-bit out
            lo, hi = self.ffn_rng["%d.h" % li]
            sh, zh = qp(lo[c].min(), hi[c].max())
            out[1:, c] = (q8(sl * up[1:, c], sh, zh) - zh) * sh  # shader-core MUL, 8-bit out
        return out

    def hadconv(self, h):
        """down_in for row hadconv: the block Hadamard as an 8-bit conv over the per-job 8-bit h levels."""
        li = self.cur_li
        _, gid = self.wgroups("%d.gateup" % li, None)
        g = gid[:self.I]
        lo, hi = self.ffn_rng["%d.h" % li]
        sh = np.empty(self.I, np.float32)
        zh = np.empty(self.I, np.float32)
        for j in range(JOBS):
            c = np.where(g == j)[0]
            sh[c], zh[c] = qp(lo[c].min(), hi[c].max())
        lev = np.round(h[1:] / sh) + 0.0                       # h holds (level - zp) * scale already
        Wf = (self.Hm * sh[:, None]).T                         # [out, in]: job scales folded into the columns
        r = np.abs(Wf).max(1)
        order = np.argsort(r)
        sw = np.empty(Wf.shape[0], np.float32)
        for j, grp in enumerate(np.array_split(order, JOBS)):
            sw[grp] = max(float(r[grp].max()), 1e-12) / 127.0
        Wq = np.clip(np.round(Wf / sw[:, None]), -127, 127)
        y = (lev.astype(np.float64) @ Wq.T.astype(np.float64)) * sw[None, :]
        return np.concatenate([self.hd(h[:1]), y.astype(np.float32)])

    def down_folded(self, key, x, W):
        """down reading the concatenated per-job 8-bit h (no Hadamard): job scales folded into the weight columns."""
        li = key.split(".")[0]
        _, gid = self.wgroups("%s.gateup" % li, None)
        g = gid[:self.I]
        lo, hi = self.ffn_rng["%s.h" % li]
        sh = np.empty(self.I, np.float32)
        zh = np.empty(self.I, np.float32)
        for j in range(JOBS):
            c = np.where(g == j)[0]
            sh[c], zh[c] = qp(lo[c].min(), hi[c].max())
        y0 = x[:1] @ _w(W).T                                    # position 0: CPU, fp (int8 weights below differ by little)
        lev = q8(x[1:], sh, zh) - zh                            # 8-bit h levels (a no-op re-rounding for row npu)
        Wf = _w(W) * sh[None, :]                                # scales folded into the columns
        r = np.abs(Wf).max(1)
        order = np.argsort(r)
        sw = np.empty(Wf.shape[0], np.float32)
        dgid = np.empty(Wf.shape[0], np.int32)
        for j, grp in enumerate(np.array_split(order, JOBS)):
            sw[grp], dgid[grp] = max(float(r[grp].max()), 1e-12) / 127.0, j
        Wq = np.clip(np.round(Wf / sw[:, None]), -127, 127)
        y = (lev.astype(np.float64) @ Wq.T.astype(np.float64)) * sw[None, :]
        y = self.out_q(key, y.astype(np.float32), dgid)
        return np.concatenate([y0.astype(np.float32), y.astype(np.float32)])


def ffn_ranges(m, ids_list, I):
    """per-channel min / max of silu(gate) and h = silu(gate) * up over the calibration texts (fp32, rotated)."""
    rng = {}

    class Rec(Quant):
        def linear(self, key, x, W):
            y = x @ _w(W).T
            if key.endswith(".gateup"):
                li = key.split(".")[0]
                s, h = silu(y[1:, :I]), silu(y[1:, :I]) * y[1:, I:]
                for k, v in (("silu", s), ("h", h)):
                    lo, hi = v.min(0), v.max(0)
                    o = rng.get("%s.%s" % (li, k))
                    rng["%s.%s" % (li, k)] = (lo, hi) if o is None else (np.minimum(o[0], lo), np.maximum(o[1], hi))
            return y

    for ids in ids_list:
        m.forward(ids, Rec("fp32", True))
    return rng


def main():
    from tokenizers import Tokenizer
    single = os.path.join(MD, "model.safetensors")
    big = not os.path.exists(single) or os.path.getsize(single) > 1 << 30
    Quant.CACHE, Quant.F64 = not big, not big
    t0 = time.monotonic()
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    m = SmolLM(MD)
    name = os.path.basename(MD)
    I = m.cfg["intermediate_size"]
    calib, evals = l2_quality.texts(tok)
    refs = {tn: logsoftmax(m.forward(ids, Quant("fp32"))) for tn, ids in evals.items()}
    base = list(m.layers)
    what = G.rotate(m, base)
    rec, chin, chout = G.calibrate3(m, calib)
    rot_layers, hd = m.layers, m.down_in
    # the same rotation without the down-input Hadamard (rows nohad / npu): down weights back to Q.left_t(W) only
    Q = m.post_embed.__self__
    plain = []
    for d0, d in zip(base, rot_layers):
        nd = dict(d)
        nd["down"] = lambda W=d0["down"]: Q.left_t(_w(W))
        plain.append(nd)
    m.layers, m.down_in = plain, None
    rec2, chin2, chout2 = G.calibrate3(m, calib)
    rng_plain = ffn_ranges(m, calib, I)
    m.layers, m.down_in = rot_layers, hd
    rng_rot = rng_plain                                          # silu / h ranges don't depend on the down rotation
    Hm = hd(np.eye(I, dtype=np.float32))                         # the block Hadamard as a matrix: hd(A) = A @ Hm
    print("%s, %d tokens per text; %s; calibration + fp32 %.0f s" % (name, G.T, what, time.monotonic() - t0), flush=True)
    print("%-44s %-7s %8s %10s %10s" % ("FFN glue (%d jobs per step, pairs by %s)" % (JOBS, PAIR_BY), "text", "ppl", "top1=fp32", "KL"), flush=True)
    out = {"model": name, "tokens_per_text": G.T, "rotation": what, "rows": []}
    for tn, ids in evals.items():
        s = score(refs[tn], refs[tn], np.asarray(ids))
        out["rows"].append({"row": "fp32", "text": tn, **{k: float(v) for k, v in s.items()}})
        print("%-44s %-7s %8.2f" % ("fp32", tn, s["ppl"]), flush=True)
    label = {"cpu": "cpu: fp glue + Hadamard (today)", "pairs": "pairs: today + paired gateup jobs",
             "nohad": "nohad: paired, no Hadamard, per-job h scales", "npu": "npu: nohad + 8-bit SiLU and MUL",
             "hadconv": "hadconv: npu + Hadamard as an 8-bit conv"}
    for row in ROWS:
        Quant.WGROUPS, Quant.WGROUPS_BY = JOBS, None
        Quant.WQ.clear()
        G.QuantG.GRP.clear()
        if row in ("nohad", "npu"):
            m.layers, m.down_in, r_, co = plain, None, rec2, chout2
        else:
            m.layers, m.down_in, r_, co = rot_layers, hd, rec, chout
        q = NpuMLP(row, I, rng_plain if row in ("nohad", "npu") else rng_rot, "w8a8", True, r_, chout=co,
                   per_job_out=True, quant_head=False, fp_first=True)
        if row == "hadconv":
            q.Hm, q.hd = Hm, hd
            m.down_in = q.hadconv
        for tn, ids in evals.items():
            t = time.monotonic()
            s = score(refs[tn], logsoftmax(m.forward(ids, q, ffn_hook=q.ffn)), np.asarray(ids))
            out["rows"].append({"row": row, "text": tn, **{k: float(v) for k, v in s.items()}})
            print("%-44s %-7s %8.2f %9.1f%% %10.4f   (%.0f s)" % (label[row], tn, s["ppl"], 100 * s["top1_vs_fp32"],
                                                              s["kl_vs_fp32"], time.monotonic() - t), flush=True)
    out["jobs"], out["pair_by"] = JOBS, PAIR_BY
    path = os.path.join(RUNS, "l2i-npu-mlp-%s-%s.json" % (name, time.strftime("%Y%m%d-%H%M%S")))
    json.dump(out, open(path, "w"), indent=1)
    print("wrote %s (%.0f min)" % (path, (time.monotonic() - t0) / 60), flush=True)


if __name__ == "__main__":
    sys.exit(main())

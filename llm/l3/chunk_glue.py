#!/usr/bin/env python3
"""chunk_glue.py (2026-11-21, LLM engine R2 step 2): one whole prompt chunk end to end - every linear layer step
from the NPU packs (llm/l3/real_packs.py: the same int8 weights, the same static input scale, the same 4 jobs with
their own output scales), everything else on the CPU (the "glue"): embedding lookup + residual rotation, RMSNorm,
RoPE, attention over the chunk, SiLU x up, the block Hadamard on the down input, the residual adds, the final norm
and the fp32 head. Position 0 is the CPU's (int8 weights from the packs, fp activations), positions 1..256 = one
256-token chunk (a 16 x 16 token image) per step.

The glue per step: quantize the step input with the pack's input scale / zero point -> int8 [T, IC] -> planar
[IC, 16, 16] -> backend -> 4 planar int8 outputs [n_g, 16, 16] -> dequantize each with its job's scale / zero point
-> scatter the rows back to model order (groups.npz: row -> job).

Backends:
  ref   exact int8 reference on the CPU (jb_pack.ref_conv arithmetic, float64 accumulation): no NPU
  npu   the NPU through `jb_engine --serve` (shared-memory buffers, one step per request). NPU runs ONLY via
        scripts/npu-run with Jordan's OK for that step.

Checks: (1) every step through the backend on its pack's recorded input must give the pack's expected levels (to
within 1); (2) perplexity / KL / top-1 vs the fp32 model on GPL-3 and README, and the time split (backend vs glue)
per chunk; --save-logits keeps the log-probabilities so an NPU run can be compared with the ref run (top-1 agreement,
KL between the two). The glue's own step inputs do NOT match the packs' recorded inputs level for level: a 1-level
rounding tie (float order) in one step spreads through the block Hadamard and attention (2026-11-21: layer 0 gateup
2 ties -> layer 1 down 21% of levels differ, later layers 70-85%, max ~20 levels), so that check is only a printout.

Usage: chunk_glue.py [--backend ref|npu [--offline]] [--engine PATH] [--json OUT] [--save-logits OUT.npz] [--jitter K]
                     MODEL_DIR PACK_DIR
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

# OpenBLAS (DYNAMIC_ARCH) picks its kernel from the core the process starts on: cortexa53 or cortexa73 kernels sum in a
# different order, and the format's rounding ties spread that 1-ulp difference to whole perplexity points (2026-11-21:
# GPL-3 ppl 16.13 started on an A53 vs 15.70 on an A73, same code). Pin one kernel so runs compare.
os.environ.setdefault("OPENBLAS_CORETYPE", "CORTEXA73")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))
sys.path.insert(0, HERE)
from jb_pack import read_pack  # noqa: E402

STEPS = ("qkv", "o", "gateup", "down")


class Step:
    """One layer step at one size, from its pack: input q, per-job weights / output q, row map. A split step
    (2026-11-23, real_packs.py DOWN_SPLIT): the input channels in K pieces (pack op 3 = SPLIT), each job reads its
    piece's columns, and the jobs' outputs are partial sums that add up row by row."""

    def __init__(self, path, gids, key, tokens):
        T, ops, x, exp, _ = read_pack(path)
        self.name = os.path.basename(path)[:-4]
        convs = [op for op in ops if op[0] == 1]
        splits = [op for op in ops if op[0] == 3]
        cin = T[convs[0][1]]
        self.sx, self.zx = float(cin["q"][0]), cin["q"][1]
        self.ic = T[splits[0][1]]["dims"][3] if splits else cin["dims"][3]
        K = splits[0][3] if splits else 1
        n = self.ic // K
        self.w, self.sw, self.sy, self.zy, self.idx, self.cols = [], [], [], [], [], []
        per_piece = {}
        for _, i, a, b, o in convs:
            k = i - splits[0][2] if splits else 0
            g = per_piece.get(k, 0)
            per_piece[k] = g + 1
            gid = gids["%s.p%d" % (key, k)] if splits else gids[key]
            m = T[a]["dims"][0]
            self.w.append(np.frombuffer(T[a]["data"], np.int8).reshape(m, n))
            self.sw.append(float(T[a]["q"][0]))
            self.sy.append(float(T[o]["q"][0]))
            self.zy.append(int(T[o]["q"][1]))
            self.idx.append(np.sort(np.nonzero(gid == g)[0]))
            self.cols.append(slice(k * n, (k + 1) * n))
            assert len(self.idx[-1]) == m, (self.name, k, g)
        self.split = K
        self.oc = sum(len(i) for i in self.idx) // K
        self.tokens = tokens
        self.rec_in = np.frombuffer(x, np.int8).reshape(self.ic, tokens).T          # recorded input [T, IC]
        self.rec_out = [np.frombuffer(e, np.int8).reshape(len(i), tokens).T for e, i in zip(exp, self.idx)]
        self._wdq = None

    def wdq(self):
        """Dequantized [OC, IC] weights in model row order (position 0 on the CPU)."""
        if self._wdq is None:
            W = np.zeros((self.oc, self.ic), np.float32)
            for g, i in enumerate(self.idx):
                W[i, self.cols[g]] = self.w[g].astype(np.float32) * self.sw[g]
            self._wdq = W
        return self._wdq


class RefBackend:
    name = "ref (CPU int8 reference)"

    def run(self, k, st, xq):
        out = []
        for g in range(len(st.w)):
            acc = (xq[:, st.cols[g]].astype(np.float64) - st.zx) @ st.w[g].astype(np.float64).T
            out.append(np.clip(np.round(acc * (st.sx * st.sw[g] / st.sy[g])) + st.zy[g], -128, 127).astype(np.int8))
        return out                                                   # per job [T, n_g]

    def close(self):
        pass


class NpuBackend:
    """jb_engine --serve SHM PACK_DIR: compiles every 256-token step, then per request line "<step index>\\n" reads the
    planar input from the shared file's input slot, runs the step on the NPU, reads the outputs straight into the
    output slot (jobs back to back) and answers "ok <npu ms> <read-out ms>\\n"."""
    name = "npu (jb_engine --serve)"
    IN_MAX, OUT_MAX = 4 << 20, 4 << 20

    def __init__(self, engine, pack_dir, tokens, offline=False):
        # /run/user/<uid> (tmpfs): bwrap's empty /dev of the offline run hides /dev/shm
        self.shm = "/run/user/%d/jb_serve_%d" % (os.getuid(), os.getpid())
        env_prefix, extra = [], []
        if offline:
            shim = "/home/jordan/Documents/AI_Projects/npu-shader/shim/libnpu_etna_shim.so"
            import tempfile
            self.dumps = tempfile.mkdtemp(prefix="jb_serve_shim_", dir=os.path.dirname(self.shm))
            env_prefix = ["bwrap", "--dev-bind", "/", "/", "--dev", "/dev", "env", "LD_PRELOAD=" + shim,
                          "NPU_SHIM_DUMP_DIR=" + self.dumps]       # the shim's submit dumps (else into the cwd)
            extra = ["--offline"]
        with open(self.shm, "wb") as f:
            f.truncate(self.IN_MAX + self.OUT_MAX)
        self.mm = np.memmap(self.shm, np.int8, "r+")
        self.p = subprocess.Popen(list(env_prefix) + [engine] + extra + ["--serve", self.shm, "--tokens", str(tokens), pack_dir],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        line = ""
        while not line.startswith("ready"):
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("jb_engine --serve exited before ready")
            print("  engine: " + line.rstrip(), flush=True)
        self.npu_ms = self.ro_ms = 0.0

    def run(self, k, st, xq):
        T = xq.shape[0]
        self.mm[:st.ic * T] = np.ascontiguousarray(xq.T).reshape(-1)
        self.p.stdin.write("%d\n" % k)
        ans = self.p.stdout.readline().split()
        if not ans or ans[0] != "ok":
            raise RuntimeError("jb_engine: step %s: %r" % (st.name, ans))
        self.npu_ms += float(ans[1])
        self.ro_ms += float(ans[2])
        out, o = [], self.IN_MAX
        for i in st.idx:
            n = len(i) * T
            out.append(np.array(self.mm[o:o + n]).reshape(len(i), T).T)
            o += n
        return out

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(10)
        finally:
            os.unlink(self.shm)
            if getattr(self, "dumps", None):
                import shutil
                shutil.rmtree(self.dumps, ignore_errors=True)


class PackQuant:
    """smollm.Quant stand-in: every linear step from the packs through a backend; position 0 on the CPU."""
    mode, fused, quant_head = "w8a8", True, False

    def __init__(self, steps, order, backend, check=False):
        self.steps, self.order, self.be, self.check = steps, order, backend, check
        self.start = 0
        self.t_be = self.t_q = 0.0
        self.worst_in = self.worst_out = 0
        self.n_in_diff = 0

    def linear(self, key, x, W):
        from smollm import _w
        if key == "head":
            return x @ _w(W).T
        st = self.steps[key]
        t0 = time.monotonic()
        y = np.zeros((x.shape[0], st.oc), np.float32)
        if self.start == 0:                                   # position 0: the CPU's (attention sink)
            y[0] = x[0] @ st.wdq().T
            xs = x[1:]
        else:
            xs = x
        assert xs.shape[0] == st.tokens, "chunk must be %d tokens after position 0" % st.tokens
        xq = np.clip(np.round(xs / st.sx) + st.zx, -128, 127).astype(np.int8)
        t1 = time.monotonic()
        lv = self.be.run(self.order[key], st, xq)
        t2 = time.monotonic()
        if self.check:
            d = np.abs(xq.astype(int) - st.rec_in.astype(int))
            self.worst_in = max(self.worst_in, int(d.max()))
            self.n_in_diff += int((d > 0).sum())
            if d.max() == 0:                                  # same input -> outputs must match the pack
                self.worst_out = max(self.worst_out, max(int(np.abs(a.astype(int) - b.astype(int)).max())
                                                         for a, b in zip(lv, st.rec_out)))
        o = 1 if self.start == 0 else 0
        for g, i in enumerate(st.idx):                        # split steps: the partials add up
            y[o:, i] += (lv[g].astype(np.float32) - st.zy[g]) * st.sy[g]
        self.t_be += t2 - t1
        self.t_q += (t1 - t0) + (time.monotonic() - t2)
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("ref", "npu"), default="ref")
    ap.add_argument("--engine", default="/home/jordan/Documents/AI_Projects/npu-llm/bin/jb_engine")
    ap.add_argument("--json")
    ap.add_argument("--save-logits")
    ap.add_argument("--jitter", type=int, default=0, help="K extra ref-backend runs with a relative 1e-7 random nudge "
                    "on the embeddings: the format's noise floor (rounding ties spread), to judge other runs by")
    ap.add_argument("--offline", action="store_true", help="npu backend against the fake NPU (plumbing only: "
                    "the shim executes nothing, so the levels and the quality are meaningless)")
    ap.add_argument("model_dir")
    ap.add_argument("pack_dir")
    a = ap.parse_args()
    MD, PD = os.path.abspath(a.model_dir), os.path.abspath(a.pack_dir)
    sys.argv = [sys.argv[0], MD]                     # l2g_outliers reads the model directory from argv at import
    import l2g_outliers as G
    import l2_quality
    from l2_quality import logsoftmax, score
    from smollm import SmolLM, Quant
    from tokenizers import Tokenizer

    t0 = time.monotonic()
    man = json.load(open(os.path.join(PD, "manifest.json")))
    gids = np.load(os.path.join(PD, "groups.npz"))
    TOK = 256
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    m = SmolLM(MD)
    l2_quality.T = TOK + 1
    _, evals = l2_quality.texts(tok)
    fp = {tn: logsoftmax(m.forward(ids, Quant("fp32"))) for tn, ids in evals.items()}
    what = G.rotate(m, m.layers)
    assert what == man["rotation"], (what, man["rotation"])
    steps, order = {}, {}
    for li in range(m.L):
        for s in STEPS:
            key = "%d.%s" % (li, s)
            steps[key] = Step(os.path.join(PD, "L%02d_%s_t%d.jbp" % (li, s, TOK)), gids, key, TOK)
            order[key] = len(order)
    # the packs' weights must be the rotated model's weights (same rotation seed): within half a level
    d = m.layers[0]
    Wr = np.concatenate([d["q"](), d["k"](), d["v"]()])
    st = steps["0.qkv"]
    err = max(float(np.abs(Wr[i] - st.w[g] * st.sw[g]).max() / st.sw[g]) for g, i in enumerate(st.idx))
    print("%s: %d steps loaded from %s; %s; layer 0 qkv weights vs rotated model: max %.3f levels; fp32 + load %.0f s"
          % (man["model"], len(steps), PD, what, err, time.monotonic() - t0), flush=True)
    assert err <= 0.5 + 1e-3

    if a.backend == "ref":
        be = RefBackend()
    else:
        be = NpuBackend(a.engine, PD, TOK, offline=a.offline)
    res = {"model": man["model"], "backend": be.name, "tokens_per_chunk": TOK, "texts": {}}
    saved = {}
    try:
        t = time.monotonic()
        worst_step, bad = 0, []
        for key, st in steps.items():                         # (1) every step on its recorded input
            lv = be.run(order[key], st, st.rec_in)
            d = max(int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(lv, st.rec_out))
            worst_step = max(worst_step, d)
            if d > 1:
                bad.append("%s %d" % (st.name, d))
        res["steps_vs_pack_max"], res["steps_bad"] = worst_step, bad
        print("steps on their recorded inputs: max %d levels from the packs, %d steps > 1 (%.1f s)%s" % (
            worst_step, len(bad), time.monotonic() - t, ": " + ", ".join(bad[:8]) if bad else ""), flush=True)
        if isinstance(be, NpuBackend):
            be.npu_ms = be.ro_ms = 0.0
        for tn, ids in evals.items():
            q = PackQuant(steps, order, be, check=(tn == "gpl3"))
            t = time.monotonic()
            lg = m.forward(ids, q)
            wall = time.monotonic() - t
            lp = logsoftmax(lg)
            saved[tn] = lp.astype(np.float32)
            s = score(fp[tn], lp, np.asarray(ids))
            r = {**s, "wall_s": wall, "backend_s": q.t_be, "quant_scatter_s": q.t_q,
                 "other_cpu_s": wall - q.t_be - q.t_q}
            if isinstance(be, NpuBackend):
                r["npu_ms"], r["readout_ms"] = be.npu_ms, be.ro_ms
                be.npu_ms = be.ro_ms = 0.0
            if q.check:
                r["input_vs_pack_max"], r["input_levels_differing"] = q.worst_in, q.n_in_diff
                r["output_vs_pack_max"] = q.worst_out
            fs = score(fp[tn], fp[tn], np.asarray(ids))
            r["fp32_ppl"] = fs["ppl"]
            res["texts"][tn] = r
            print("%-7s ppl %.2f (fp32 %.2f)  top1=fp32 %.1f%%  KL %.4f   wall %.2f s = backend %.2f + quant/scatter "
                  "%.2f + other CPU %.2f%s" % (tn, s["ppl"], fs["ppl"], 100 * s["top1_vs_fp32"], s["kl_vs_fp32"],
                                              wall, q.t_be, q.t_q, wall - q.t_be - q.t_q,
                                              "   (NPU %.1f ms, read-out %.1f ms)" % (r["npu_ms"], r["readout_ms"])
                                              if "npu_ms" in r else ""), flush=True)
            if q.check:
                print("        (info) glue inputs vs the packs' recorded inputs: max %d levels, %.1f%% of levels differ "
                      "(rounding ties spread, see the docstring)" % (q.worst_in, 100.0 * q.n_in_diff / (
                          sum(s.ic for s in steps.values()) * TOK)), flush=True)
    finally:
        be.close()
    if a.jitter:
        # the noise floor: the same ref computation with a 1-ulp-sized nudge on the embedding output
        rb, pe, rows = RefBackend(), m.post_embed, []
        for seed in range(a.jitter):
            rng = np.random.default_rng(1000 + seed)
            m.post_embed = lambda x, rng=rng: pe(x) * (1 + 1e-7 * rng.standard_normal(x.shape)).astype(np.float32)
            for tn, ids in evals.items():
                s = score(fp[tn], logsoftmax(m.forward(ids, PackQuant(steps, order, rb))), np.asarray(ids))
                rows.append({"seed": seed, "text": tn, **s})
        m.post_embed = pe
        res["jitter"] = rows
        for tn in evals:
            pp = np.array([r["ppl"] for r in rows if r["text"] == tn])
            kl = np.array([r["kl_vs_fp32"] for r in rows if r["text"] == tn])
            t1 = np.array([r["top1_vs_fp32"] for r in rows if r["text"] == tn])
            res["jitter_" + tn] = {"ppl_mean": pp.mean(), "ppl_sd": pp.std(ddof=1) if len(pp) > 1 else 0.0,
                                   "ppl_min": pp.min(), "ppl_max": pp.max(), "kl_mean": kl.mean(),
                                   "kl_min": kl.min(), "kl_max": kl.max(), "top1_mean": t1.mean()}
            print("jitter %-7s %d runs: ppl %.2f +- %.2f (%.2f .. %.2f)  KL %.4f (%.4f .. %.4f)  top1 %.1f%%" % (
                tn, len(pp), pp.mean(), res["jitter_" + tn]["ppl_sd"], pp.min(), pp.max(), kl.mean(), kl.min(),
                kl.max(), 100 * t1.mean()), flush=True)
    if a.save_logits:
        np.savez_compressed(a.save_logits, **saved)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
        print("wrote " + a.json)
    if a.offline:
        return 0                                         # plumbing only
    return 0 if res["steps_vs_pack_max"] <= 1 else 1


if __name__ == "__main__":
    sys.exit(main())

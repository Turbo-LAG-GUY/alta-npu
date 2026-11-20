# l2g_margin_smoke.py (2026-11-20, CPU only): smoke test behind l2g_outliers.py row B - per-job output scales with
# groups by |w| vs by output range, with/without headroom, on the rotated SmolLM2-135M, 32 tokens per text.
import sys, time
sys.argv = ["x", "/home/jordan/Documents/AI_Projects/npu-llm/models/SmolLM2-135M-Instruct", "32"]
sys.path.insert(0, "/home/jordan/alta-npu/llm/ref")
import numpy as np
import l2g_outliers as g
from smollm import SmolLM, Quant
from tokenizers import Tokenizer
import l2_quality
from l2_quality import logsoftmax, score
l2_quality.T = 32
tok = Tokenizer.from_file(g.MD + "/tokenizer.json")
m = SmolLM(g.MD)
calib, evals = l2_quality.texts(tok)
refs = {tn: logsoftmax(m.forward(ids, Quant("fp32"))) for tn, ids in evals.items()}
g.rotate(m, m.layers)
rec, chin, chout = g.calibrate3(m, calib)
kw = dict(quant_head=False, fp_first=True)
for label, gb, pj, mg in [("R per-step out", "w", False, 1.0), ("per-job out, groups by |w|, x1.0", "w", True, 1.0),
                          ("per-job out, groups by |w|, x1.5", "w", True, 1.5), ("per-job out, groups by out, x1.5", "out", True, 1.5),
                          ("per-job out, groups by out, x2.0", "out", True, 2.0)]:
    Quant.WGROUPS = 16; Quant.WQ.clear(); g.QuantG.GRP.clear()
    q = g.QuantG("w8a8", True, rec, chout=chout, group_by=gb, per_job_out=pj, margin=mg, **kw)
    r = [score(refs[tn], logsoftmax(m.forward(ids, q)), np.asarray(ids)) for tn, ids in evals.items()]
    print("%-36s KL %.4f / %.4f  top1 %.1f / %.1f%%  ppl %.2f / %.2f" % (label, r[0]["kl_vs_fp32"], r[1]["kl_vs_fp32"],
          100 * r[0]["top1_vs_fp32"], 100 * r[1]["top1_vs_fp32"], r[0]["ppl"], r[1]["ppl"]), flush=True)

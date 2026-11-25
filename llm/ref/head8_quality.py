#!/usr/bin/env python3
"""head8_quality.py (2026-11-25, CPU only, x86 PC): what the int8 head (jb_prefill JB_HEAD8=1) costs, on the float
model's own final hidden states: logits from the bf16 table (fp32 math) vs from the per-row int8 table with int16
activations (jbg_head_i8's arithmetic: x -> rint(x / sx), sx = max|x| / 32767; E8 = rint(E / s_v), s_v = max|E_v| / 127;
exact integer dots). Usage: head8_quality.py MODEL_DIR [TEXT ...] (default: every l2_quality eval text)"""
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "ZEN")
import numpy as np  # noqa: E402

MD = os.path.abspath(sys.argv[1])
names = sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], MD]
import l2_quality  # noqa: E402
from l2_quality import logsoftmax  # noqa: E402
from smollm import SmolLM, Quant  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(MD + "/tokenizer.json")
l2_quality.T = 257
_, evals = l2_quality.texts(tok)
m = SmolLM(MD)
V = m.embed.shape[0]
E = np.concatenate([m.embed.rows(np.arange(a, min(a + 8192, V))) for a in range(0, V, 8192)]).astype(np.float32)
s = np.abs(E).max(1) / 127.0
s[s == 0] = 1.0
E8 = np.rint(E / s[:, None]).astype(np.int8)
grab = {}
m.pre_head = lambda x: grab.__setitem__("x", x) or x
for tn in names or list(evals):
    ids = evals[tn]
    m.forward(ids, Quant("fp32"))
    x = m.rms(grab["x"], m.norm).astype(np.float32)                      # [T, D]
    lb = (x @ E.T).astype(np.float64)
    sx = np.abs(x).max(1) / 32767.0
    xq = np.rint(x / sx[:, None]).astype(np.int64)
    l8 = (xq.astype(np.float64) @ E8.T.astype(np.float64)) * s[None, :] * sx[:, None]   # exact in float64
    pb, p8 = logsoftmax(lb), logsoftmax(l8)
    kl = (np.exp(pb) * (pb - p8)).sum(1)
    tgt = np.asarray(ids[1:])
    nll_b, nll_8 = -pb[np.arange(len(tgt)), tgt].mean(), -p8[np.arange(len(tgt)), tgt].mean()
    print("%-12s KL mean %.2e max %.2e  top-1 agree %.1f%%  ppl bf16 %.3f int8 %.3f" %
          (tn, kl[:-1].mean(), kl.max(), 100 * (lb.argmax(1) == l8.argmax(1)).mean(), np.exp(nll_b), np.exp(nll_8)))
    sys.stdout.flush()

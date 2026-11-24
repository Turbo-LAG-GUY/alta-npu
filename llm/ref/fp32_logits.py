#!/usr/bin/env python3
"""fp32_logits.py (2026-11-23, CPU only): full-precision log-probabilities of one eval text (l2_quality.texts, 257
tokens), for scoring jb_prefill's logits. Usage: fp32_logits.py MODEL_DIR TEXT OUT.npy"""
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "CORTEXA73")
import numpy as np  # noqa: E402

MD, TEXT, OUT = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], MD]
import l2_quality  # noqa: E402
from l2_quality import logsoftmax  # noqa: E402
from smollm import SmolLM, Quant  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(MD + "/tokenizer.json")
l2_quality.T = 257
_, evals = l2_quality.texts(tok)
ids = evals[TEXT]
lp = logsoftmax(SmolLM(MD).forward(ids, Quant("fp32"))).astype(np.float32)
np.save(OUT, lp)
np.save(OUT.replace(".npy", "-ids.npy"), np.asarray(ids, np.int32))
print("wrote", OUT, lp.shape)

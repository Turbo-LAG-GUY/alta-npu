#!/usr/bin/env python3
"""step6i_pairs.py RUNDIR (CPU only): for each step 6i U8/I8 pair, NPU U8 - 128 vs NPU I8 (must be byte-identical:
same hardware job), and each saved output vs its CPU reference (max / mean / count off)."""
import json, glob, os, sys
import numpy as np

rd = sys.argv[1]
M = "/home/jordan/Documents/AI_Projects/npu-shader/models"
refs = {"id768_i8": M + "/layout/conv_1x32x768_i8", "ln64_off": M + "/layernorm/ln_64x384_i8"}
for f in sorted(glob.glob(os.path.join(rd, "*.npy"))):
    n = os.path.basename(f)[:-4]
    ref = refs.get(n, M + "/int8nn/" + n)
    got = np.load(f).astype(np.int32).reshape(-1)
    want = np.load(ref + "_reference.npz")["output"].astype(np.int32).reshape(-1)
    d = np.abs(got - want)
    line = "%-10s vs CPU: max %d mean %.3f off %d/%d" % (n, d.max(), d.mean(), int((d > 0).sum()), d.size)
    if n.endswith("_i8") and os.path.exists(f[:-6] + "u8.npy"):
        u = np.load(f[:-6] + "u8.npy").astype(np.int32).reshape(-1) - 128
        line += " | NPU u8-128 vs i8: max %d" % np.abs(u - got).max()
    print(line)

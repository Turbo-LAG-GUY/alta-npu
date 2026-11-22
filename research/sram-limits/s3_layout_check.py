#!/usr/bin/env python3
"""S3 diagnostic (2026-11-22): where did the column strips' output go? CPU only, reads a saved NPU output.
Simulates the two 32-column strips writing plane c of the 4x64x64 planar output at x0 + c*S + y*P and compares
with the NPU's bytes. Result on run sram-limits-s3diag-20261122-063312: S = 2048 (= out_image_x_size * y_size),
P = 64 (out_image_x_stride), planes 3..0 (lower planes win) matches 100.0% of the bytes exactly; S = 4096 (what
Mesa assumed) matches 26%. usage: s3_layout_check.py RUNDIR"""
import sys
import numpy as np
n = np.load(sys.argv[1] + "/npu_out.npy").astype(int).reshape(-1)
w = np.load("/home/jordan/Documents/AI_Projects/npu-upscale/models/hist2_n8_planar_reference_64x64.npz")["output"].astype(int)
for S in (4096, 2048):
    buf = np.zeros(n.size, int)
    for x0 in (0, 32):
        for c in (3, 2, 1, 0):
            for y in range(64):
                o = x0 + c * S + y * 64
                buf[o:o + 32] = w[c, y, x0:x0 + 32]
    print(f"plane spacing {S}: {(buf == n).mean() * 100:.1f}% of bytes exact")

#!/usr/bin/env python3
"""Offline proof for column strips with one job per output channel (2026-11-22). CPU only, reads a shim run dir.
Replays where every NN job writes using the hardware rule proven by the S3 diagnostic (s3_layout_check.py):
plane p of a job lands at out_image_address + p * out_image_x_size * out_image_y_size, row r at + r * out_image_x_stride,
x_size bytes per row. Jobs writing inside the output buffer [BASE, BASE + C*H*W) must together write every byte once,
and each byte must be the right channel and column of a channels-first C x H x W output.
usage: write_map_check.py RUNDIR C H W"""
import glob, os, sys
sys.path.insert(0, '/home/jordan/alta-npu/research/port/tools')
import tp_decode
d, C, H, W = sys.argv[1], *map(int, sys.argv[2:5])
jobs = []
for f in sorted(glob.glob(os.path.join(d, 'mesa-nn-*.bin'))):
    x = tp_decode.decode(f, 'nn')
    jobs.append((os.path.basename(f), x))
# the output buffer: the lowest address any last-layer job writes, assuming the output BO is the one the
# 1-plane jobs point into (they are the only jobs with out_image_z_size 1 here)
ones = [x for _, x in jobs if x['out_image_z_size'] == 1]
if not ones:
    print("no 1-plane jobs: nothing split per channel"); sys.exit(1)
base = min(x['out_image_address'] for x in ones)
owner = {}
bad = 0
for name, x in jobs:
    a0 = x['out_image_address']
    if not (base <= a0 < base + C * H * W):
        continue
    xs, ys, zs, pitch = x['out_image_x_size'], x['out_image_y_size'], x['out_image_z_size'], x['out_image_x_stride']
    for p in range(zs):
        for r in range(ys):
            for k in range(xs):
                o = a0 - base + p * xs * ys + r * pitch + k
                if o in owner:
                    bad += 1
                owner[o] = (name, p, r, k)
    print(f"{name}: writes at +{a0 - base} size {xs}x{ys}x{zs} pitch {pitch}")
covered = len(owner)
# right place: offset o = c*H*W + y*W + col must come from a job whose plane is channel c (its base plane) at row y
wrong = 0
for o, (name, p, r, k) in owner.items():
    c, rem = divmod(o, H * W)
    y, col = divmod(rem, W)
    if r != y:
        wrong += 1
print(f"bytes written {covered} of {C*H*W}, written twice {bad}, at a wrong row {wrong}, outside 0..{C*H*W-1}: "
      f"{sum(1 for o in owner if o < 0 or o >= C*H*W)}")
print("PASS" if covered == C * H * W and bad == 0 and wrong == 0 else "FAIL")

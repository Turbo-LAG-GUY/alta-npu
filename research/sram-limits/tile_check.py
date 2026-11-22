#!/usr/bin/env python3
"""List every dumped NN config of an offline run: image x/y, tile x/y, strides; flag a tile wider than its image
(the hang signature of incident 2026-11-12-1003-STRIPS-C64). Also counts submits. usage: tile_check.py RUNDIR"""
import glob, os, sys
sys.path.insert(0, '/home/jordan/alta-npu/research/port/tools')
import tp_decode
d = sys.argv[1]
bad = 0
files = sorted(glob.glob(os.path.join(d, 'mesa-nn-*.bin')))
for f in files:
    x = tp_decode.decode(f, 'nn')
    w, h, tw, th = x['out_image_x_size'], x['out_image_y_size'], x['out_image_tile_x_size'], x['out_image_tile_y_size']
    flag = tw > w
    bad += flag
    print(f"{os.path.basename(f)}: out {w}x{h} tile {tw}x{th} in_x_stride {x['in_image_x_stride']} "
          f"out_x_stride {x['out_image_x_stride']}{'  <-- TILE WIDER THAN IMAGE' if flag else ''}")
print(f"{len(files)} NN configs, {bad} with a tile wider than the image; "
      f"{len(glob.glob(os.path.join(d, 'shim_submit_*.stream')))} submits")

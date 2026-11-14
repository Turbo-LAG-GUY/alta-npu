# Incident 2026-11-12 10:03 — STRIPS-C64: NPU hang, recovered by the kernel (STOP latch set)

**What ran:** one npu-attempt job, approved by Jordan ("yes you have the OK to run the test and models"):
npu_detail_test.py, 20 invokes, model `npu-upscale/models/hist2_n8_planar_64x64.tflite` (3 convolutions, channels-first
input and output, 64x64), test library `mesa-strips/build/src/gallium/targets/teflon/libteflon.so.a7058018-strips`
(patch 0005, first build with column strips), `ETNA_ML_STRIPS=32 ETNA_ML_STRIPS_VERBOSE=1 ETNA_ML_AXI_SRAM=all
ETNA_ML_DISK_CACHE=0`. Boot 46d8402b, Steam closed, audit clean (audit-20261112-092619).

**What happened:** each of the 20 invokes hung the NPU. The kernel logged `etnaviv-gpu ff100000.npu: recover hung GPU!`
20 times (10:03:46 to 10:03:52), recovered it each time, and the program finished; npu-run set runs/STOP (rc 90).
No MMU fault, no other kernel errors, no freeze. Afterwards: NPU runtime status active, nothing holds the NPU's
render node, the board answers normally. The output is wrong (saved as out-STRIPS-C64.npy).

**Evidence here:** kernel-thisboot.txt (journalctl -k -b 0 at 10:05), the run directory and its capture (kmsg, observe,
steps), devcoredump-listing.txt (devcd1 existed at 10:05:16; not saved, it needs root and expires after ~5 minutes).

**What was new in this job:** the strip pass split the chain into 2 strips of 32 *columns*. Each convolution's input or
output was a window of a wider buffer (in_image_x_stride / out_image_x_stride 64 while the strip is 33-35 wide) and
the strips pad only at the frame's left and right edges (in_image_x_offset 0 inside). Every earlier strip job in this
session used strips of *rows* and passed (STRIPS-T64, T64B, T540-8, T540-16; the row-window jobs gave wrong output but
no hang).

**Likely cause (not proven):** Mesa's tiling code takes the tile width from the operation's height field (TFLite H =
64 here), so each column strip got out_image_tile_x_size 64 on an image only 32-35 wide. Full frames and row strips
never have a tile wider than the image. The x-stride windows (never used by Mesa before) are the other suspect.

**What I changed afterwards (no NPU):** the strip pass no longer makes column strips unless ETNA_ML_STRIPS_COLUMNS=1 is
set (see CHANGELOG). No NPU job has run since.

**For Jordan:** review this, then (per the handoff, 12.1) reboot if you want a clean NPU, and `rm ~/alta-npu/runs/STOP`
when you're happy for NPU work to continue.

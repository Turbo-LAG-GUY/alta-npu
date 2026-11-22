# AXI SRAM known limits: fixes (2026-11-22)

Status: fixed and checked **offline** (fake-NPU shim, empty /dev). NPU run 1 (2026-11-22 06:12, Jordan's OK): S1 MobileNet
PASS; S2 failed without a kernel line: the shared tensor came back in the NPU's channels-first layout (Mesa, upstream
too, detransposed only outputs that nothing inside reads; so `ETNA_ML_AXI_SRAM=0` was never a real workaround: it gave
wrong data silently). Fixed with a detransposed copy the CPU reads (`etna_ml_tensor::cpu_copy`); libraries now kit
1593e603, strips e962cc17 (`*-sram-fixes2`), offline PASS (`offline-copyfix-*`). NPU run 2 (06:20): S1 PASS, S2 PASS (max 1 vs CPU),
S3 column strips **no longer hang** (20 invokes; was 20 of 20 hangs) but give wrong output (mean 53 levels): still off by
default. **Cause found** (diagnostic job SRAMFIX-S3D, output saved, `s3_layout_check.py`: 100.0% of bytes explained): the NN
unit spaces output planes by out_image_x_size x out_image_y_size (2048 for a 32x64 strip), ignoring both strides, while
out_image_x_stride does work as the row pitch. So channel 0 is exact and the higher planes overlap (lower ones win).
Fix built (patch 0007, mesa-strips `libteflon.so.3d1c6999-perchannel`): the last layer of a column strip runs as one NN
job per output channel (z = 1, a slice of the output-channel-major weights and biases each), writing its plane at
c x H x W + first column. Offline (`offline-perchannel-*`): `write_map_check.py` replays every job with the proven rule and
finds each output byte written once in the right place (64x64: 2 strips; 960x540 n8: 9 strips of 107 columns; n16:
17 of 57); everything else byte-identical. **Proven on the NPU 2026-11-22** (`perchannel-ladder.sh`, all 5 PASS, no hang):
64x64 max 1 vs CPU; 960x540 planar n8 12.36 ms whole vs 12.17 with column strips; n16 22.84 vs **14.32** (1.6x).
Column strips now follow `ETNA_ML_STRIPS` by default (`ETNA_ML_STRIPS_COLUMNS=0` off): `libteflon.so.4ab532ed-columns`,
same commands as the proven 3d1c6999 (`offline-colsdefault-*`). All three limits closed.

Kit switched 2026-11-22 06:3x: mesa-axi-sram/build/.../libteflon.so = 1593e603 (41b08ede kept as libteflon.so.41b08ede-kit). The kit path (`mesa-axi-sram/build/.../libteflon.so`) still holds the tested 41b08ede library, and
`.alta-env` is unchanged.

| Limit | Cause found | Fix | Offline evidence |
|---|---|---|---|
| 1. Not exclusive | An SRAM tensor is safe only inside one hardware job. Besides `npu_no_batching`, two more ways broke that: `npu_parallel` (operations of a job overlap, but the allocator reuses space by operation order), and libdrm flushing the command buffer early when it passes 0x4000 words, which silently splits a partition into two submits | SRAM refused under `npu_no_batching` and `npu_parallel`, and for partitions whose commands could pass 0x4000 words (bound: 10 words per NN job, 10 x TP cores + 2 per TP job, + 64). Invoke flushes what is queued first if the batch wouldn't fit after it, and warns once if a split is ever seen | Forced 540-strip split: old library put 2160 tensors in the SRAM over 2 submits; new library refuses (61624 words > 16384) and uses RAM |
| 2. Shared outputs crash | Teflon didn't tell the driver which tensors leave the partition | Backport of upstream Mesa's `pipe_tensor::is_external_output` (p_state.h); Teflon sets it around `ml_subgraph_create`; the placement keeps those tensors in RAM. The abort stays as a safety net | `branch_32x32.tflite`: old kit aborts (rc 134), new runs |
| 3. Column strips hung | Offline replay of the hung job: all 6 NN jobs had a 64-wide tile on a 32-34-wide image, the V7 tiling bug (tile from TFLite H, image x = W) fixed and proven on the NPU 2026-11-14 (upstream series 0002) | 0002 applied to mesa-strips. Column strips stay opt-in (`ETNA_ML_STRIPS_COLUMNS=1`) until S3 passes | Same job: 0 of 6 tiles wider than the image (34/33/32 wide) |

No-change checks: MobileNet (kit) and 540p row strips auto (strips): all NN/TP configs, compressed weights and submit
streams byte-identical to the old libraries.

Libraries: kit `libteflon.so.1593e603-sram-fixes2` (mesa-axi-sram, 41b08ede + patch 0006), strips
`libteflon.so.e962cc17-sram-fixes2` (mesa-strips, ad2c9c5c + 0006 + upstream 0002). Old ones kept:
`libteflon.so.41b08ede-kit`, `libteflon.so.ad2c9c5c-strips`. Patch: `patches/mesa/0006-...diff`.
Changing p_state.h means a full teflon rebuild (~9 min per tree at -j2).

Files: `make_branch_model.py` (`--exact`: the CPU layer is MAXIMUM(t1, t1), first output, for teflon_check),
`offline_check.sh` (shim runner), `tile_check.py` (decodes dumped NN configs), `offline-final-*.VERDICT.txt`.

After the ladder passes: point the kit at 1593e603 (copy it over `mesa-axi-sram/build/.../libteflon.so`, which a
`ninja` there would produce anyway), and drop the column-strip opt-in if S3 passes. Not covered: the mesa-llm port of
patch 0002 (llm/l3/axi-ladder.sh, written 2026-11-22) uses its own invoke path; check it splits nothing across submits
before that ladder runs. True cross-process exclusivity (one SRAM owner at a time) would need a kernel change; with
one submit per invoke it isn't needed for Mesa.

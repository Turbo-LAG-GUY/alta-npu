# Incident 2026-11-14 01:27 - S6F-W1-NN (W = 1 localization probe, boot 273eb301)

- Job: research/port/step6f-ladder.sh run 2, rung 8 = npu-attempt S6F-W1-NN = teflon_check.py -n 1 on
  npu-shader/models/layout/conv_64x1x384_u8.tflite ([1,64,1,384], one 1x1 CONV) with ETNA_ML_SHADER=transpose:
  shader-core transpose, NN conv, shader-core detranspose (Jordan's OK "You have the OK to run the ladder", after
  incident 2026-11-14-0020-S6F-C64x1-OFF, reboot and STOP clear).
- Result: "recover hung GPU!" once (npu-run's new stop-at-first-hang worked: one invoke, job stopped 0.15 s after the
  line), no MMU fault, output all zeros (wrong). npu-run rc 90, STOP set. Board up.
- **devcoredump saved: devcoredump.bin** (289088 bytes, copied by Jordan at ~01:31; root-owned until chown'd).
- Rungs before it in the same run, all PASS, 0 kernel lines: C8x8-OFF, C1x64-OFF, C1x50-OFF, T384-OFF (exact),
  C1x64-TR (same 16-thread shader transposes as this job, byte-identical), and **S6F-W1-TP**: the hung model's TP
  transpose (config identical word for word, incl. addresses) + shader MEAN, exact vs the CPU, no hang.
- **Localized: the W = 1 hang is the NN job.** With both TP jobs replaced by proven shader grids, the NN conv of
  [1,64,1,384] alone hangs; its TP transpose alone does not. Its NN config = the kit library's (only the weights
  address differs), so the kit library has the same hang for [1,H,1,C] convs.
- Prime suspect (offline, incident 2026-11-14-0020 NOTES): etnaviv_ml_nn.c swaps width/height for the image sizes,
  but etna_ml_calculate_tiling_v7 computes the tile from the unswapped dims, so the W = 1 NN image is 1 x 64 with a
  64 x 1 tile (tile x > image x). The H = 1 case that works has a 1 x 24 tile on a 64 x 1 image (overshoot in y).
- Next: decode the devcoredump (hung FE address -> which state/command), then Jordan's choice: guard [1,H,1,C] NN
  ops off the NPU, or compute the tiling from the swapped dims (would also change other non-square tilings: check
  offline that every proven model's config stays identical), or canonicalize pointwise W = 1 convs to H = 1.

## devcoredump decoded (2026-11-14, research/port/tools/devcoredump_decode.py; cmd.bin = the hung submit, decoded with
## research/port/cmdstream_decode.py)
- Submit order: shader transpose grid (CL_KICKER) -> flush + FE->PE stall -> NN job (PS_NN_INST_ADDR 0xfffba000) ->
  flush + 2 stalls -> shader detranspose grid.
- FE_DMA_ADDRESS 0x2280 = cmd offset 0x280: the FE got past the NN kick and the stalls behind it and sits at the
  start of the detranspose grid's uniform LOAD_STATE (SH_HALTI5_UNIFORMS). FE_DMA_STATUS 1, DEBUG_STATE 5.
- HI_IDLE_STATE 0x7ffffff6: everything idle except FE (bit 0) and bit 3 (SH): the core is still busy, so the FE
  waits to write the next grid's shader state. The FE->PE stall evidently doesn't wait for the NN job to finish.
- HI_AXI_STATUS 0x20, no MMU fault (MMU object present, no fault registers set in the log).
- Reading: the NN job never completes. With incident 2026-11-14-0020 (TP + NN + TP, no shader grid at all, also hung
  every invoke) the common factor is the W = 1 NN conv; the busy "SH" bit is most likely how the NN cores show up in
  the idle register. The identical sequence with H = 1 (S6F-C1x64-TR) completes.

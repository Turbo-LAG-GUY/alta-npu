# Incident 2026-11-14 00:20 - S6F-C64x1-OFF (partition-output detranspose fix, boot 8f7f01ab)

- Job: research/port/step6f-ladder.sh rung 3 = npu-attempt S6F-C64x1-OFF = npu-shader/teflon_check.py -m
  npu-shader/models/layout/conv_64x1x384_u8.tflite -e mesa-compute libteflon.so (5a70661e) -n 10, ETNA_ML_SHADER
  unset (Jordan's OK "You have the OK, run the ladder"). Model: one 1x1 CONV_2D, [1,64,1,384] uint8 (W = 1).
  Submit: TP transpose (NHWC -> planes), NN 1x1 conv, TP detranspose (new with the fix).
- Result: "recover hung GPU!" on EVERY invoke (10x, 00:20:39-00:20:46, ~0.6 s each = the hangcheck), no MMU fault;
  teflon_check went on invoking after each recovery (rc 3: output all zeros, the detranspose never wrote it).
  npu-run rc 90, STOP set. Board up. devcoredump devcd1 existed at 00:22:36 but expired (~00:25) before Jordan could
  copy it (root only); Jordan then OK'd sudo for that purpose for the rest of the session.
- Rungs before it PASSED: S6F-C8x8-OFF (3D control, stream = kit library) and **S6F-C1x64-OFF ([1,1,64,384], H = 1,
  with the new detranspose): max 1 / mean 0.20 levels vs the CPU (a channel-plane misread would be ~100), 10/10
  identical, 2.64 ms - the detranspose fix works on the NPU for transformer-shaped [1,1,T,C] outputs.**
- Offline (runs/cs-offline/step6f-detranspose/tpdump, configs decoded with research/port/tools/tp_decode.py):
  - the TP DETRANSPOSE config of the hung model is identical, word for word incl. addresses, to the one that passed
    in rung 2 (the fix maps H = 1 onto the [1,T,1,C] form);
  - the TP TRANSPOSE and NN configs equal what the KIT library builds for the same model (NN differs only in the
    weights address, allocation order). So the kit library would issue the same two jobs for a W = 1 tensor; the
    hang is in etnaviv's existing handling of [1,H,1,C], not introduced by the fix (not proven on hardware: the kit
    library was not run on this model);
  - W = 1 never ran on this board before (6e's [1,1,T,C] models are H = 1). New in this job vs the passing H = 1
    one: TP transpose with y size 1 / z 64 / out_loop_0_count 1, and an NN config whose image is 1 x 64 but whose
    tile is 64 x 1: etnaviv_ml_nn.c swaps width/height for the image sizes (SWAP before filling the map) but
    etna_ml_calculate_tiling_v7 computes the tile from the unswapped dims, so the tile x (64) exceeds the image x
    (1). In the passing H = 1 case the tile overshoots in y instead (1 x 24 on 64 x 1). Prime suspect: the NN job.
- Not localized (the devcoredump would have shown the hung job). Ways to localize later, each a possible hang:
  W = 1 MEAN on the graph input with ETNA_ML_SHADER=mean (TP transpose + SH, no NN); conv_64x1x384 with
  ETNA_ML_SHADER=transpose (only the NN job left on the NN/TP units).
- Changes after this: none to the driver yet. The ladder drops the W = 1 rungs (C64x1-OFF/-TR).
- Next: reboot (npu-run refuses this boot), Jordan clears runs/STOP after review; options for W = 1 go to Jordan
  (guard [1,H,1,C] off the NPU, or canonicalize pointwise ops to the proven H = 1 form, or localize first).

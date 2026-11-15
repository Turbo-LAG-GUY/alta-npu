# 2026-11-15 07:44 - NPU hang in Mesa's test_teflon, family ds_cnn, test 0014 (int8 FULLY_CONNECTED)

- Run: `research/port/teflon-suite-ladder.sh` (unit alta-tfs2), upstream build lib a05298e9 (main 8a9009e0482 + 8
  patches), Jordan's OK "you have my OK to run the NPU when you want". Run dir `runs/teflon-suite-20261115-074242`.
- Before it, same boot (64cd6b68): mobilenet_v1 32/32 passed, mobilenet_v2 66/66, posenet_mobilenet_v1 29 of 33,
  inception_v1 83 of 84. No kernel messages in those.
- Hang: `teflon/ds_cnn.0014`, a single FULLY_CONNECTED, int8, input [1,1,1,276] -> output 12 (weights 12x276,
  zero points -128 / 0 / 34). Kernel 07:44:09 (uptime 107646.6): "recover hung GPU!", offending task test_teflon.
  The test took 1061 ms and the NPU output was all 0x80. npu-run stopped the job (rc 90) and set runs/STOP; the
  ladder stopped. Board stayed up.
- ds_cnn 0000-0013 passed before it (only the two RESHAPE tests of those use the NPU).
- This is the only FULLY_CONNECTED in the suite that reaches the NPU, and the first FC job this kit has run.
- devcoredump: /sys/class/devcoredump/devcd1 (needs root; command given to Jordan at once).
- Not yet known: whether upstream without the series hangs here too (int8 FC goes through the weight path
  patches 3 and 4 change), and the cause.
- Files: `ds_cnn.stdout` and the earlier families' outputs, `run/` (npu-run record), `capture/` (kmsg, observe).

## Follow-up 2026-11-15 afternoon (boot 94e82477)
- The devcoredump was lost (nobody with root was awake; it expires after about 5 minutes). Since then
  `scripts/devcd-watch` (root, started before NPU ladders) copies every dump to `incidents/devcoredumps/`.
- Offline (fake NPU, `runs/cs-offline/fc-hang`): for this test the unpatched upstream library (385f828f) and the
  series library (a05298e9) send identical command streams and the same NN config except
  kernel_cache_end_address (6656 vs 7168); the compressed weights differ (padding kernels and the sign shift,
  patches 3 and 4). The job is configured as a 12x1x23 image with one 12x1x23 kernel per output, 2 kernels per
  core on 8 cores, so cores 6 and 7 hold only padding kernels. Cause still unknown; not rerun on the NPU yet.
- The rest of the suite then ran without a hang with this test left out
  (upstream/etnaviv-ml-fixes/evidence/teflon-suite-20261115).

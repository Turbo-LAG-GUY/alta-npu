# 2026-11-15 14:38 - NPU hang with the UNPATCHED upstream library in teflon/micronet_vww.whole

- Run: `research/port/teflon-baseline-ladder.sh` (unit alta-tfb1, boot 94e82477), Jordan's OK "run the baseline
  comparison after the suite finishes". Library: Mesa main 8a9009e0482 without the series
  (libteflon.so.base-8a9009e0, sha 385f828f). Run dir `runs/teflon-baseline-20261115-143652`.
- Purpose: rerun the 27 tests that failed with the series library (a05298e9) and are not in Mesa's
  expected-failure list, to see which failures the series causes.
- Before the hang, same run: deeplabv3 0009, 0052, 0056, 0060, efficientdet_lite0.whole and inception_v1.whole
  FAILED with the baseline too (wrong output, no kernel message) = these six are not caused by the series.
- Hang: `teflon/micronet_vww.whole` (ten int8 NN ADD jobs; every convolution is per-channel and stays on the CPU).
  Kernel uptime 1548.88 and 1549.92: "recover hung GPU!" twice, offending task test_teflon. npu-run rc 90, STOP set,
  ladder stopped, board up.
- With the series library the same test ran 27 minutes earlier in the same boot WITHOUT a hang (wrong output:
  CPU 09 c2, NPU 80 7f). So the unpatched driver hangs the NPU here and the series does not.
- devcoredump: `devcoredump.bin` (37624 bytes), copied automatically by scripts/devcd-watch (first dump of a
  Teflon hang we have). FE DMA address 0x1188 inside the ring, idle state 0x7ffffff6, 4 BOs. Not analysed further.
- Not run (21 tests): movenet_lightning/thunder 0115 0118 0151 whole, ssd_mobilenet_v2 0011 0052 whole,
  ssd_mobilenet_v3.whole, yolox_nano 0001 0104 0114 0124 0127 0130 0133 whole. Needs a reboot.
- Files: `compare.tsv`, the family outputs, `run/` (npu-run record), `capture/`.

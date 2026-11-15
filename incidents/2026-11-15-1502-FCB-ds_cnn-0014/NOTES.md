# 2026-11-15 15:02 - NPU hang with the UNPATCHED upstream library in teflon/ds_cnn.0014 (int8 FULLY_CONNECTED)

- Run on purpose, last NPU job of boot e67a48e7, Jordan's OK "run the fully connected test with the baseline
  library last". Library: Mesa main 8a9009e0482 without the series (libteflon.so.base-8a9009e0, sha 385f828f),
  one test through scripts/npu-attempt (label FCB-ds_cnn-0014).
- Result: "recover hung GPU!" at uptime 1187.49, offending task test_teflon; npu-run rc 90, STOP set, board up.
- Meaning: the hang of 07:44 (incidents/2026-11-15-0745-TFS-ds_cnn, series library) is NOT caused by the series:
  unpatched main hangs on the same test. A FULLY_CONNECTED on this NPU (V7) hangs in upstream.
- The job (offline dump, runs/cs-offline/fc-hang): input [1,1,1,276] is configured as a 12x1 image with 23
  channels, kernel 12x1x23, 12 outputs, 2 kernels per core on 8 cores. Cause not analysed.
- devcoredump: `devcoredump.bin` (45944 bytes), saved by the boot service alta-devcd-watch without a password.
- Files: `ds_cnn.0014.stdout`, `run/`, `capture/`.

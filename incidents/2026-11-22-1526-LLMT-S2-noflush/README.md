# Incident 2026-11-22 15:26 — LLMT-S2: nn_layer_flush = 0 hangs the NPU (recovered, STOP latched)

- Ladder: llm/l3/tile-ladder.sh synthetic 135m (Jordan's OK), boot d6e90679, driver f317fc8f (job-queue build),
  hw_job_limit 4 at the start. S1 (A2 pack, tile 16x8 + keep-image) PASSED: 0.75 ms NPU, max diff 1.
- S2 = the same pack + ETNA_ML_NO_FLUSH_INDEPENDENT=1 (jb_run 674d417f): job 0 of 2 with nn_layer_flush = 0, job 1
  flushed. Nothing else in the configs differs (offline-A2-noflush-dumps; tp_decode diff = nn_layer_flush only).
- Every invoke hung: "MMU 0 fault (page not present) addr 0x00000000", "recover hung GPU!", 20 recoveries
  (hw_job_trips 0 -> 20), hw_job_auto lowered hw_job_limit 4 -> 1. Board stayed up; the driver recovered each time.
  npu-run rc 90, STOP set: NO NPU JOBS UNTIL A REBOOT (and review). Devcoredump: devcoredump.bin (alta-devcd-watch).
- Reading: with nn_layer_flush = 0 the NN unit does not just run the next job; it fetches something from address 0
  (probably a link to a next layer / config it expects in the same descriptor, i.e. the field is part of a chained
  "layer list" mode, not a cache flush). Do NOT retry nn_layer_flush = 0 without understanding that mode.
- Follow-up: the hook is now refused unless ETNA_ML_NO_FLUSH_INDEPENDENT=I-know-it-hangs; tile-ladder.sh no longer
  runs the no-flush rungs.

# Incident 2026-11-24 04:07: half-ladder H2, 128-token 1.7B packs hung the NPU (2 recovered hangs, STOP)

Ladder: llm/l3/half-ladder.sh (Jordan's OK: "You have my OK to run the NPU"), boot cbbc9ac1, driver f317fc8f,
jb_engine 751f40b7 (mesa-llm 4eb662fc + keep-image + the gated tile rule), env ETNA_ML_KEEP_IMAGE_CACHE=1
ETNA_ML_V7_TILE_FIT=262144. Purpose: design (b)'s question - how fast is the NPU on 128-token images vs 256?

## What happened
- H1 (1.7B, first 4 layers, 256 tokens as 16 x 16, packs real-1p7b-split4-L4-a built on the x86 PC): **PASS**,
  max diff 1, 154.2 ms per round (0.602 ms per token; rsplit-ladder S2 2026-11-23: 170.7 ms), read-out 29.2 ms.
- H2 (the same weights, 128 tokens as 8 x 16, packs -L4-a): compiled 16 steps in 8.3 s, then in the first round
  **two recovered hangs** ("recover hung GPU!", 04:07:08 and 1.1 s later), hw_job_auto dropped hw_job_limit 4 -> 1,
  hw_job_trips 2; npu-run stopped it at the first kernel error (rc 143 -> 90) and latched runs/STOP (04:07:14). No
  MMU fault, no kernel WARNING. H3 (128 as 16 x 8) did not run.
- The devcoredump of the first hang was saved by alta-devcd-watch (incidents/devcoredumps/cbbc9ac1-up34374-devcd1.bin,
  copied here as devcoredump.bin); the second hang's dump was not kept (one devcoredump at a time).
- The fake-NPU checks before the ladder (npu-llm/runs/offline-half-a, -b) compiled and submitted all 16 steps of both
  128-token sets (17 submits each); the fake NPU executes nothing, so it cannot see a hang.

## The hung job (devcoredump)
- The submit is layer 0's **down** step: 16 NN jobs = 4 input pieces (SPLIT, 2048 channels each) x 4 range-grouped
  jobs. The front end stopped at byte 0x228 of the command buffer, right after triggering **job 12** (config BO
  0xfaf64000; the first, largest job of the 4th piece: 1127 output channels, weights in a 2.5 MiB BO). Jobs 0-11
  were triggered before it.
- Its NN config vs job 8 (the same position in the 3rd piece, 1092 channels, completed): identical apart from the
  addresses, the output channels and the per-job scales: tile 8 x 16 (the image is configured 8 x 16 after the
  v7 swap, interleave 2), image_caching_mode 1, kernel_caching_mode 2 (partial), kernel pattern 1 of 20 (low
  0x00080000, msb 19); pieces 1-2 have a different partial-kernel pattern (msb 61, high 0x2aaaaaa0). Job 12 reads the
  LAST 256 KiB of the input BO (0xfc0ef000..0xfc12f000, the BO's end), as the 4th piece does at 256 tokens too.
- So no single field of job 12 stands out. What is NEW in H2 vs every proven LLM configuration: the whole 128 x
  2048 image (exactly 256 KiB) kept in the on-chip SRAM's upper half (0x40000..0x80000, ETNA_ML_KEEP_IMAGE_CACHE)
  with an 8-wide tile and the kernels streamed partially through the lower half (0x800..0x40000). At 256 tokens the
  gated tile rule tiles these steps instead (16-wide tiles of 8 rows, also 256 KiB), which ran fine (H1, prefill3-5).
  Both 128-token shapes give 8-wide tiles (Mesa tiles the shorter side), so H3 would have hit the same path.

## Not the cause (checked)
- The 21-kernels-per-superblock limit (2026-11-19 hangs): calc_superblocks caps 1x1 convs at nn_accum_buffer_depth / 3
  = 21; at this tile height it gives 8 kernels per superblock.
- Memory: MemAvailable 2585 MB at the start, no swap pressure (16 s run).
- The packs themselves: built by the same real_packs.py as the proven 256-token packs (same weights and groups as H1's
  256-token packs in the same directory); expected vs the simulator's levels max 1, pack graph vs expected max 0.

## State after
- Board up, driver recovered both hangs; hw_job_limit 1 (auto, trip 2, trip_from 1 - the 0006 overwrite: the limit
  before was 4); runs/STOP latched (left for Jordan's review). By the kit's rule npu-run refuses NPU jobs after a hang
  until a reboot.

## Follow-ups (none run; offline / Jordan's call)
- Reboot (clears the driver state), Jordan reviews + clears runs/STOP.
- Design (b)'s question is still open: the 128-token step time was not measured. Before any retry, find a 128-token
  configuration on the proven path offline: e.g. ETNA_ML_V7_TILE_FIT=131072 (two tiles of 8 rows -> 128 KiB image
  tiles) or without the kept image (ETNA_ML_KEEP_IMAGE_CACHE unset), compare their NN configs with the proven
  256-token ones, then a single-step jb_run of one down job before a whole layer.
- Mesa side (test tree): find why an 8-wide tile with a full-SRAM-half kept image and partial kernel caching hangs
  (dump the configs of jobs 8 and 12 with ETNA_MESA_DEBUG=ml_dbgs offline; compare with the 135M's kept images, which
  are smaller: 256 x 576 = 144 KiB).

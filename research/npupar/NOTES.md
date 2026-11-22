# npu_parallel test (2026-11-21)

Question (Jordan, 2026-11-21): "are we using the NPU properly because a lot of the work we've sent to it was very
serial rather than parallel". Estimates (memory note npu-utilization-serial): LLM 1 token ~0.4% of the best measured
rate, 256 tokens ~25%, upscaler 8 ch ~4%, MobileNet ~9%. One serial point in the driver: every operation inside a
submit runs strictly after the previous one unless `ETNA_MESA_DEBUG=npu_parallel` is set.

## History
- Mesa d6473ce28ee0 (2023-11-16) shipped parallel mode ON (switch `npu_no_parallel`).
- Mesa 1277f58d8a4a (Tomeu Vizoso, 2024-04-25, MR 28918): "Make parallel jobs disabled by default. It doesn't work
  for all models, with the same happening to the proprietary driver. There may be some hardware limitation on at
  least the HW that is currently supported in Mesa." The MR page couldn't be read (bot wall), so whether "doesn't
  work" meant wrong output or hangs is unknown.

## What the switch changes (offline, fake NPU, kit library mesa-axi-sram)
Evidence: runs/cs-offline/npupar/ (mn-*, u8-*: 1 invoke; n8-*10, n16-*10: 10 invokes; *.diff decoded).
- GL_NN_CONFIG: 0x10 (SMALL_BATCH) -> 0x0
- PS_NN_INST_ADDR / PS_TP_INST_ADDR: config address + op ID in the low bits (default: + 0)
- PS_UNK10A4: op ID (default 0)
- GL_FLUSH_CACHE at the end of the batch: 0xc23 -> 0x403 (drops UNK11 and SHADER_L1)
- Nothing else; buffer lists identical.

## Hazard: the op ID never resets
`static unsigned i = 0;` in etna_ml_subgraph_invoke (etnaviv_ml.c) counts across partitions and invokes for the
life of the process. MobileNet: partition 1 uses IDs 1-30, partition 2 31-32, the second invoke continues at 33.
The TP code uses 0x1f as a marker for non-last TP cores in parallel mode, so the ID field may be 5 bits; IDs above
30 may then collide with the marker or spill into the address bits (the NPU would read its instruction from the
wrong offset). Not proven, but it is a plausible cause of upstream's "doesn't work for all models": any model
reaches it after enough invokes. The ladder keeps every ID <= 30.

## Ladder
research/npupar/npupar-ladder.sh (6 rungs, upscaler nets only, see its header). Results below.

## Results (2026-11-21 15:17-15:20, boot f9492f52, runs/npupar-ladder-20261121-151757)
All 6 rungs PASS; kernel log clean, no STOP. Parallel output byte-identical to default in every invoke.

| rung | mode | invokes | median ms | min-max |
|---|---|---|---|---|
| U8-BASE | default | 10 | 13.63 | 12.31-24.85 |
| U8-PAR1 | parallel | 1 | 16.17 | (first invoke) |
| U8-PAR10 | parallel | 10 | 13.24 | 11.43-14.87 |
| U8-BASE2 | default | 10 | 13.14 | 11.86-16.07 |
| U16-BASE | default | 10 | 25.16 | 22.32-25.92 |
| U16-PAR10 | parallel | 10 | 24.63 | 22.28-26.56 |

Reading: no gain beyond noise on the upscaler nets (3 large layers that depend on each other, ~4-8 ms each: little to
overlap). Where it could matter: many small operations per submit (MobileNet's 7x7 tail, LLM steps). Those exceed
ID 30, so the next step would be a test-tree fix (op ID reset per invoke, kept within 1-30), an offline check, then
a MobileNet ladder. Not started; needs Jordan's OK.

# Width test (2026-11-21 15:28-15:31, boot f9492f52, runs/width-ladder-20261121-152825)
Jordan: "go ahead with the upscaler width test". Models: npu-upscale/width_models.py (random Gaussian weights, real
hist2 layout 5 -> N -> N -> 4, 3x3, planar 960x540). Offline (runs/cs-offline/width): synthetic n8's NN configs =
the real n8's; kernels per core 1 / 2 / 4 / 4 (n64: two kernel rounds, image caching on); tiles 64x10 for all.
Ladder research/npupar/width-ladder.sh: 8 rungs, all PASS (every invoke = first, CPU max diff 1), kernel log clean.

| width | MACs/pixel | math vs n8 | median ms | time vs n8 | GOPS | % of 1.73 TOPS | NPU-bound FPS (2 in flight) |
|---|---|---|---|---|---|---|---|
| 8 (real) | 1224 | 1x | 13.50 | - | 94 | 5% | 74 |
| 8 | 1224 | 1x | 13.02 | 1x | 97 | 6% | 77 |
| 16 | 3600 | 2.9x | 23.95 | 1.84x | 156 | 9% | 42 |
| 16 (real) | 3600 | 2.9x | 24.57 | 1.89x | 152 | 9% | 41 |
| 32 | 11808 | 9.6x | 41.76 | 3.2x | 293 | 17% | 24 |
| 64 | 42048 | 34x | 72.26 | 5.6x | 603 | 35% | 14 |

Reading: time grows roughly in step with width (x1.7-1.8 per doubling) while the math grows with its square, so the
NPU gets more efficient as the net widens (6% -> 35%). Synthetic and real nets time the same (13.0 vs 13.5, 24.0 vs
24.6). For a real-time hook the budget decides: 16 channels fits ~40 FPS, 32 ~24 FPS, 64 ~14 FPS (NPU time only,
upper bounds; the live n8 hook does 50-52 FPS with 14 ms of NPU time, so other costs already limit it).
Quality so far (memory npu-game-upscaler, chooser net, squared-error loss): 16 channels beat 8 by ~0.4 dB
(trained 44.00 vs 43.55, unseen 37.69 vs 37.32). 32 and 64 channels were never trained (Mali trainer: 54 ms/step at
8 channels, 412 ms at 16x4 layers), so whether they help is open.

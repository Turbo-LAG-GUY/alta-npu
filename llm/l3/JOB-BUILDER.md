# LLM engine job builder / runtime (L3b) - design, started 2026-11-19 on the fake NPU

## Why
- The L3 ladder proved Teflon's per-job tiling is already at the hardware limit for one token (kernels per core
  above 21 hang the NPU: incidents/2026-11-19-1106-LLM3-k24). What is left to win is around the job, not inside it:
  1. **Per-invoke work.** Teflon's `partition_invoke` creates AND destroys a whole pipe_context on every invoke
     (tfl_device.c:1244/1306), on top of the TFLite interpreter. Each NPU step of a token (4 per layer: qkv, o, gateup,
     down; 120 per SmolLM2-135M token) pays it.
  2. **Copies of the weights.** TFLite keeps the model file mapped, Teflon `fill_tensor` mallocs a second copy of every
     constant (tfl_device.c:812) and the driver builds a third (etnaviv_ml_nn.c:587, freed after compile) before the
     NPU buffer. For 1.7B (1.7 GB int8) that does not fit; our runtime passes weights from caller memory, which can be
     freed (or unmapped) right after compile: the NPU buffer is the only copy that stays.
  3. **Several jobs per submit.** Weight channel groups (L2e: G jobs with their own scales), CPU+NPU row splits and
     chained NPU-only steps should share one submit (one kernel round trip) instead of one invoke each.
- No new NN-config or weight compiler: the runtime calls the driver's own `ml_subgraph_create` (Mesa's compiler, the
  configs the NPU already ran), so it inherits every fix and every safety cap.

## What one invoke is (mesa-llm 678775c5 = mesa-upstream-fixes + L3 hooks, read from the source)
- `ml_subgraph_invoke`: (first time only `init_npu`: 1 submit with PA_SYSTEM_MODE / GL_API_MODE = OpenCL), 8 zero
  words, the input copied into its buffer (int8: a per-byte +128 loop), then per NN job 4 states (OCB remap start/end,
  NN_CONFIG = SMALL_BATCH, PS_NN_INST_ADDR = the job's 4 KiB config buffer, UNK10A4), then close_batch (2 cache
  flushes). Nothing is submitted yet.
- `ml_subgraph_read_output`: `flush` (= the submit ioctl), map the output buffer (= wait for the fence), copy out (int8:
  -128 per byte).
- A gateup job (576 -> 3072, one token) is 96 bytes of commands and 4 buffers (config, 1.8 MB weights, input,
  output) at fixed softpin addresses, no relocations.
- Several ops in one subgraph are emitted back to back with no barrier between them; the hardware runs them in order
  (this is how all 28 MobileNet ops already run in one submit).
- Graph inputs are channels-last for the driver; a conv wants channels-first, so the driver adds a TP transpose job
  unless the graph starts with an explicit TRANSPOSE (which it turns into a bypass: the planar trick). It adds it even
  for 1-pixel inputs, where both layouts are the same bytes (etnaviv_ml.c:388 does not check `layouts_differ`): a
  possible small upstream fix that would save a TP job on every one-token NN job without explicit transposes.

## Built so far (CPU only, fake NPU)
- `jb_pack.py`: job packs (.jbp): tensors + ops (1x1 conv, explicit planar transposes), test input, expected outputs.
  Modes: `tflite` (an L1c model's conv, its weights and XNNPACK reference), `groups G` (output-channel groups, own
  weight scale each), `chain` (gateup -> a dependent second conv). Packs in npu-llm/models/jb/.
- `jb_run.c` (build `build_jb_run.sh`, binary npu-llm/bin/jb_run, sha 2b12a1ed): loads a pack, `get_ml_device` ->
  `ml_subgraph_create` -> frees its weight copies -> N x (invoke [x batch] + read_output), one persistent context or
  one per invoke (`--ctx per-invoke`, Teflon's way). Refuses `--offline` without the shim and a device run with it.
- `jb_offline.sh`: jb_run against the shim inside bwrap with an empty /dev (JB_DUMPS=1 driver dumps, JB_NOSTRACE=1
  for timing).

## Offline results (npu-llm/runs/offline-jb-20261119)
- **Byte-identical to Teflon**: jb_run on the smol_gateup_1x1 pack vs Teflon on smol_gateup_1x1_P.tflite (L3 offline
  run, same library): NN config, compressed weights, input buffer, both submit streams and both buffer lists (handles,
  softpin addresses, sizes) identical. So J1 on the NPU runs exactly the job L1c and L3 rung 1 ran.
- Multi-job submits (one submit per invoke in every case):
  | pack | NN jobs / submit | kernels per core | stream bytes | buffers |
  |---|---|---|---|---|
  | gateup, 1 job | 1 | 21 | 96 | 4 |
  | gateup, 4 groups | 4 | 20 | 216 | 13 |
  | gateup, 16 groups | 16 | 12 | 696 | 49 |
  | gateup -> down chain | 2 (2nd reads the 1st's output buffer) | 21, 18 | 136 | 7 |
- CPU side per invoke on the fake NPU (median of 300, while the L2e job loaded the CPU; includes the shim writing
  2 dump files per submit): **Teflon 0.47 ms, jb_run 0.19 ms** for the same job; a context per invoke adds ~0.01 ms on
  the shim (its GEM calls are fake: the real kernel cost is unknown); 16 groups 0.23 ms; chain 0.14; batch 4 0.20.
  The NPU part is not in these numbers; J1 vs J2 on the NPU shows how much of the 0.30 ms per-invoke cost was CPU.
- numpy reference = XNNPACK exactly on the gateup conv (0 of 3072 differ), so groups / chain expectations are sound
  (chain allowed 2 levels: its second job sees the first's rounding).

## NPU results (2026-11-19 14:00, runs/llm-jb-ladder-20261119-135828, Jordan's OK, boot e01fb03e, 7/7 PASS)
| rung | what | max diff | ms per invoke (median of 199) |
|---|---|---|---|
| J1 | gateup 576 -> 3072, one token, jb_run direct | 0 | 0.499 |
| J2 | the same job through Teflon (teflon_check) | 0 | 0.82 |
| J3 | jb_run, context created + destroyed per invoke | 0 | 0.541 |
| J4 | 4 copies of the job in one submit | 0 | 1.508 |
| J5 | 4 weight groups in one submit (same bytes) | 0 | 0.605 |
| J6 | 16 weight groups in one submit | 1 | 0.710 |
| J7 | gateup -> down chain (2 dependent jobs, 3.55 MB) in one submit | 1 | 0.810 |
- **Direct beats Teflon by 0.32 ms per invoke (39%)** on the identical job: 0.28 ms is TFLite + the delegate, 0.04 ms
  the context Teflon creates and destroys per invoke.
- Inside one submit a job costs 0.336 ms = **5.26 GB/s** weight streaming; the **fixed cost per submit is ~0.16 ms**
  (was 0.30 ms per invoke through Teflon in L1).
- Each extra group job costs 35 us (4 groups) / 14 us (16 groups) on top of the same bytes.
- Chaining two dependent jobs in one submit saves ~0.19 ms (one fixed cost) vs two submits.
- SmolLM2-135M estimate (30 layers x 4 steps = 120 submits, 106 MB): Teflon-like 57 ms/token (L1c) -> direct
  **40 ms/token (25 tok/s)**, of which ~20 ms is the fixed cost of 120 submits; 4 groups 52 ms, 16 groups 65 ms.
  CPU llama.cpp: 21 ms/token. So for the 135M the submit count is now the main cost: next = fewer submits per
  token (NPU-side SiLU x mul to chain gateup -> down, CPU + NPU split per step, groups only where quality needs them).

## Plan
| step | what | needs |
|---|---|---|
| R1 | jb_run: direct Gallium ML API, persistent context, weights freed after compile | DONE offline |
| R1-NPU | `jb-ladder.sh`: J1 direct vs J2 Teflon (same job), J3 context per invoke, J4 batch 4, J5/J6 4/16 groups, J7 chain | PASS 7/7 2026-11-19 (see NPU results) |
| R2 | packs from real SmolLM2 weights (safetensors -> int8, memory-mapped), one pack per layer step, all 30 layers; compile time + peak memory (target: weights once) | CPU only |
| R3 | zero-copy activations: write the int8 (+128) input straight into the input buffer and read the output in place (needs a small driver hook to hand out the tensor buffers; removes both copy loops). 2026-11-20: the output loop reads write-combined (uncached) memory byte by byte, ~150 ns per byte on an A53 / ~35 on an A73 (1 ms for 6.6 KB): a bulk copy first is the quick fix. DONE in mesa-llm 2026-11-20 (commit 1c441f99, lib a26f0852; upstream series patch 0010): pipe_buffer_read + convert in cached memory; offline: jobs byte-identical; NPU 2026-11-20 12/12 PASS: read-out x9-12 faster (256-token gateup 25.6 -> 5.6 ms, Teflon 1 token 0.85 -> 0.63 ms); bulk read of the uncached buffer ~0.32 GB/s is the next limit. Cached output buffers 2026-11-20 (lib 6caca26e): NPU 14/14 PASS, big read-outs x1.7-1.8 (256 tokens 2.61 -> 1.48 ms), still ~0.5 GB/s | read-out, cached and fused done (upstream patches 10-12); zero-copy hook built 2026-11-20 (test tree, llm/l3/zero-ladder.sh needs OK) |
| R4 | CPU + NPU split: one layer step's rows partly on the NPU (one submit) and partly on the CPU at the same time (the L1c bandwidth result: ~10.5-11.5 GB/s together) | jb_run ca878c69 `--cpu-rows`, packs + `split-ladder.sh` built offline 2026-11-19; **CLOSED 2026-11-20**: both split ladders correct, but NPU + CPU together read only 7.1-7.4 GB/s (CPU alone 6.7), so a one-token split does not pay (DESIGN.md L3c) |
| R5 | fewer submits per layer: SiLU x mul on the NPU (TP lookup table / eltwise) so gateup -> down is one submit; later attention pieces | research |
- Safety rules for the runtime: only configs from the stock compiler with the L3 overrides unset (kernels per core
  <= 21); every new pack shape runs offline first and its configs are decoded; device runs only via scripts/npu-run
  with Jordan's OK for that step.

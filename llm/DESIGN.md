# LLM engine for the Alta NPU - design and handoff

Started 2026-11-18. Every new session on this work starts here (keep it current). Working name: none yet,
Jordan picks it. Code: `~/alta-npu/llm/`; models and run output: `~/Documents/AI_Projects/npu-llm/` (the big card).

## Goal
Run a transformer LLM (8-bit) with the NPU doing the heavy math, without Teflon/TFLite's limits: our own engine
that builds NPU jobs per layer, keeps one copy of the weights, and splits work between NPU and CPU.

## Hardware facts (verified, not assumed)
Source: Mesa `src/etnaviv/hwdb/amlogic/gc_feature_database.h`, the only GC8000 rev 0x7120 entry (customer 0x88),
plus measurements on this board.

| Fact | Value | Meaning for the engine |
|---|---|---|
| NN data types | int8 x8 cores, int16 x8 cores, no fp16/bf16, no 4-bit (`NN_4BIT_PHASE1` 0) | int8 weights; int16 is a later option for outliers |
| Per-channel quantization | `NN_PER_CHANNEL_QUANT` 0 | one scale per job: group output channels by scale (one job per group) or rescale on the CPU |
| VIP SRAM (on-chip) | 512 KiB | holds activations / weight tiles, not a model |
| AXI SRAM | 1 MiB | same |
| Weight compression | zero run length only (`NN_ZRL_BITS` 5, no Huffman) | dense LLM weights do not compress (offline: zrl_bits 0, +0.5%) |
| NPU address space | 32 bits (4 GiB) | enough for a 1-2B int8 model |
| NPU bus | 16 bytes wide (`AXI_BUS_WIDTH`) | 12.8 GB/s if it runs at the 800 MHz core clock (unverified) |
| CPU memory read | 7.5-8 GB/s measured 2026-11-18, one A73 core almost saturates it | the bar the NPU must beat |
| CPU ISA | ARMv8.0: no dot-product, no fp16 | CPU int8 math is slow, CPU fp32 for small ops |
| RAM | 3.85 GB, ~0.8-1.3 GB free with the desktop | Qwen3-0.6B int8 (~0.6 GB) is the realistic first real model |

## What the physics allows (why the plan looks like this)
- Generating one token reads every weight once. A 0.6B int8 model = ~0.6 GB per token, so
  tokens/s <= bandwidth / 0.6 GB (8 GB/s -> ~13 tok/s ceiling). The 1.5 MiB of on-chip SRAM cannot hold the
  weights, so for ONE token the caches do not change this ceiling.
- The caches pay off when one weight read serves many tokens (M > 1): reading a prompt, and checking several
  guessed tokens at once (speculative decoding: a tiny draft model guesses K tokens, the NPU checks all K in one
  pass over the weights). That is the realistic way to speed up replies.
- Two more possible gains, both to be measured: (1) the NPU may read memory faster than the CPU (16-byte bus);
  (2) the CPU works on attention / norms / sampling while the NPU streams the next layer (overlap).
- Reading fewer weights per token is the one way past the bandwidth ceiling (Jordan's idea, 2026-11-18: a
  perceptron for the cache). A small predictor (perceptron / tiny MLP) guesses from the current hidden state
  which FFN neurons will matter for this token; only those weight rows are read. Prior art: Deja Vu (contextual
  sparsity, MLP predictors), PowerInfer (hot/cold neurons), Apple's LLM in a Flash (predicted FFN rows),
  TEAL (training-free magnitude sparsity, ~40-50% on Llama-3). Natural sparsity is high in ReLU models, much lower
  in SwiGLU models (Qwen3, SmolLM2, Llama 3), so L2 measures it first. On the NPU, skipping needs weights
  pre-compiled in neuron blocks (e.g. 64 output channels each) and per-token job lists of the predicted blocks;
  the VIP/AXI SRAM is software-managed (the driver places data), so any cache policy lives in our engine anyway.
- Byte budget per token, Qwen3-0.6B int8 (computed from its config): 28 layers x (qkv 4.19M + o 2.10M +
  FFN 9.44M) + tied output head 155.6M = ~596 MB. FFN 44%, attention projections 30%, output head 26%.
  (SmolLM2-135M: FFN 59%.) So neuron skipping alone caps at ~1.8x even if the whole FFN were skipped.
- TAGE + perceptron (Jordan, 2026-11-18). Two places it fits:
  (a) neuron blocks: TAGE-style tables over the history of recently active blocks (active neurons repeat across
      neighbouring tokens) + a perceptron on the hidden state as the corrector (as TAGE-SC-L's statistical
      corrector). Fewer bytes per token, capped by the FFN share above.
  (b) tokens: TAGE over the token history = a variable-length n-gram table = a draft model that reads no weights.
      It guesses K tokens, the NPU checks all K in one weight pass (M = K), the perceptron corrector sets how many
      to guess (confidence). More tokens per byte; L1's M = 1 vs M = 16 timing tests "K for the price of 1".
  Tension: when K tokens are checked together, the union of their active neurons is larger, so (a) saves less.
- Memory tiers (Jordan, 2026-11-18: models may spill past RAM, active weights must stay in RAM).
  Measured 2026-11-18 (O_DIRECT, board busy, load ~6): eMMC 58 MB/s sequential / 23 MB/s random 64 KiB
  (2.8 ms each); SD card 47 / 31 MB/s (2.1 ms). RAM is ~8000 MB/s, so a spilled MB costs ~20-40 ms.
  Tier 1, always in RAM: attention projections, output head (tied with the embeddings), norms, KV cache, the hot
  FFN blocks (NPU buffers are pinned while mapped; user mlock limit is 481244 KiB). Tier 2: cold FFN blocks in
  the model FILE, memory-mapped (clean pages drop without a write: no swap writes, no eMMC wear). Rule: a cold
  block predicted active is NOT waited for: skip it this token, prefetch it in the background, promote it if the
  predictor keeps asking for it (the TAGE/perceptron predictor doubles as the cache-replacement policy).
  Example Qwen3-1.7B int8: always-active ~664 MB + FFN 1.06 GB; with ~30% hot FFN ~1 GB in RAM.
- One copy of the weights: the matrix weights live only in NPU format (compiled once, cached on disk, loaded into
  NPU buffers); the CPU keeps only embeddings, norm weights and the KV cache. Streaming weights from disk per token
  is not an option (eMMC/SD ~100-300 MB/s -> < 1 tok/s); the model must fit in RAM once.

## Architecture (planned)
- Linear layers = int8 1x1 convolutions on the NN unit (FULLY_CONNECTED hangs this NPU, also in upstream:
  incidents/2026-11-15-1502-FCB-ds_cnn-0014). Input [1,1,M,IC] -> [1,1,M,OC], M = tokens.
- Our own job builder instead of a TFLite graph: configs and weights per layer shape built once, then each
  token only changes buffer addresses and submits. Start from the etnaviv ML code (NN config + weight encoder,
  `etnaviv_ml_nn_v7.c`), run on our tree, checked offline with the fake NPU (npu-shader/shim) first.
- CPU: RMSNorm, RoPE, attention over the KV cache, sampling, embedding lookup (later candidates for the shader
  core, which already runs softmax / mean / layernorm / lookup tables).
- Quantization: per-tensor int8 weights + activations, with smoothing (SmoothQuant-style scaling) and channel
  grouping if quality needs it. Checked on the CPU in numpy before any NPU work depends on it.

## Resume here (2026-11-22: WHY wider jobs are slower; read before the evening section below)
- **Cause found (CPU only, from the driver source + the offline runs' ML_DBG lines; no NPU job):** Mesa's NN config
  (mesa-llm etnaviv_ml_nn.c create_nn_config, ~line 1194) puts the job's weights ("kernel cache") and its input image
  ("image cache") in the 512 KiB VIP SRAM. When 0x800 + compressed weights >= 512 KiB it takes the PARTIAL branch and
  sets image_cache_size = 0: the input is then NOT cached and is re-read from DDR once per superblock (output-channel
  group). At 256 tokens the v7 tile is the whole 16x16 image (1 spatial tile), so caching the weights buys nothing,
  but losing the image cache costs superblocks x input: 1-job gateup = 48 x 147 KB = 7 MB extra per step.
  DDR model per step (weights + input x (cached ? 1 : superblocks) + output) vs the wide-ladder times: every row
  fits at 2.0-3.4 GB/s; gateup 2 jobs and 1 job both 9.66 MB -> both 2.86 ms (the odd "2 jobs = 1 job" result).
  Which jobs lose the image cache: qkv 1 job (548 KB), gateup 2 / 1 jobs (876 / 1752 KB), down 1 job (869 KB).
  down 4 / 2 jobs (218 / 434 KB weights + 393 KB image > 512 KiB) take the third branch instead: partial weights +
  FULL image, and ran fine on the NPU (their partial-cache patterns ask for more than the 129 KB weight area: the
  hardware coped).
- **FIX 1 BUILT + PROVEN 2026-11-22** (Jordan's OK): mesa-llm hook ETNA_ML_KEEP_IMAGE_CACHE=1 (etnaviv_ml_nn.c, uncommitted
  in the test tree; jb_run 84bd6ece, jb_engine 39c22a60, sources llm/l3/jb_*-<sha>.c). Offline
  (npu-llm/runs/offline-keepimg-20261122-045938, llm/l3/keep_image_offline.sh, VERDICT.txt): hook OFF = upstream byte
  for byte; ON changes only kernel_cache_end + the 3 image-cache fields of the 300 jobs whose weights don't fit (g1 180,
  g2 120; 4-job and mixed sets untouched), weights identical, weight areas 126-501 KB. NPU ladder
  runs/llm-keepimg-ladder-20261122-051706 (llm/l3/keepimg-ladder.sh): 6/6 PASS, every output within 1 level, trips 0.
  ms per 256-token round (all 120 steps, read-out ~30-34 ms included): **1 job 211.3 -> 103.8** (-51%), 2 jobs 177.7 ->
  123.1, 4 jobs (unchanged, same-session baseline) 139.2. One pack: 1-job gateup NPU 0.79 ms (was ~2.85). One token: 1
  job 27.9 ms (OFF 27.2: no change, its input is 9 KB). So FEWER, WIDER jobs are now fastest (each job re-reads the
  input once); jobs per step is a quality choice: 1 job costs quality (135M KL .053 vs .023; breaks 1.7B).
## int16 question (2026-11-23, Jordan, from session ai-projects-f4's bf16 analysis)
- bf16 on the NPU: not possible (hwdb GC8000 rev 7120: NN cores INT8 8 / INT16 8 / FLOAT16 0 / BFLOAT 0, shader fp32
  only); bf16 -> fp32 by `<< 16` is what llm/ref/smollm.py Lazy already does. The int16 NN path (int16 x int16, no
  int16 x int8 mix) is real but Mesa never drives it (format reverse engineering + hang risk, 2x weight bytes).
- Simulation first (llm/ref/l2k_int16.py: today's 1.7B layout, one step kind at a time in 16 bits: input, weights,
  outputs). 135M (KL GPL-3 / README): base .0196 / .0340; qkv 16-bit .0149 / .0279; down 16-bit .0184 / .0319; all
  16-bit .0033 / .0122. 1.7B rows base, qkv, o, gateup, down, all running (unit alta-llm-l2k, log
  npu-llm/runs/l2k-int16-SmolLM2-1.7B-Instruct.log, ~5-6 h). Worth an int16 experiment only if one step kind carries
  most of the 1.7B gap.

## Resume here (2026-11-23 afternoon): C glue built (CPU side of a chunk), next: fusion + the loop in C
- **libjbglue.so** (llm/l3/glue/jb_glue.c, build.sh -> npu-llm/bin/libjbglue.so + frozen libjbglue-<sha>.so; ctypes
  bindings glue/jbglue.py): RMSNorm, quantize + transpose to the NPU's planar int8 (NEON divide + round-to-even +
  16x16 byte transposes, exact like numpy), whole-step scatter of all jobs' outputs back to model order (planar
  buffers transposed in 16x16 NEON blocks, then row by row in L1; split pieces add up; can add straight into the
  residual), whole-step position-0 int8 GEMV from the packs' int8 weights (no fp32 copy: 6.8 GB at 1.7B), RoPE,
  causal attention (4 queries x 16 keys / 16 dims register micro-kernels, transposed K, NEON exp), SiLU * up (NEON
  exp), block FWHT (NEON butterflies). glue/test_jbglue.py: every kernel vs the numpy it replaces at 135M and 1.7B
  shapes: quantize / scatter / FWHT bit-exact, the rest ~1e-7 relative. ALL OK.
- **chunk_glue.py --glue c** (glue/cforward.py: the rotated forward with the C kernels, Python orchestration):
  135M with the int8 reference backend: every step on its recorded input 0 levels from the packs; quality = the numpy
  glue within the noise (GPL-3 15.81 KL .0189 / README 49.03 KL .0342 on real-135m; numpy 15.70 / .0195, 49.09 / .0327).
- **Per-chunk CPU cost** (glue/bench_layer.py: all kernels at real shapes, random int8 in place of the NPU, 4 A73
  threads, OMP_WAIT_POLICY=active): 135M 0.72 s (numpy glue ~21.6 s); **1.7B 3.4 s** (attention 1.05, scatter 0.77,
  position-0 GEMV 0.47, quantize 0.45, SiLU + FWHT 0.43, norm / RoPE / splits 0.2) vs the NPU's ~1.0 s. Memory-traffic
  floor ~0.5 s per chunk. Lessons: OpenMP threads asleep between calls cost up to 3 ms per call (active wait fixes
  it, but spinning threads starve the ref backend's BLAS: use active only with the NPU backend); GCC spills 2-D
  NEON accumulator arrays (named registers: attention 44 -> 29 ms per 1.7B layer); per-element float64 divides.
- **Next for the glue**: fuse the float passes (gateup -> SiLU * up straight from the int8 job outputs; qkv scattered
  into q / k / v; quantize fused into the norms), the position-0 GEMV overlapped with the NPU (independent of it), then
  the whole loop in C inside jb_engine (no Python between steps), then the 135M chunk end to end on the NPU with it.

## Resume here (2026-11-23 00:05): gated tile rule + 1.7B down split CONFIRMED on the NPU
- **Gated tile rule** (mesa-llm etnaviv_ml_nn_v7.c, uncommitted on 4eb662fc): ETNA_ML_V7_TILE_FIT=262144 now tiles
  only when (a) the untiled image can't be cached next to the 0x20000 kernel area and there is more than one
  superblock, and (b) a DDR estimate (untiled = input x superblocks + weights; tiled = input + weights + uncached
  weights x (tiles - 1)) favours tiles. Offline byte-identical to the best measured layouts (1.7B 768/768 configs =
  tile-ladder B3; 135M 960/960 = keep-only). ETNA_ML_V7_TILE_FIT_ALWAYS=1 = the ungated first version.
- **1.7B down split into 4 input pieces of 2048** (llm/l3/jb_pack.py `ksplit SRC K OUT`: the same input and int8
  weights, per-piece jobs with their own 8-bit outputs; the CPU adds the partials into the residual). New pack op 3 =
  channel SPLIT (jb_pack.py, jb_run.c, jb_engine.c: MAX_T 128, MAX_OPS 64): the driver aliases the pieces into the
  one input buffer at k x 512 KiB, no job (offline checked). Packs npu-llm/models/jb/real-1p7b-ksplit (L00-L03 down,
  t256; other steps symlinked). Partials vs the source's outputs within 1.3 levels (19 at 2 outputs the source had
  clipped). Quality (llm/ref/l2j_down_split.py, 135M): K 1 / 2 / 3 KL .0229/.0324, .0180/.0298, .0198/.0291 = no cost;
  1.7B (K 2, 4) running (unit alta-llm-l2j-1p7b, log npu-llm/runs/l2j-SmolLM2-1.7B-Instruct.log).
- **NPU confirmation 2026-11-23 00:03 (Jordan's OK), runs/llm-confirm-ladder-20261123-000247, 4/4 PASS, trips 0**
  (jb_run 3b8f7fdf, jb_engine 751f40b7; the 1.7B sim paused for it): C0 one split-down pack 9.1 ms NPU; C1 135M gated
  132.0 ms per 256-token round (untiled, as intended); C2 1.7B first 4 layers gated 218.3 ms (= B3); **C3 + split
  down 174.6 ms**: down 110.2 -> 55.4 ms (NPU 108.3 -> 47.1; read-out 1.9 -> 7.8 for 4x the outputs). 1.7B 4 layers:
  keep-only 457.5 -> 174.6 ms (2.6x); all 24 layers ~1.05 s per 256-token chunk for the linear steps.
- **Calibrated split packs (2026-11-23 01:00):** real_packs.py MODEL OUT GROUPS DOWN_SPLIT (4th arg K): per-piece jobs
  grouped by max |w| of the piece, output scales from the calibrated partial ranges (l2j piece_ranges), the forward
  simulates the split; expected levels now use float32-rounded scales (as stored; fixed 1 tie in 442368).
  chunk_glue.py reads split packs. 135M K=3 (npu-llm/models/jb/real-135m-split3): offline PASS; ref backend end to end:
  every step on its recorded input 0 levels off; GPL-3 16.18 KL .0210, README 48.35 KL .0292 (unsplit 15.70 / .0195,
  49.09 / .0327; noise band 15.74-16.14 / .0197-.0228, 47.52-49.57 / .0293-.0321) = no measurable cost.
  1.7B K=4 build queued after the 1.7B sim (unit alta-llm-packs-1p7b-split -> npu-llm/models/jb/real-1p7b-split4).
- **1.7B quality of the split (2026-11-23 03:26, l2j, weight-grouped jobs): it COSTS on 1.7B**: KL GPL-3 / README
  one piece .182 / .089; 2 pieces .282 / .129; 4 pieces .251 / .125 (top-1 86.7 -> 82.8%). 4 roundings of partial
  sums, and the 1.7B down output has huge outlier channels that widen the job scales. The calibrated 1.7B split pack
  build was stopped before writing anything. Trying range-grouped jobs (llm/ref/l2j_down_split.py L2J_GROUP=range:
  channels sorted by calibrated output range, uneven groups by DP minimising rows x range^2, >= 8 rows per job):
  135M one piece .0173 / .0308 (w: .0229 / .0324), 3 pieces .0197 / .0396 (w: .0198 / .0291). 1.7B run (4 and 8 jobs,
  K 1 and 4) in unit alta-llm-l2j-range, log npu-llm/runs/l2j-SmolLM2-1.7B-Instruct-range.log (~3.5 h).
- **1.7B range-grouped result (2026-11-23 08:02): the split now IMPROVES quality.** KL GPL-3 / README (ppl):
  today (1 piece, weight groups) .182 / .089 (2.52 / 25.38); 1 piece x 4 range jobs .149 / .096; **4 pieces x 4 range
  jobs .145 / .086 (2.41 / 25.13)**; 1 x 8 .163 / .079; 4 x 8 .139 / .083 (2.40 / 26.27). Pick: 4 pieces x 4 range
  jobs (16 jobs, the shape that ran 2x faster). real_packs.py DOWN_SPLIT=Krange; 135M 3range packs offline PASS (job
  rows per piece e.g. 287 / 192 / 89 / 8: the outliers in 8-33-row jobs, kpc 1..21). 1.7B 4range packs building (unit
  alta-llm-packs-1p7b-split -> npu-llm/models/jb/real-1p7b-split4).
- **1.7B 4range packs (npu-llm/models/jb/real-1p7b-split4, 192, 2 h 7 min): sim max 1, read-back 0, offline PASS**
  (npu-llm/runs/offline-split4r-1p7b-*; down jobs per piece ~1000 / 630 / 300 / 40-100 rows, kpc 8-16).
  **NPU 2026-11-23 12:40 (Jordan's OK), runs/llm-rsplit-ladder-20261123-123937, 4/4 PASS, trips 0:** first 4 layers,
  256 tokens **170.7 ms** (unsplit 218.3, equal-size split 174.6): qkv 31.0, o 13.2, gateup 72.1, **down 52.0** (was
  110.2); 1 token 57.4 ms (unsplit 58.1; down 14.4 vs 18.5). One down pack: 10.1 ms NPU (256 tokens), 3.1 ms (1 token).
  => the calibrated, range-grouped split is better in quality (sim KL .145 / .086 vs .182 / .089) AND faster. This is
  the 1.7B layout now: real_packs.py MODEL OUT 4 4range + ETNA_ML_V7_TILE_FIT=262144 + ETNA_ML_KEEP_IMAGE_CACHE=1.
  All 24 layers: ~1.02 s per 256-token chunk for the linear steps, ~345 ms per token decode (linear steps).
- Next: a whole-model 1.7B run (1.7 GB of pinned NPU weights on a 3.7 GB board: needs a plan), chunk_glue end to end
  on 1.7B (CPU glue in C first: numpy glue is ~10 s per 135M chunk), and range grouping for the other steps (1.7 GB of pinned NPU weights on a 3.7 GB board: needs a
  plan, e.g. the desktop closed, or the 1-token packs dropped from the process).

- **tile-ladder RAN after the reboot 2026-11-22 16:35 (Jordan's OK): 7/7 PASS, trips 0** (runs/llm-tile-ladder-20261122-163507,
  boot b887d66c; load ~3.5 from login indexing, same-run comparisons). Real 1.7B weights, first 4 layers, 256 tokens,
  NPU ms incl. submit, keep-image only (B1) / + tile rule (B2) / rule on IC <= 2048 only (B3):
  qkv 82.5 / 24.3 / 25.0; o 28.7 / 13.9 / 14.8; gateup 211.2 / 52.7 / 53.3; down 111.0 / **137.8** / 116.1;
  round (incl. read-out ~21 ms) 457.5 / 254.5 / **233.4** = 2.0x. The 2048-input steps get 2.1-4.0x; down (IC 8192,
  2 MiB input, 4 jobs x 4.2 MB weights) does NOT: at 16x2 its weights are re-read for each of the 8 tiles (DDR model
  ~36 MB per job). Best now = TILE_FIT=262144 + TILE_FIT_MAX_IC=2048: ~1.4 s per 256-token chunk for all 24 layers'
  linear steps, of which down ~0.7 s -> down is the next target (more superblock kernels per tile, or down split).
  One token (B5): 58.1 ms per 4 layers (~350 ms per token, 1.7 GB of weights at ~4.9 GB/s). Synthetic: S1 0.81 ms,
  S3 2.83 ms (= the PT runs). 135M: tile rule 143.8 vs 140.9 ms (its down at 16x10 is slightly slower): the rule
  should fire only when the whole-image cache can't fit (next: refine it), so the 135M stays untiled.
- **tile-ladder 2026-11-22 15:25 (Jordan's OK, synthetic + 135m): S1 PASS (A2, tile rule: 0.75 ms NPU, = PT); S2 with
  ETNA_ML_NO_FLUSH_INDEPENDENT HUNG every invoke (MMU fault at address 0, 20 recovered hangs, hw_job_limit auto 4 -> 1),
  STOP latched: REBOOT before any NPU job (incidents/2026-11-22-1526-LLMT-S2-noflush). nn_layer_flush = 0 is not safe
  on V7; hook refused in the source unless =I-know-it-hangs; no-flush rungs removed from tile-ladder.sh (left: S1 S3,
  R1 R2, B1 B2 B3 B5). The 135M rungs did not run.**
  1.7B packs DONE: 192 (4 jobs, 1 + 256 tokens, 3.3 GB, 61 min), within 1 level of the simulator; offline PASS for
  the tile rule (2048-input steps 16x8, down 16x2, kpc <= 21), keep-only and rule-without-down layouts
  (npu-llm/runs/offline-tile-1p7b-*). Next after the reboot (Jordan's OK): tile-ladder.sh synthetic 135m 1p7b.
- **2026-11-22 afternoon (Jordan: 1.7B packs with the tile rule, then idea 2, then idea 3):**
  - Tile rule hook ETNA_ML_V7_TILE_FIT=<bytes> (+ _MAX_IC) in mesa-llm etnaviv_ml_nn_v7.c: tile height so one input
    tile x IC <= N (262144: IC 2048 -> 16x8, IC 8192 -> 16x2; 135M down IC 1536 -> 16x10, a new layout; 135M other
    steps unchanged). Byte-identical to ETNA_ML_V7_TILE_H=8 / 4 on the axi-1p7b packs (which ran).
  - Idea 3 step 1 built: ETNA_ML_NO_FLUSH_INDEPENDENT=1 (nn_layer_flush = 0 on every NN job whose next hardware job
    reads nothing written since the last flush; the last job always flushes; a bypass only aliases). Offline: A8 7 of 8
    jobs unflushed, the TP job before the convs keeps its flush. jb_run 674d417f, jb_engine 4002516c (uncommitted in
    mesa-llm on top of 4eb662fc). Research note research/2026-11-22-nn-pipelining-fields.md (all candidate fields).
  - tile_offline.sh (per-pack offline check of every job's layout, kpc, tile <= image, flush pattern): 135M 4-job PASS.
    NPU ladder llm/l3/tile-ladder.sh (synthetic S1-S4, 135M R1-R4, 1.7B B1-B5 with --layers 4) written, NOT run.
  - 1.7B packs: real_packs.py 4 jobs -> npu-llm/models/jb/real-1p7b (llm/l3/packs-1p7b.sh, unit alta-llm-packs-1p7b).
  - **Idea 2 (gateup -> SiLU*up -> down in one submit) in 8 bits: NOT viable** (llm/ref/l2i_npu_mlp.py, 135M; the NPU
    has no MUL on NN/TP: the multiply would be mesa-compute's shader-core 8-bit MUL, SiLU a TP lookup table). KL vs
    fp32 GPL-3 / README: today 4 jobs .0229 / .0324; paired gate/up jobs .0231 / .0357; any 8-bit h before down
    (no Hadamard .0546 / .0790, + 8-bit SiLU and MUL .0551 / .0789, + Hadamard as an 8-bit conv .0671 / .0820);
    8 jobs: today .0161 / .0236 vs 8-bit glue .0488 / .0738; 16 jobs .0155 / .0229 vs .0490 / .0815; pairs grouped by h
    range: .33 / .62 (wrecks the gateup output scales). The loss is the 8-bit h itself (outlier channels quantized
    BEFORE the rotation spreads them), not SiLU or the multiply. Viable form: one shader-core kernel doing SiLU * up +
    the block FWHT + the down input quantization in fp32 (= today's arithmetic), inside the submit; needs the
    mesa-compute shader port in mesa-llm. Cost estimate at 6 GFLOPS: ~8 ms per 1.7B layer (~200 ms per chunk).
- **AXI SRAM ladder RAN 2026-11-22 14:04 (Jordan's OK): 12/12 PASS, trips 0 (runs/llm-axi-ladder-20261122-140434) -
  and the AXI SRAM does NOT pay.** NPU ms per invoke, P0 planar input in DDR / PT tile 16x8 + keep-image / N1 shared TP
  copy into DDR / N2 the copy in the AXI SRAM: A2 (2048 -> 2 x 224, 256 tok) 1.70 / **0.75** / 3.53 / 3.58;
  A8 (8 jobs) 6.57 / **2.81** / 8.51 / 8.87; D4 (8192 -> 4 x 56, 64 tok, 1 superblock per job) 1.73 / 1.76 / 3.62 / 3.85.
  Conv jobs re-reading their input from the AXI SRAM are no faster than from DDR (the NN core's input fetch runs ~3 GB/s
  either way: matches the 2026-11-08 MobileNet result), and the TP transpose of 512 KiB costs ~1.8-1.9 ms (~280 MB/s).
  What pays is the VIP SRAM image cache: ETNA_ML_V7_TILE_H=8 + ETNA_ML_KEEP_IMAGE_CACHE=1 (tile 16x8, the 256 KiB image
  tile fits, kpc 14) = 2.3x on the 2048-input steps. For the 1.7B: tile height so that tile x IC <= ~256 KiB (IC 2048:
  8 rows; down IC 8192: 2 rows) + keep-image; leave the AXI SRAM out of the LLM path. Down (8192 in) gains nothing at
  64 tokens (1 superblock per job: its input is read once per job already).
- **1.7B input in the AXI SRAM (2026-11-22, built + offline PASS, NPU ladder NOT run):** the kernel patch gives the
  CPU no mapping of the AXI SRAM, so the NPU copies the input there: channels-last graph input -> ONE driver transpose
  (TP) job shared by all convs (new hook ETNA_ML_SHARE_INPUT_TRANSPOSE=1; upstream adds one per conv) -> output placed
  in the AXI SRAM by patch 0002 (ported into mesa-llm, ETNA_ML_AXI_SRAM=all; zero-copy read-out guarded too) -> convs
  read it there. Also a guard: an image of EXACTLY 512 KiB (2048 ch x 256 tokens = every 1.7B step but down) made
  upstream set kernel_cache_end = 0 < start 0x800 (a layout never run); now the image is not cached (the 1-job gateup
  layout). jb_run 943bf0c2. Synthetic 1.7B-shaped packs npu-llm/models/jb/axi-1p7b (jb_pack.py synth: A2 / A8 = 2048 ->
  2 / 8 jobs x 224, 256 tokens; D4 = 8192 -> 4 x 56, 64 tokens). Offline matrix (llm/l3/axi_offline.sh,
  npu-llm/runs/offline-axi-20261122-053100): 1 TP job, copy placed (512 of 1024 KiB), convs read the TP output address,
  all cache layouts valid. Alternative without AXI in the same ladder: ETNA_ML_V7_TILE_H 8 + keep-image (tile 16x8:
  the image cache fits, kpc 14). Down at 256 tokens (2 MiB input) does not fit the AXI SRAM: 64-token packs only.
  NPU ladder llm/l3/axi-ladder.sh (12 rungs P0 / PT / N1 / N2 per pack) waits for Jordan's OK.
- **SHELVED by the 14:04 AXI ladder above (SRAM reads no faster than DDR, each piece pays a ~1.8 ms TP copy per 512 KiB): expect pieces to LOSE to W1-P0; run pieces-ladder.sh only to confirm. Kept: the 0006 guards, multi-input jb_run. Token pieces through the AXI SRAM (2026-11-22, built + offline PASS, NPU ladder NOT run; Jordan: "it can shuffle
  data between RAM and SRAM as long as the data is still hot"):** an input bigger than the 1 MiB SRAM (1.7B down at 256
  tokens: 8192 ch = 2 MiB) is cut into P token pieces in ONE submit: per piece a channels-last graph input, one shared
  TP copy into the SRAM, then the piece's conv jobs. A piece's copy lives only until its last conv, so the allocator of
  patch 0002 (already lifetime-aware, no Mesa change needed) gives every piece the SAME SRAM space. DDR per step: input
  once, weights once PER PIECE (the trade). Full 1.7B down (~37 jobs x 56 ch), DDR model: whole 16.8 MB weights + 37 x
  2 MiB input ~ 94 MB; 2 pieces of 128 tokens ~ 36 MB; 4 of 64 ~ 69 MB -> 2 pieces (1 MiB each = the whole SRAM) best,
  if the NN core reads the AXI SRAM faster than DDR (unmeasured). Built: mesa-llm 4b666c08 (the AXI port as found,
  committed) + 4eb662fc (patch 0006's one-submit guards: SRAM refused under npu_no_batching / npu_parallel and past the
  0x4000-word command buffer, pre-flush, split warning); jb_run 21b8ac1f (several graph inputs: the pack's input blob is
  split across them in tensor order; MAX_T 1024, MAX_OPS 512); jb_pack.py `pieces IC OCJ G H W P` (weights shared by
  the pieces). Packs npu-llm/models/jb/axi-pieces: W1 whole 16x16, P2 2 x 16x8, P4 4 x 8x8 (IC 8192, 4 jobs x 56).
  Offline llm/l3/pieces_offline.sh -> npu-llm/runs/offline-pieces-20261122-135940 PASS: job order T nnnn per piece, each
  piece's convs read its own copy, N2 places P of P using one piece's size (1024 / 512 KiB), all copies at one address,
  one submit per invoke, W1 places nothing; reference = direct conv of the token slices (max 0). One-input packs
  byte-identical to 943bf0c2 (offline-pieces-regress-20261122-140010). **Tiling trap found offline:** 8x16 (H < W)
  pieces got a 16-row tile on an 8-row image: the V7 tile is computed from the unswapped sizes and 348aef38 only fixes
  H > W. So pieces are laid out H >= W (16x8, the fixed path that ran on the NPU); an upstream fix (always tile the
  configured orientation) would change non-square configs, not done. NPU ladder llm/l3/pieces-ladder.sh (W1-P0,
  W1-N1, P4-N1, P4-N2, P2-N1, P2-N2) waits for Jordan's OK. Real-model use later needs: weights compiled once per step
  and shared by the pieces (today Mesa builds a coefficient buffer per conv op, so P pieces = P copies of the weights).
- **AXI SRAM (1 MiB): NOT used by the LLM engine** (still true on the NPU: the pieces and 1.7B-input builds above are
  offline only). jb_engine links mesa-llm, which had no AXI SRAM code (that patch
  lives in mesa-axi-sram / mesa-strips, kit Teflon only); the ladders also strip ETNA_ML_AXI_SRAM. Even that patch
  only places INTERMEDIATE tensors between ops of one submit (graph inputs/outputs excluded); a pack has none. The NN
  core's caches address only the VIP SRAM: nothing spills into the AXI SRAM by itself. Kernel switch axi_sram = Y.
- **Fix options (none built yet):** (a) Mesa policy: when the weights don't fit, keep the image cache (kernel area =
  512 KiB - image, partial weights: the config kind down 4/2 jobs already ran) -> predicted 1 job/step ~67 ms NPU
  per chunk vs 98 (4 jobs), and jobs per step becomes a pure QUALITY choice; test tree hook + offline + an NPU
  ladder (Jordan's OK). (b) Put the step input in the AXI SRAM (CPU writes it there), so superblock re-reads hit
  on-chip memory; matters for 1.7B, whose 256-token input (IC 2048 -> 512 KiB, down IC 8192 -> 2 MiB) can never
  fit the VIP SRAM at tile 16x16 (or use ETNA_ML_V7_TILE_H / smaller chunks). (c) Keep the size rule, but with a
  margin: the test is on COMPRESSED weights + 2 KiB (gateup 4 jobs 449024 vs 442368 raw); the 1.7B "sram" layout
  (o / gateup / down jobs exactly 512 KiB raw) would fall into the slow branch.
- 1.7B quality (l2h-mixed, rows 4,1,sram, 100 min): 4 jobs 2.52 / 25.38 (KL .182 / .089), 1 job 7.18 / 43.87 (KL 1.20
  / .73: 1 job breaks 1.7B), sram 24/8/64/32 2.48 / 25.87 (KL .155 / .070) = better than 4 jobs. fp32 2.18 / 24.49.

## Resume here (2026-11-21 evening, new session; read with the handoff below)
- **wide-ladder RAN 2026-11-21 21:13 (Jordan's OK), 7/7 PASS** (runs/llm-wide-ladder-20261121-211302): every output
  within 1 level, no kernel warnings, hw_job_trips 0 (first jb_engine run at hw_job_limit 4; 4-job numbers = the
  afternoon's at limit 1: 137.8 vs 140 ms per chunk, 31.7 vs 32.7 per token).
  Chunk (256 tokens), ms per round (all layer steps) 4 / 2 / 1 jobs: 137.8 / 177.7 / 211.3. Wider is NOT always
  faster: NPU time per step kind (30 steps), 4 / 2 / 1 jobs: qkv 17.6 / **14.3** / 29.7, o 15.1 / 11.5 / **9.7**,
  gateup **30.4** / 85.8 / 85.5, down 35.8 / **27.0** / 49.3. Rule: a job gets faster as it widens ONLY while its
  weights fit the 512 KiB VIP SRAM (best jobs 276-442 KB; past 512 KiB 2-3x slower). Best mix = qkv 2, o 1,
  gateup 4, down 2 jobs -> ~114 ms per chunk estimated (-18%). One token (weight streaming): fewer jobs slightly
  faster everywhere, 31.7 / 29.8 / 27.2 ms. Read-out is 32 ms of the 138 (gateup 17.6: 23.6 MB per chunk).
  X1 = first whole prompt chunk END TO END on the NPU (chunk_glue --backend npu, 4-job packs): GPL-3 ppl 15.86
  (band 15.74-16.14) KL .0173, README 48.25 (band 47.52-49.57) KL .0270: inside the noise band; NPU vs ref top-1
  88.7%, KL(ref||NPU) .020 (= the format's own noise size). NPU side per chunk 0.32-0.38 s through the Python pipe
  (NPU 127-137 ms + read-out 34-42 ms); numpy glue ~12 s.
  Running (CPU): unit alta-llm-l2h-mixed = l2h_groups.py with L2H_ROWS (new: "mixed" = the best mix above;
  QuantG.wgroups now honours Quant.WGROUPS_BY): 135M + 360M rows 4,mixed; 1.7B rows 4,1,mixed (logs
  npu-llm/runs/l2h-mixed-<model>.log). 135M done: 4 jobs 15.81 / 49.24 (KL .0229 / .0324), mixed 15.78 / 47.48
  (KL .0198 / .0333) = no measurable cost (inside the noise band). Mixed packs built (real_packs.py ... mixed ->
  npu-llm/models/jb/real-135m-mixed) + offline PASS (0 problems, kpc max 20); NPU rung `llm/l3/mixed-ladder.sh` (M1
  mixed 256, M2 4-job baseline, M3 mixed 1 token, X2 chunk end to end mixed) written, NOT run (needs Jordan's OK, quiet
  CPU). The first l2h-mixed unit died after 135M (inline systemd-run bash -c: rc came out empty); 360M + 1.7B rerun
  via llm/l2h-mixed-chain.sh. 360M (log l2h-mixed-SmolLM2-360M-Instruct-4-mixed.log): 4 jobs 11.14 / 40.12 (KL .0128 /
  .0215), mixed 11.20 / 41.14 (KL .0159 / .0306) = a small cost, BUT the mix was sized for 135M: on 360M 2 qkv jobs are
  768 KB each, over the SRAM. The rule is a SIZE: jobs per step = ceil(weight bytes / 512 KiB) (l2h row "sram";
  135M = mixed 2/1/4/2, 360M qkv 3 o 2 gateup 10 down 5, 1.7B 24 / 8 / 64 / 32): bigger models get MORE jobs, so
  speed and quality pull the same way there. 360M sram: 11.02 / 39.66 (KL .0153 / .0220) = 4-job quality (better ppl,
  KL level). Running: 1.7B rows 4,1,sram (unit alta-llm-l2h-mixed, ~1 h).
  real_packs.py "mixed" = the 135M sizes only (add "sram" before packing a bigger model). Next: if mixed quality is OK, real_packs with per-step groups + its NPU
  rung; C glue.
- Handoff step (1), wider jobs: packs with 1 / 2 jobs per step built (`llm/l3/real_packs.py MODEL OUT GROUPS`;
  npu-llm/models/jb/real-135m-g1, -g2; 240 each, within 1 level of the simulator) and offline PASS on the fake NPU
  (npu-llm/runs/offline-jb-20261121-real135m-g1 / -g2: 0 problems; kpc max 21 for g1 = L00/L29 gateup_t1, the
  driver's default cap, which ran clean in L3 rung 1; g2 max 20). NPU ladder `llm/l3/wide-ladder.sh` written, NOT
  run (needs Jordan's OK; message ai-projects-60 first): W1-W6 = jb_engine-3b2807b6 on 4 / 2 / 1 jobs at 256 and 1
  tokens, X1 = one chunk end to end on the NPU.
- Step (2) started early, CPU only: `llm/l3/chunk_glue.py` runs the whole 257-token forward with every linear step
  from the packs through a backend (ref = exact CPU int8; npu = `jb_engine --serve SHM`, shared file in /run/user
  because bwrap's empty /dev hides /dev/shm; offline plumbing PASS, 360 steps served). Ref backend, 4-job packs:
  every step on its recorded input within 1 level of its pack; GPL-3 ppl 15.70 (fp32 15.62) KL .0195, README 49.09
  (46.68) KL .0327 = the simulator's quality (npu-llm/runs/chunk-glue-ref-135m.json, -logits.npz).
- Found: the format has a noise floor. A 1-level rounding tie spreads through the block Hadamard and attention
  (layer 0 gateup 2 ties -> layer 1 down 21% of levels differ -> later layers 70-85%, max ~20 levels), so glue inputs
  never match the packs' recorded inputs level for level, and any 1-ulp change moves the result. OpenBLAS
  (DYNAMIC_ARCH) picks its kernel from the core the process STARTS on (cortexa53 vs cortexa73): the same code gave
  GPL-3 ppl 16.13 started on an A53 vs 15.70 on an A73 (bit-identical within one kernel). chunk_glue pins
  OPENBLAS_CORETYPE=CORTEXA73. Earlier simulator runs (l2g, l2h) were not pinned: compare rows within one run only.
  Noise band (`--jitter 6`, 1e-7 nudge on the embeddings, chunk-glue-ref-135m-jitter.json): GPL-3 ppl 15.94 +- 0.18
  (15.74-16.14), KL .0197-.0228; README 48.44 +- 0.83 (47.52-49.57), KL .0293-.0321. Judge NPU runs and job counts
  against this band (135M 2 jobs GPL-3 KL .033 is outside it, README .037 just outside).
- Glue cost (numpy, cProfile): ~10-11 s per chunk without the backend: attention einsum 6.4 s (no BLAS), quantize /
  scatter ~1.6-3 s, fwht 1.1 s; vs 140 ms of NPU linear steps and llama.cpp's ~2.5 s per 256 tokens. The real glue
  must be C (NEON, threads); attention ~2.3 GFLOP per chunk may become the largest CPU cost (Mali / shader core
  later). Prompt reading needs the head for the last position only.
- Still to do: rerun the 1.7B 1-job groups row (CPU, `systemd-run --user --unit=alta-llm-l2h --collect
  llm/l2h-chain.sh SmolLM2-1.7B-Instruct`).

## Resume here (2026-11-21, handoff: context window full)
- State: boot d6e90679 (rebooted 2026-11-21 by the other session's work), nothing of this session running. The NPU
  driver is now the job-queue build f317fc8f (root step 250, by session ai-projects-60): hw_job_limit is settable at
  runtime (now 4; /etc/modprobe.d still says 1 at load). My root step 240 (hw_job_limit=4 via modprobe) is
  SUPERSEDED: do not run it. Another Claude session (ai-projects-60) also does NPU work: message it before NPU runs.
- Done 2026-11-21: llm/l3/jb_engine.c = all 120 steps of SmolLM2-135M compiled in one process (2.5 s) and run back to
  back on the NPU, every output within 1 level (runs/llm-engine-ladder-20261121-153843): 256-token chunk 140 ms of
  linear steps (0.547 ms/token), 1 token 32.7 ms. Groups test (llm/ref/l2h_groups.py, llm/l2h-chain.sh): fewer jobs
  per step cost quality; KL GPL-3 / README for 4 / 2 / 1 jobs: 135M .023/.032, .033/.037, .053/.064; 360M .013/.022,
  .020/.053, .034/.083; 1.7B 4 jobs 2.52/25.38 (KL .182/.089), 2 jobs 2.68/24.67 (.242/.109), 1 job not run (the
  reboot stopped the chain: rerun with `systemd-run --user --unit=alta-llm-l2h --collect llm/l2h-chain.sh
  SmolLM2-1.7B-Instruct`).
- Next, in order: (1) the speed of 1- and 2-job packs on the NPU (real_packs.py with GROUPS = 1 / 2, offline, then
  the engine ladder, Jordan's OK) to decide jobs per step; (2) the CPU glue for a 256-token chunk (RMSNorm, RoPE,
  attention, SiLU x mul, block Hadamard, scatter by groups.npz) and one chunk end to end vs the simulator and
  llama.cpp (pp128 104 tok/s); (3) 1.7B / SmolLM3 packs (wider steps, where the NPU gains most); (4) with the
  job-queue driver: submit ahead / two chunks in flight (the other session found the deeper ring alone gives no
  speedup, 2 threads +13% from CPU overlap).

## Earlier resume note (2026-11-20, end of the session)
- **Priority (Jordan, 2026-11-21): bigger jobs, not driver switches.** ETNA_MESA_DEBUG=npu_parallel is a small lever at
  best (session ai-projects-60's ladder, research/npupar/NOTES.md). The big loss is too little work per job: one token
  at a time uses about 0.4% of the NPU's best measured rate (a 256-token chunk about 25%). The fix is bigger jobs:
  chunked prompt reading and checking several guessed tokens at once (L6).
  Same lesson from session ai-projects-60's width ladder (2026-11-21, runs/width-ladder-20261121-152825): 3x3 convs at
  960x540 went from 97 GOPS at 8 channels to 603 GOPS at 64 (34x the math for 5.6x the time).
  For the engine (Jordan, 2026-11-21: "don't forget the NPU is wider now"): make every job as wide as quality allows.
  (1) Fewer weight-group jobs per step: 4 groups split each step into 4 narrower jobs (135M gateup 768 rows per job),
  and the simulator found 4 groups ~ 16, so try 1-2 groups (quality first, in the simulator). (2) Bigger chunks
  (more tokens per job). (3) The wider models: 1.7B / SmolLM3 steps are 3.6x wider than 135M (hidden 2048, gateup
  16384 rows), and the outlier fix made them usable on the NPU, so the engine's NPU gain is largest there.
  Groups test (llm/ref/l2h_groups.py, 2026-11-21): fewer jobs cost quality. KL vs fp32 GPL-3 / README, 4 / 2 / 1 jobs:
  135M 0.023 / 0.032, 0.033 / 0.037, 0.053 / 0.064 (ppl 1 job 16.31 / 48.95 vs fp32 15.71 / 47.38); 360M 0.013 /
  0.022, 0.020 / 0.053, 0.034 / 0.083 (1 job README 45.73 = +17%). Even 1 job beats the old format by far (135M
  19.48 / 57.55). 1.7B running (it needed per-job output scales most). Whether wider jobs pay for this depends on
  their speed on the NPU, not measured yet (packs with 1 / 2 jobs + the engine ladder).
- State: nothing running; NPU idle and healthy this boot (c0a2b9d9: no hang, no kernel warning). perf is installed
  (root step 230) and kernel profiling stays open until the next reboot.
- Done today: L3c closed (one-token CPU + NPU split does not pay: memory bus); Mesa read-out patches 10-12 (upstream
  series, all NPU-tested, test_teflon 1503 / 34 with every count of patches; Jordan rewrites the texts and sends);
  zero-copy hook (test tree); L2g: rotation + per-job output scales make every-step 8-bit nearly lossless on all
  four models (1.7B 2.63 vs fp32 2.18, was 27.2); R2: all 240 real-weight packs of SmolLM2-135M pass on the NPU
  (runs/llm-real-ladder-20261120-120736): 256-token chunk 124 ms for all 120 layer steps = 0.485 ms per token.
- 2026-11-21: engine process (llm/l3/jb_engine.c) PASS on the NPU (runs/llm-engine-ladder-20261121-153843): all 120
  steps compiled in one process in 2.5 s and run back to back, every output within 1 level; a 256-token chunk takes
  140 ms for all linear steps (0.547 ms per token; 13% slower than the per-step sum, colder caches), 1 token 32.7 ms.
  GEM buffers do not show in the process RSS on the device (peak 10.5 MB at 1 token); measure them system-wide later.
- Next: the CPU glue for a 256-token chunk (attention over the chunk, RMSNorm, RoPE, SiLU x mul, the block Hadamard
  on the down input, planar transposes, scattering the 4 jobs' outputs back with groups.npz), then one full prompt
  chunk end to end against the simulator's logits and against llama.cpp (pp128 104 tok/s). Also: compile all 120
  steps in one process (memory), and the 1.7B / SmolLM3 packs.
- Uncommitted in ~/alta-npu since caaa985: the L2g results, R2 (real_packs.py, real_offline.sh, real-ladder.sh,
  runs), these notes.

| Step | What | Where | Status |
|---|---|---|---|
| L0 | LLM-shaped int8 1x1 conv models (SmolLM2-135M, Qwen3-0.6B layer shapes, M = 1, 16) | `tools/llm_shapes.py` | DONE 2026-11-18 |
| L0b | Offline fake-NPU run of all 18 | `npu-llm/runs/offline-20261118*` | DONE: 1 TP + 1 NN job each, no driver error, weights uncompressed |
| L1 | NPU: correctness + speed of those 18 (effective weight GB/s) | `llm-l1-ladder.sh` | PASS 2026-11-18 (Jordan's OK): all max diff <= 1 vs CPU, no hang; see "L1 results" |
| L1b | NPU: layouts that make M > 1 cheap | `llm-l1b-ladder.sh`, `tools/llm_layouts.py` | PASS 2026-11-19: 4x4 token image = 16 tokens for 2-4x one token |
| L3a | Tiling hooks (mesa-llm, llm-tiling-experiments, lib 678775c5): ETNA_ML_V7_KPC_CAP / ETNA_ML_V7_TILE_H | `l3/` | offline PASS; NPU ladder 2026-11-19: baselines PASS, KPC 24 HUNG the NPU (incidents/2026-11-19-1106-LLM3-k24) -> the 1x1 cap of 21 kernels per core is a hardware limit; lever closed, see "L3 results" |
| L3b | Own runtime / job builder: NN jobs through the Gallium ML API directly (no TFLite), persistent context, weights freed after compile, several jobs per submit | `l3/JOB-BUILDER.md`, `l3/jb_run.c`, `l3/jb_pack.py` | NPU PASS 7/7 2026-11-19: same job 0.50 ms direct vs 0.82 Teflon; fixed cost per submit ~0.16 ms, 5.26 GB/s inside a submit; group jobs +14-35 us each; chains save one fixed cost; see l3/JOB-BUILDER.md |
| L3c | Fewer submits per token: CPU takes the small steps, CPU + NPU split the big ones (row split, one submit per big step) | `l3/plan_cost.py`, `ref/l2f_hybrid.py`, `l3/split-ladder.sh`, jb_run ca878c69 (`--wait block|spin`, `--cpu-rows`) | model + offline DONE 2026-11-19: NPU-only loses on every model; split = x1.10-1.18 (360M), x1.17-1.41 (1.7B), none for 135M decode; NPU gateup ~lossless on 135M, +5-12% ppl on 360M, BROKEN on 1.7B (2.24 -> 5.7-5.9: input outliers); NPU down costs quality everywhere; **CLOSED 2026-11-20**: both split ladders correct, but side by side the NPU and CPU together read only 7.1-7.4 GB/s (CPU alone 6.7): the memory bus is the limit, so a split of a one-token step does not pay (measured model x1.00, SmolLM3 x1.02-1.06); thread placement ruled out (split-ladder2). The NPU's use is M > 1 work |
| L1c | NPU: cost curve over 2D token images (1x1 ... 16x16); CPU + NPU reading memory at the same time | `llm-l1c-ladder.sh`, `tools/llm_sizes.py`, `tools/bw_loop.c` | PASS 2026-11-19: irregular costs (tuning headroom), prompt chunks ~9x/token, CPU+NPU share memory well |
| L6a | CPU: TAGE-style n-gram drafter acceptance on SmolLM2 greedy outputs | `ref/l6a_generate.py`, `ref/l6a_drafters.py` | DONE 2026-11-19: 1.24-1.41 tokens/step overall, 1.6-2.4 on rewrite/code; win needs cheap 4-8 position checks |
| L2 | CPU: quality of per-tensor int8 (W8A8) on a real small model, numpy, vs fp32; also measure FFN activation sparsity (how many neurons per token matter) | - | next, CPU only |
| L2b | CPU: train a perceptron predictor of active FFN neuron blocks; bytes saved vs quality lost | - | after L2, if sparsity is worth it |
| L2g | CPU: fixes for the 8-bit activation outliers, every step on the NPU (M > 1 work): per-job output scales, SmoothQuant, rotation (QuaRot-style, exact in fp32: residual Q folded into the weights, per-head rotation folded into v/o, block Hadamard on the down input online on the CPU) | `ref/l2g_outliers.py`, `l2g-chain.sh` | DONE 2026-11-20 on all four models: rotation + per-job output scales (jobs grouped by max |w|), every step 8-bit: 135M 16.06 / 47.67 (fp32 15.71 / 47.38), 360M 10.93 / 40.55 (10.93 / 39.12), 1.7B 2.63 / 25.43 (2.18 / 24.49; was 27.2 / 416), SmolLM3 4.22 / 31.11 (4.27 / 30.20); KL 10-38x lower; 4 groups ~ 16 |
| L3 | Own job builder: one layer without TFLite, offline first | - | after L1 |
| L4 | One full transformer block on NPU + CPU, vs CPU reference | - | |
| L5 | Whole model (SmolLM2-135M, then Qwen3-0.6B), tokens/s vs llama.cpp Q8_0 on the CPU | - | |
| L6 | Speculative decoding (M > 1 verify): TAGE-style n-gram drafter + perceptron confidence; first measured on the CPU (acceptance rate on real prompts) | - | |
| L7 | Predicted-block FFN on the NPU (per-token job list from the L2b predictor) | - | |
| G1 | DONE 2026-11-19, 2.6-3.0x, see G1 section. Mali-G52 int8 dot product: Bifrost ISA has `IDP.v4i8` (mesa bifrost/ISA.xml:1480, FMA unit, no accumulate) but Mesa emits dot products natively only for arch >= 9 (bifrost_compile.h:148, panvk "accelerated" = arch >= 9); a compiler patch (sdot_4x8_iadd -> IDP.v4i8 + IADD on v7) + a GPU test would show if the G52 runs it. For prompt reading / a third memory reader | - | idea (Jordan, 2026-11-19) |

## L1 results (2026-11-18, runs/llm-l1-ladder-20261118-183408, lib b6c2a28d, via Teflon)
- Correct: all 18 shapes (576-3072 in, 192-6144 out, M = 1 and 16) within 1 level of XNNPACK, mean ~0, every
  invoke identical. The deeplab 960-channel problem does not show up in these 1x1 shapes.
- M = 1: invoke time = 0.30 ms fixed + 4.2 GB/s of weights (fit over 9 shapes; qwen_gateup 6.3 MB in 1.65 ms).
  Below the CPU's raw read speed (7.5-8 GB/s). The fixed part (Python + TFLite invoke + a TP job + submit) is what
  our own engine removes by chaining all of a token's jobs in one submit.
  SmolLM2-135M per token, from these numbers: ~121 jobs, ~134 MB -> ~32 ms streaming + ~36 ms fixed via Teflon.
- M = 16 in this layout ([1,1,M,IC], tile 1x16): 13-20x the M = 1 time, only 4-8 GOPS (the NN unit did 790 GOPS
  on 1x1 convs over big images). So "K tokens for the price of 1" FAILS in this layout, and prompt reading would be
  slower than the CPU. L1b candidates: tokens as a 2D image ([1,4,4,IC]), tokens along H ([1,M,1,IC]), larger M
  (does the per-token cost fall?), and swapped roles: the weight matrix as the IMAGE (OC pixels x IC channels,
  streamed the way the NN streams big images) and the M tokens as 1x1 KERNELS (encoded per step by the engine;
  all tokens share one scale, so no per-channel need), output [OC, M] transposed back on the CPU.

## L1b results (2026-11-19, runs/llm-l1b-ladder-20261119-015157, Jordan's OK, 24/24 PASS, all max diff 1)
16 tokens (64 for row64) per job; cost relative to ONE token (L1 M = 1) and per token:
| layout | smol_gateup | qwen_down | qwen_gateup |
|---|---|---|---|
| row [1,1,16,IC] (L1 layout) | 11.3x, 0.61 ms/tok | 12.4x, 0.77 | 15.9x, 1.64 |
| rowP (planar, no TPs) | 11.6x | 11.7x | 16.1x |
| sq [1,4,4,IC] | 5.1x, 0.27 ms/tok | 2.9x, 0.18 | 4.1x, 0.42 |
| sqP (planar) | 2.9x, 0.15 ms/tok | 2.1x, 0.13 | 4.3x, 0.45 |
| col [1,16,1,IC] | 6.6x | 6.1x | 8.7x |
| row64 [1,1,64,IC] | 66x (0.89 ms/tok) | 85x | 103x |
| swapP (weights as the image) | 9.9x | 11.3x | 16.0x |
| swap (same, Teflon TPs) | 40x | 63x | 66x |
- The thin 1xN tile was the problem, not the transposes. As a 4x4 image, 16 tokens cost only 2-4x one token
  (23-49 GOPS vs 4-8): multi-token verify (speculative decoding) and chunked prompt reading are viable.
- Swapped roles do not help (only 16 kernels over 8 cores, kpc 2), and are much worse without the planar trick.
- Next NPU question (L1c): the cost curve over 2D token images (2x2, 2x4, 4x4, 4x8, 8x8, 16x16) to pick K and
  the prompt chunk size. Next CPU question (L6a): how many of K drafted tokens a TAGE-style n-gram drafter gets
  accepted by SmolLM2 on real prompts (decides whether K = 16 at 2-4x pays off).

## G1: Mali-G52 int8 dot product (2026-11-19, llm/gpu/)
- Patch (worktree ~/Documents/AI_Projects/mesa-gpu-dot, branch gpu-dot-v7, llm/gpu/0001-*.patch): arch 7 gets its
  own NIR options with has_sdot_4x8 / has_udot_4x8 (the _sat forms stay lowered); the backend emits IDP.v4i8 +
  IADD (IDP has no accumulator; the per-source sign is the abs modifier in the generated packer); panvk reports
  4x8 packed dot as accelerated on v7. Build: Vulkan panvk only + drm-shim, LLVM 20 for mesa_clc (native.ini).
- Test: llm/gpu/vk_dot.c via run_gpu.sh (NPU guard preloaded, refuses non-Mali devices, counts new kernel GPU
  errors). dot_check.spv = sdot(a,b)+7 vs CPU; dot_bench.spv = 8 dot chains per invocation (OpSDot);
  emul_bench.spv = the same with the 4 byte products written out (what NIR does without the patch).
- Baseline, system panvk (Mesa 25.2.8, unpatched): check 0/65536 wrong; dot_bench 13.05 G int8 MAC/s, emul 13.57
  (same: the dot is lowered); plain 32-bit multiply chain 9.7 G MAD/s, so a 4-wide IDP at that rate could reach
  ~39 G MAC/s. For scale: llama.cpp prompt reading on the CPU = ~13.5 G MAC/s.
- RESULT 2026-11-19 (patched panvk, mesa-gpu-dot f023bc2e, build 17 min): offline on drm-shim (fake G52) the
  compiler emits `IDP.v4i8 r1.sext, r2.sext` + `IADD.s32` per dot (4 in dot_check, 32 in dot_bench); on the
  real Mali-G52 MC2: dot_check 0/65536 wrong; dot_bench 37.1 G int8 MAC/s (256 rounds) / 38.9 (1024), all
  correct; emul_bench on the same driver 14.05; unpatched system panvk dot_bench 13.11 => 2.6-3.0x. No kernel
  GPU errors in any run (npu-llm/runs/g1-offline-20261119 has the dumps). ~78 G int8 ops/s in registers:
  above the CPU's 27 GOPS prompt reading and the NPU's 23-54 GOPS with Teflon configs; a real GEMM that
  loads its data will get less. Upstream-worthy (Jordan sends): same shape as the NPU series.

## CPU baseline (2026-11-19, llama.cpp b5dbca1, ARMv8.0 NEON no dotprod, 4 threads on the A73s, weights in RAM)
npu-llm/runs/cpu-baseline2-20261119.log (pass 1, cpu-baseline-20261119.log, was spoiled for 1.7B/3B prompt
reading by page-cache evictions: the timing included SD-card reads). Tokens/s:
| model | file | pp128 (prompt) | tg64 (generate) | memory speed used by tg |
|---|---|---|---|---|
| SmolLM2-135M Q8_0 | 136 MiB | 103.8 | 47.3 | 6.7 GB/s |
| SmolLM2-360M Q8_0 | 367 MiB | 37.9 | 18.4 | 6.7 GB/s |
| SmolLM2-1.7B Q8_0 | 1.69 GiB | 7.8 | 4.2 | 7.5 GB/s |
| SmolLM3-3B Q4_K_M | 1.78 GiB | 5.0 | 3.1 | 6.0 GB/s |
| SmolLM3-3B Q8_0 (3.04 GiB > free RAM, pages from SD) | | 0.20 | 0.02 (50 s/token) | |
- Generation already runs at 80-90% of the board's 8.5 GB/s. The NPU alone streams 4.2 GB/s (+0.3 ms per Teflon
  job), so the NPU ALONE cannot beat llama.cpp at generating; 135M: CPU 21 ms/token vs NPU ~64 ms via Teflon,
  ~28 ms estimated with our own submit path.
- Prompt reading on the CPU = ~27 GOPS for every size (103.8 x 0.27, 37.9 x 0.72, 7.76 x 3.4 GFLOP/token). The NPU
  did 23-54 GOPS at 16x16 with Teflon's configs (790 GOPS on 1x1 convs over big images): a win that grows with
  tuning (L3).
- So the realistic NPU roles: (1) prompt reading, (2) a second memory reader next to the CPU (L1c: together
  ~10.5-11.5 GB/s vs 8.5) by splitting each matrix multiply's rows, (3) freeing the CPU.
- Whole-model spilling is unusable (50 s/token); only rarely used weights may live outside RAM.

## L1c results (2026-11-19, runs/llm-l1c-ladder-20261119-022619, Jordan's OK, 40/40 + bandwidth PASS, max diff 1)
Planar jobs, tokens as an H x W image. Cost relative to one token:
| image (M) | smol_o | smol_qkv | smol_down | smol_gateup | qwen_gateup | SmolLM2 layer (4 jobs) | ms/token/layer |
|---|---|---|---|---|---|---|---|
| 1x1 (1) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 (1.91 ms) | 1.91 |
| 1x2 (2) | 1.09 | 1.34 | 1.32 | 1.79 | 2.08 | 1.45 | 1.38 |
| 2x2 (4) | 1.24 | 1.79 | 0.94 | 3.29 | 1.56 | 1.97 | 0.94 |
| 2x4 (8) | 1.56 | 2.66 | 1.49 | 3.50 | 2.70 | 2.43 | 0.58 |
| 4x4 (16) | 1.82 | 6.74 | 1.74 | 4.98 | 3.67 | 3.87 | 0.46 |
| 4x8 (32) | 3.21 | 12.39 | 3.13 | 7.56 | 8.75 | 6.52 | 0.39 |
| 8x8 (64) | 10.21 | 10.55 | 4.06 | 19.00 | 11.05 | 11.61 | 0.35 |
| 16x16 (256) | 19.91 | 26.29 | 15.85 | 47.23 | 38.95 | 29.49 | 0.22 |
- Costs are IRREGULAR between shapes (smol_qkv 16 tokens 6.7x vs smol_down 1.7x; smol_down 4 tokens cheaper than
  1): Teflon's tile / kernels-per-core choices are far from the best for these shapes. Tuning them is a job for our
  own job builder (L3), which is the next big lever for every M.
- Speculative decoding with these measured costs (L6a replay): lookup K=3 0.84x, warm TAGE K=3 0.77x, warm TAGE +
  perceptron K=3 1.01x overall (qa 0.93, list 1.04, rewrite 1.18, code 1.12), K=7 0.99x (rewrite 1.23). Free
  guessing only breaks even today; the perceptron gate is what keeps it from losing.
- Prompt reading: 256-token chunks cost 0.22 ms per token per layer vs 1.91 one at a time: ~9x faster per token.
- Shared memory (qwen_gateup 1x1 x2000, NPU work on cpu 0-1, bw_loop on cpu 2-5): CPU alone 8.55 GB/s; NPU alone
  2.27 ms/job; together NPU 2.40 ms (-6%), CPU's slowest 4 s stretch 7.87 GB/s (-8%; one run). Together they read
  roughly 10.5-11.5 GB/s, more than either alone: splitting each matrix multiply's rows between CPU and NPU at
  the same time is the way to use both for single-token decoding. **NOT reproduced 2026-11-20 by real split steps
  (L3c split ladders): together 7.1-7.4 GB/s, the split does not pay.**

## L6a results (2026-11-19, CPU only, npu-llm/runs/l6a-greedy.json + l6a-drafters-20261119.log)
48 greedy fp32 answers of SmolLM2-135M-Instruct (28 train / 20 test prompts; qa, list, rewrite, code), replayed
exactly with each drafter. Tokens per step (= the most a check may cost, relative to one normal step, and win):
| drafter | ALL | qa | list | rewrite | code |
|---|---|---|---|---|---|
| prompt lookup K=3 | 1.24 | 1.13 | 1.15 | 1.91 | 1.63 |
| prompt lookup K=15 | 1.29 | 1.14 | 1.17 | 2.44 | 1.84 |
| TAGE online K=15 | 1.29 | 1.14 | 1.17 | 2.44 | 1.84 |
| TAGE warm K=15 | 1.41 | 1.21 | 1.32 | 2.32 | 2.28 |
| TAGE warm + perceptron K=15 | 1.26 | 1.09 | 1.20 | 2.25 | 1.78 |
- Speedup with the L1b cost (16 positions = r x one step, r 2-4): best overall lookup K=3, 1.13x at r = 2 (linear
  cost model), ~1.0 at r = 3; rewrite/code 1.3-1.7x. With only 16-position jobs (fixed cost r = 3) every drafter
  LOSES overall (0.55-0.88x); the perceptron is the best there (0.88x, rewrite 1.14x): it cut drafting from 76% of
  steps to 22% and raised accepted tokens per try from 0.55 to 1.22.
- Warm TAGE tables guess more often (76% of steps) but each guess is worse; online TAGE = prompt lookup.
- So: free n-gram/TAGE guessing helps copy-heavy work (rewrite, code edits, summaries quoting the text), barely open
  chat (0.3 tokens accepted per try). The win depends on SMALL checks being cheap (K = 3 -> 4 positions, 2x2): the
  break-even for lookup K=3 is a 4-position check under 1.24x one step overall, 1.6-1.9x for rewrite/code.
- A drafter that runs the model itself (fewer layers, sparse FFN) is not free: with draft cost d per token and
  agreement p, speedup = (1 + p + ... + p^K) / (K d + c(K+1)); for SmolLM2 the head and attention alone are ~40% of
  the bytes, so d >= 0.4 and it does not pay on a bandwidth-bound board.

## L2e: weight channel groups (2026-11-19, llm/l2e-chain.sh, npu-llm/runs/l2e-groups-*.{log,json}, CPU only)
Output channels sorted by max |w|, split into G groups = G NPU jobs with their own weight scale; head fp32 in every row
(so the rows measure the layers only). ppl GPL-3 / README, KL vs fp32 on GPL-3.
| model | fp32 | w8 1 group | 4 groups | 16 groups | per-channel | w8a8 pos0 CPU, 16 groups | per-channel |
|---|---|---|---|---|---|---|---|
| 135M | 15.71 / 47.38 | 16.19 / 52.88 (KL 0.060) | 15.99 / 49.15 (0.019) | 15.63 / 47.89 (0.007) | 15.73 / 47.71 (0.002) | 19.48 / 57.55 (0.236) | 20.72 / 60.07 |
| 360M | 10.93 / 39.12 | 11.52 / 42.48 (0.062) | 11.06 / 40.97 (0.029) | 11.07 / 39.76 (0.019) | 10.90 / 38.94 (0.003) | 12.50 / 47.87 (0.142) | 12.02 / 51.91 |
- 16 groups make 8-bit weights practically lossless on 135M and 360M (KL /9 and /3 vs one scale); per-channel only
  a little better. On the NPU that is 16 jobs per layer step in ONE submit (built offline, L3b: kernels per core 12).
- With 8-bit activations (position 0 on the CPU) the activations are now nearly all of the loss (135M KL 0.236 vs
  0.007 for the weights): next quality lever = down_proj / outliers (L2b: down_proj on the CPU gave 17.47 on 135M).
- 360M with 16 groups + 8-bit activations: 12.50 / 47.87 (was 13.58 / 58.14 with one scale and an 8-bit head).
| SmolLM3-3B | 4.27 / 30.20 | 6.32 / 31.15 (0.522) | 4.38 / 29.25 (0.116) | 4.40 / 31.03 (0.030) | 4.32 / 30.81 (0.007) | 5.32 / 35.18 (0.406) | 5.20 / 31.03 |
| SmolLM2-1.7B | 2.18 / 24.49 | 3.56 / 28.87 (0.475) | 2.62 / 26.74 (0.170) | 2.27 / 24.95 (0.043) | 2.24 / 24.28 (0.010) | 27.20 / 416.23 (2.401) | 28.51 / 252.80 |
- Groups fix the WEIGHTS on every model, also the big ones: 16 groups bring 1.7B from 3.56 to 2.27 (fp32 2.18) and
  SmolLM3 from 6.32 to 4.40 (fp32 4.27). The 1.7B head caveat is settled: with the head in fp32, one scale still gives
  3.56 (3.57 with the 8-bit head), so its loss was the layers' weights.
- SmolLM3-3B with 16 groups + 8-bit activations (pos 0 on CPU): 5.32 / 35.18 = usable; per-channel 5.20 / 31.03.
- SmolLM2-1.7B's 8-bit ACTIVATIONS stay broken (27 / 416 with 16 groups): it needs activation outlier handling
  (ablation of which layers break, then SmoothQuant-style scaling, outlier channels on the CPU, or int16) before
  the NPU format can run it. SmolLM3-3B does not.
- Cost of groups on the NPU (jb ladder): +35 us per extra job at 4 groups, +14 us at 16; 4 groups already give most
  of the gain on SmolLM3 (4.38) and 360M, less on 1.7B (2.62 vs 2.27).

## L2g: activation outlier fixes (2026-11-20, llm/ref/l2g_outliers.py, CPU only)
Target: every layer step on the NPU in its 8-bit format (prompt chunks, drafted-token checks), position 0 on the CPU,
fused qkv / gateup jobs, 16 weight-group jobs per step, head fp32. All fixes are free at run time.
- Rotation (QuaRot-style): an orthogonal Q on the residual stream folded into every weight that reads or writes it
  (RMSNorm weights folded into the readers, embedding rows rotated, the head reads the rotated-back residual); a
  per-head Hadamard folded into v and o; a block Hadamard on the down_proj input, applied online by the CPU (it
  computes SiLU x up anyway; a fast Walsh-Hadamard transform on 1536-11008 values is ~20 us). Q = randomized
  Hadamard for hidden size 2048 (1.7B, SmolLM3), dense random orthogonal for 576 / 960 (offline, cost irrelevant).
- Per-job output scales: an NPU job writes its own output tensor, so each weight-group job can have its own output
  scale and zero point (calibrated per channel, position 0 left out).
- SmoothQuant 0.5 folded into the RMSNorm weights (qkv, gateup) and the up_proj rows (down), for comparison.
- Smoke test, SmolLM2-135M, 32 tokens per text (npu-llm/runs/l2g-outliers-SmolLM2-135M-Instruct-smoke-T32*):
  | variant | KL vs fp32 GPL-3 / README | top-1 = fp32 |
  |---|---|---|
  | A baseline (one output scale per step) | 0.355 / 0.483 | 62 / 56% |
  | S SmoothQuant + per-job out (grouped by output range) | 0.707 / 0.792 | 44 / 50% |
  | rotated model in fp32 | 0.000 / 0.000 | 100 / 100% |
  | R rotation | 0.038 / 0.087 | 91 / 81% |
  | R + per-job out scales, jobs grouped by max abs w | 0.024 / 0.049 | 91 / 91% |
  | R + per-job out scales, jobs grouped by output range (x1.0 / x1.5 / x2.0 headroom) | 0.40 / 0.07 / 0.04 (GPL-3) | 62-91% |
  Input outlier ratio (largest / median channel max, median over layers): qkv 14.9 -> 1.9, gateup 6.4 -> 2.0,
  down 12.9 -> 2.1 (worst layer 152 -> 12) with rotation; SmoothQuant 4.8 / 2.9 / 4.5.
  Grouping jobs by output range mixes small and large weights in one job and costs weight precision: keep the
  max |w| grouping (rows B / S / RB in the full runs use it).
- Full runs (256 tokens, all four models; llm/l2g-chain.sh, unit alta-llm-l2g, started 02:31 after the l2f run): rows
  A, B, S, rotated fp32 check, R, RB, RB4 (4 groups). Perplexity GPL-3 / README, (KL vs fp32 GPL-3 / README):
  | model | fp32 | A baseline | B per-job out | S SmoothQuant + B | R rotation | RB rotation + per-job out | RB4 (4 groups) |
  |---|---|---|---|---|---|---|---|
  | 135M | 15.71 / 47.38 | 19.48 / 57.55 (0.236 / 0.304) | 17.28 / 53.73 (0.119 / 0.165) | 19.26 / 52.73 (0.237 / 0.199) | 16.08 / 51.16 (0.032 / 0.057) | **16.06 / 47.67 (0.016 / 0.023)** | 15.81 / 49.24 (0.023 / 0.032) |
  | 360M | 10.93 / 39.12 | 11.96 / 49.48 (0.134 / 0.275) | 12.18 / 46.16 (0.112 / 0.206) | 11.33 / 44.91 (0.066 / 0.174) | 10.84 / 40.82 (0.017 / 0.038) | **10.93 / 40.55 (0.010 / 0.021)** | 11.14 / 40.12 (0.013 / 0.022) |
  | 1.7B | 2.18 / 24.49 | 27.20 / 416.23 (2.401 / 3.211) | 13.37 / 163.24 (1.768 / 2.200) | 6.30 / 39.56 (1.100 / 0.583) | 4.58 / 30.40 (0.722 / 0.313) | **2.63 / 25.43 (0.163 / 0.084)** | 2.52 / 25.38 (0.182 / 0.089) |
  | SmolLM3-3B | 4.27 / 30.20 | 5.29 / 35.10 (0.443 / 0.179) | 5.61 / 34.13 (0.405 / 0.162) | 4.32 / 36.39 (0.231 / 0.214) | 4.58 / 31.18 (0.103 / 0.056) | **4.22 / 31.11 (0.029 / 0.018)** | 4.19 / 30.49 (0.050 / 0.022) |
  135M: the rotated model in fp32 = fp32 exactly; RB makes EVERY step 8-bit on the NPU about as good as int8 weights
  with fp activations on the CPU (H0: KL 0.002 / 0.003, ppl 15.73 / 47.71), top-1 = fp32 91 / 93%.
  **1.7B (the model the 8-bit format broke), 173 min: FIXED.** RB gives 2.63 / 25.43 (fp32 2.18 / 24.49; the
  current format 27.20 / 416.23), KL 15-38x lower, top-1 = fp32 89 / 85%; with 4 groups per step 2.52 / 25.38.
  Rotation alone (R) 4.58 / 30.40 and per-job output scales alone (B) 13.37 / 163 are each far from it: rotation
  fixes the inputs, per-job scales the outputs, and 1.7B needs both. Input outlier ratio, largest layer: down
  2430 -> 3.1, qkv 178 -> 1.8. SmoothQuant 6.30 / 39.56. Still above int8 weights + fp activations on the CPU (H0:
  KL 0.010 / 0.008), but usable: the NPU can now run 1.7B prompt chunks and drafted-token checks.
  360M (14 min) and SmolLM3-3B (253 min), finished 05:45 / 09:58: RB 10.93 / 40.55 (fp32 10.93 / 39.12) and
  4.22 / 31.11 (fp32 4.27 / 30.20); rotated fp32 = fp32 exactly on both. **On all four models rotation + per-job
  output scales cuts KL vs fp32 10-38x** and brings every-step-8-bit perplexity to within 0-4% of fp32 (1.7B GPL-3
  +21%). 4 groups per step are about as good as 16 everywhere (fewer jobs per step).

## L3c: fewer submits per token (2026-11-19, llm/l3/plan_cost.py, CPU only so far)
- Paused by Jordan 2026-11-19 18:51, resumed 2026-11-20 (boot c0a2b9d9). split-ladder.sh has NOT run (needs OK, quiet
  CPU: pause other heavy jobs with SIGSTOP/SIGCONT); afterwards fill plan_cost.py with the measured numbers. SmolLM3
  l2f was stopped by Jordan 2026-11-19 after its fp32 rows; rerun started 2026-11-20 (unit alta-llm-l2f).
- The driver has no elementwise MUL (supported: conv, add/sub, concat, split, pad, FC, transpose, reshape, relu, abs,
  logistic), so SiLU(gate) x up cannot join gateup and down in one submit with stock ops; the shader core could, but
  a shader launch costs about what the merge saves. So: cut submits instead of merging them.
- Cost model (measured NPU numbers from the jb ladder, llama.cpp CPU rates; split overhead 0.02 ms and an 8% CPU
  slowdown next to the NPU ASSUMED, measured by split-ladder.sh):
  | model | CPU only | NPU all 4 steps | split gateup | split gateup + down | split every step + head |
  |---|---|---|---|---|---|
  | 135M | 47.6 tok/s | 22.6 | 47.6 (no gain) | 47.6 | 51.9 (x1.09) |
  | 360M | 18.5 | 11.4 | 20.5 (x1.10) | 20.6 (x1.11) | 21.8 (x1.18) |
  | 1.7B | 4.4 | 3.0 | 5.2 (x1.17) | 5.6 (x1.27) | 6.3 (x1.41) |
  | SmolLM3-3B | 2.0 | 1.7 | 2.5 (x1.29) | 2.9 (x1.49) | 3.3 (x1.67) |
  The NPU on its own, one submit per step, loses on every model; one submit per BIG step, next to the CPU, wins
  from 360M up. For 135M decode the steps are too small for the 0.16 ms fixed cost (the NPU helps it only with
  prompts and multi-token checks).
- Quality of the split (ref/l2f_hybrid.py: CPU steps int8 per-channel with fp32 activations, NPU steps in the 8-bit
  format, position 0 on the CPU), 135M: NPU gateup only 16.05 / 48.71 with 4 groups (fp32 15.71 / 47.38, all-CPU
  15.73 / 47.71) = nearly lossless; NPU gateup + down 18.4-18.9 (down's 8-bit activations). All models (ppl GPL-3 /
  README, 256 tokens each, head fp32; top-1 agreement with fp32 on GPL-3):
  | model | fp32 | H0 all CPU | H2 NPU gateup, 4 groups | H2 NPU gateup, 16 groups | H1 NPU gateup + down, 4 / 16 groups |
  |---|---|---|---|---|---|
  | 135M | 15.71 / 47.38 | 15.73 / 47.71 | 16.05 / 48.71 (top-1 0.92) | 15.61 / 49.33 (0.93) | 18.90 / 55.70, 18.43 / 61.64 (0.71-0.74) |
  | 360M | 10.93 / 39.12 | 10.90 / 38.94 | 11.43 / 43.49 (0.89) | 11.64 / 45.35 (0.89) | 11.98 / 48.00, 12.21 / 46.64 (0.82) |
  | 1.7B | 2.18 / 24.49 | 2.24 / 24.28 | 5.88 / 35.64 (0.66) | 5.66 / 37.50 (0.65) | 23.00 / 99.94, 19.10 / 101.71 (0.41) |
  | SmolLM3-3B | 4.27 / 30.20 | 4.32 / 30.81 | 4.36 / 31.39 (0.89) | 4.38 / 32.06 (0.92) | 5.51 / 33.36, 4.30 / 33.05 (0.76-0.79) |
  So the loss is the NPU's per-tensor 8-bit ACTIVATIONS, not its weights (H0 = int8 weights is ~fp32 everywhere, and
  more weight groups do not help H2): gateup's input (the RMSNorm output) is fine on 135M, costs 5-12% on 360M and
  breaks 1.7B (its known activation outliers, L2). Before a split pays on 1.7B the NPU's input needs outlier
  handling; candidates, all CPU-side and free at run time: SmoothQuant scales folded into the RMSNorm weight (gateup,
  qkv) and into up_proj's rows (down); or the few outlier channels computed on the CPU (LLM.int8 style).
- jb_run ca878c69: `--wait block|spin` (own flush + fence wait; same submits and streams as before, offline),
  `--cpu-rows R --cpu-ic IC` (4 pinned A73 threads, NEON int8 GEMV on >= 64 MB of rotating weight copies so no step
  hits the L2, verified vs a scalar reference) between flush and wait, `--cpu-only`. Split packs (synthetic):
  360M gateup NPU share 1536 x 960 in 4 jobs (kpc 16), 1.7B gateup NPU share 6656 x 2048 in 4 jobs (kpc 21), both one
  submit, no driver complaint offline. CPU timings taken while another CPU job ran were 10-25x off: the split step
  waits for its slowest thread, so CPU measurements need a quiet CPU.
- **Split ladder (2026-11-20 00:55, runs/llm-split-ladder-20261120-005522, Jordan's OK, boot c0a2b9d9, 10/10 correct;
  SmolLM3 l2f paused with SIGSTOP; desktop + system monitor open).** Medians of 199 invokes:
  | rung | what | ms per step | NPU part | CPU part |
  |---|---|---|---|---|
  | S1 / S2 | J1 job, blocking wait / polling wait | 0.543 / 0.484 | 0.407 / 0.381 | - |
  | C1 / C2 / C3 | CPU only, 3072x576 / 3584x960 / 9728x2048 | 0.237 / 0.522 / 2.930 | - | 7.47 / 6.59 / 6.80 GB/s |
  | S3b / S3c | NPU share only, 360M (1.47 MB) / 1.7B (13.6 MB), 4 jobs | 0.405 / 2.787 | 0.333 / 2.555 | - |
  | S4 | J1 job + CPU 3072x576 side by side | 0.960 | 0.558 (alone 0.381) | 0.500 (alone 0.237) |
  | S5 | 360M gateup split (NPU 1536 rows, CPU 3584) | 1.057 | 0.603 (alone 0.333) | 0.680 (alone 0.522) |
  | S6 | 1.7B gateup split (NPU 6656 rows, CPU 9728) | 5.987 | 4.608 (alone 2.555) | 4.917 (alone 2.930) |
  - Polling instead of a blocking wait saves 0.06 ms per submit. With polling the NPU streams 5.10 GB/s and the
    fixed cost is ~0.12 ms even with 4 jobs (S3b/S3c), better than the model's 0.163 + 3 x 0.035.
  - The CPU GEMV runs 6.6-7.5 GB/s alone: the llama.cpp rates in plan_cost.py were right.
  - **Side by side both slowed far more than the model assumed** (NPU x1.8, CPU x1.3-2.1 instead of -6% / -8%), and
    the split step is SLOWER than the CPU doing the whole step: 360M 1.057 vs ~0.75 ms, 1.7B 5.99 vs ~4.93 ms
    (together only 4.7-5.6 GB/s). About 0.3-0.8 ms also appears between the end of the invoke call and the CPU
    threads starting (median read time minus CPU time minus the read-out), i.e. the submit got slow.
  - Suspected then: the unpinned submit + polling thread sharing an A73 with a worker (L1c, -6% / -8%, had pinned
    the NPU side to the A53s). split-ladder2 ruled that out (below).
  - Prepared (CPU only): jb_run 3fa43e81 (`--main-cpu C`, reports `flush_ms` and `main_on_a73`; frozen copies
    bin/jb_run-ca878c69 for split-ladder.sh and bin/jb_run-3fa43e81 for the new ladder) and llm/l3/split-ladder2.sh
    (8 rungs: NPU share and CPU share alone with the main thread on cpu 0; S4 and the 1.7B split unpinned again with
    the new timers; S4, 360M and 1.7B splits pinned to cpu 0, and the 1.7B split with a blocking wait). Offline
    (npu-llm/runs/offline-jb-20261120-maincpu): one submit per invoke, submit streams and buffer lists byte-identical
    to the jobs that just ran; pinned main thread never seen on an A73, unpinned 10% of samples (under strace + l2f).
- **Split ladder 2 (2026-11-20 01:12, runs/llm-split-ladder2-20261120-011234, Jordan's OK, 8/8 correct, SmolLM3 l2f
  paused).** Thread placement is NOT the cause:
  | rung | what | ms per step | submit | NPU part | CPU part |
  |---|---|---|---|---|---|
  | Q1 / Q2 | 1.7B NPU share alone / CPU share alone, main thread on cpu 0 | 3.822 / 2.983 | 0.065 | 2.778 | 6.68 GB/s |
  | Q3 / Q5 | J1 job + CPU 3072x576, unpinned / main on cpu 0 | 1.023 / 1.029 | 0.061 / 0.059 | 0.595 / 0.602 | 0.533 / 0.541 |
  | Q6 | 360M gateup split, main on cpu 0 | 0.994 | 0.061 | 0.609 | 0.685 |
  | Q4 / Q7 / Q8 | 1.7B gateup split, unpinned / main on cpu 0 / cpu 0 + blocking wait | 5.946 / 6.158 / 5.939 | 0.087-0.089 | 4.39-4.70 | 4.74-4.96 |
  - The submit is 0.06-0.09 ms in every rung; unpinned, the main thread was on an A73 in only 3-5% of samples (the
    busy A73s push it to the A53s anyway). Pinning or a blocking wait changes nothing.
  - **The memory bus is the limit.** Side by side the NPU streams 3.0-3.2 GB/s (alone 5.46) and the CPU 4.0-4.2
    (alone 6.68): together 7.1-7.4 GB/s, only ~6-10% more than the CPU alone. L1c's "-6% / -8%, ~11 GB/s together"
    (plain read loop on the A73s, Teflon job with real weights, measured over seconds) is NOT reproduced by real
    split steps; that number is what made the split look good.
  - **The read-out of the NPU's int8 outputs is slow**: Mesa allocates the tensors write-combined
    (etnaviv_ml.c:107) and etna_ml_subgraph_read_outputs subtracts 128 byte by byte straight from that uncached
    mapping: ~35 ns per byte on an A73 (0.23 ms for 6.6 KB), ~150 ns on an A53 (1.04 ms). It is ~1.1 ms of the 6 ms
    1.7B split step (after the join, on the A53). Fix: copy the buffer out in bulk first (or NEON), then convert;
    small Mesa change, part of R3 and an upstream candidate (every signed int8 Teflon output pays it).
  - **Fixed in mesa-llm 2026-11-20** (commits 1c441f99 read-out, 2cd9c87d KPC clamp; diff also in llm/l3/mesa-llm-readout-kpcclamp.diff; libteflon
    678775c5 -> a26f0852, old copy kept as npu-llm/bin/libteflon-678775c5.so; jb_run baca6a17 adds `readout_ms`):
    each output is copied out with one pipe_buffer_read (as unsigned outputs already were), then the -128 runs on
    cached memory. Same rebuild: the L3 hook clamps ETNA_ML_V7_KPC_CAP to nn_accum_buffer_depth / 3 = 21 with a
    warning (offline: 24 now gives the default config, not the incident's). Offline: every job unchanged and the 64-
    and 256-token gateup jobs (jb_run and Teflon) byte-identical to the L1c jobs that passed on the NPU. Matters most
    for M > 1: a 256-token gateup output is 786 KB (~27 ms byte by byte on an A73). NPU timing: llm/l3/readout-ladder.sh
    (12 rungs, old vs new binary and library; needs OK).
  - **Read-out ladder 2026-11-20 02:15 (runs/llm-readout-ladder-20261120-021528, Jordan's OK, 12/12 PASS, outputs
    unchanged):**
    | job | step old -> new | read-out old -> new |
    |---|---|---|
    | 1 token, 3 KB out, main on an A53 | 0.743 -> 0.443 ms | 0.360 -> 0.032 ms |
    | 1.7B split share, 6.6 KB, A53 | 3.455 -> 2.709 ms | 0.817 -> 0.069 ms |
    | 64 tokens, 196 KB | 6.532 -> 1.503 ms | 5.57 -> 0.58 ms |
    | 256 tokens, 786 KB | 25.58 -> 5.56 ms | 22.5 -> 2.49 ms |
    | Teflon, 1 token / 256 tokens (whole invoke) | 0.85 -> 0.63 / 29.48 -> 7.21 ms | |
    The NPU part is unchanged. Even the bulk copy reads the uncached buffer at only ~0.32 GB/s (A73) / ~0.1 (A53), so
    at 256 tokens the read-out (2.5 ms) is still about the NPU job (2.8 ms). Next step for M > 1: cached output
    buffers (DRM_ETNA_GEM_CACHE_CACHED; the kernel's cpu_prep / cpu_fini do the cache maintenance) or reading the
    output in place (R3), so big outputs read at DRAM speed. Upstream candidate (Teflon -26% on a one-token job, x4.1
    on 256 tokens); not sent, Jordan sends.
  - **Cached output buffers (2026-11-20, Jordan: "go ahead with the cached output buffers"; mesa-llm working tree,
    llm/l3/mesa-llm-cached-outputs.diff, lib 6caca26e, jb_run ad393a6f):** graph outputs (is_external_output, and the
    buffer a bypass / split output aliases) get ETNA_RESOURCE_FLAG_CPU_READ, which etna_buffer_bo_new now turns into
    a cached BO on arm64 (the texture path already does this); the kernel's CPU_PREP invalidates before each read and
    CPU_FINI cleans; the zero-fill at creation is wrapped in prep / fini so no dirty line can overwrite NPU results.
    Stale-read test: jb_pack.py altinput + jb_run --alt (two inputs alternate, every invoke checked). Offline: jobs
    unchanged, exactly the graph outputs cached, prep / fini paired. NPU: llm/l3/cached-ladder.sh (14 rungs, WC vs
    cached, 1000-invoke stale-read runs, Teflon), needs OK.
    **NPU 2026-11-20 03:16 (runs/llm-cached-ladder-20261120-031603, Jordan's OK, 14/14 PASS):** 2800 alternating
    invokes checked, no stale output, no kernel warning. Read-out WC -> cached: 256 tokens 2.61 -> 1.48 ms, 64 tokens
    0.58 -> 0.34, one token 0.032 -> 0.022 (A53); Teflon 256-token invoke 7.19 -> 6.29 ms, one token unchanged
    (0.66 / 0.67). Still only ~0.5 GB/s: probably the kernel's invalidate (CPU_PREP) and clean (CPU_FINI) passes
    over every cache line of the buffer, plus the copy and the -128 pass (not profiled). Next if it matters: perf
    to split it, then reading the cached output in place (R3) so one pass converts it.
  - **Profile (2026-11-20, runs/llm-readout-profile-20261120-034535, perf 6.8 cpu-clock, root step 230; no CPU PMU
    on this board):** of the 1.56 ms cached read-out (256 tokens) the scalar -128 loop is ~0.9 ms (GCC -O2 did not
    vectorize `out[k] -= 128`: ldrb / sub / strb), memcpy ~0.25 ms, the kernel's invalidate (CPU_PREP) and clean
    (CPU_FINI) ~0.2 ms each. CPU benchmark (llm/l3/bench/ro_bench.c, A73, 786 KB): Mesa's memcpy + loop 1.39 ms,
    memcpy + 64-bit xor (-128 == ^0x80 per byte) 0.67, one fused pass (64-bit load, xor, store) 0.36 (A53 2.64 ->
    0.76). A fused conversion would take the cached read-out to ~0.75 ms; the kernel's ~0.4 ms stays. For
    write-combined buffers keep the bulk copy (wide loads) and xor 64 bits at a time after it.
  - **Fused conversion (2026-11-20, Jordan: "go ahead with the fused conversion"; mesa-llm working tree,
    llm/l3/mesa-llm-fused-conversion.diff, lib fa818da9, jb_run a070d63e):** flip_sign_bits() flips the top bit of
    each byte 8 bytes at a time (64-bit ldr / eor / str); cached outputs convert straight from the buffer, write-
    combined ones after the bulk copy, signed inputs on the way in; etna_buffer_cpu_read_cached() is shared by the
    allocator and the read-out. Offline identical (incl. the input buffer bytes), unit test PASS. NPU:
    llm/l3/fused-ladder.sh (14 rungs, every invoke checked), needs OK.
    **NPU 2026-11-20 03:57 (runs/llm-fused-ladder-20261120-035655, Jordan's OK, 14/14 PASS, 3600 checked invokes, 0
    stale, 0 kernel warnings):** 256-token read-out 1.55 -> 0.54 ms, input write 0.26 -> 0.10, step 4.65 -> 3.48 ms;
    64 tokens 0.39 -> 0.13 ms read-out; Teflon 256-token invoke 6.28 -> 4.04 ms. Over the three changes the 256-token
    read-out went 22.5 -> 0.54 ms and the Teflon 256-token invoke 29.5 -> 4.0 ms. Left per 256-token step: NPU 2.8 ms,
    read-out 0.54 (of which ~0.4 is the kernel's cache maintenance), input 0.10.
  - **Zero-copy read-out (2026-11-20, Jordan: "go ahead with the zero-copy read-out"; R3, test tree only):** Teflon
    cannot use it (TFLite owns the output memory), so it is a mesa-llm hook our runtime calls directly, not a
    Gallium interface: etna_ml_subgraph_map_output() hands out the output as the NPU wrote it (signed: byte ^ 0x80),
    unmap does the CPU_FINI; skip_fini = an experiment without the clean pass. jb_run a79b7b5c --readout
    copy|zero|zero-nofini, --consume (one NEON pass over every int8 value inside the timed read-out, as the
    engine's next step would). Offline PASS; NPU llm/l3/zero-ladder.sh (14 rungs, every invoke checked) needs OK.
    **NPU 2026-11-20 04:27 (runs/llm-zero-ladder-20261120-042723, Jordan's OK, 14/14 PASS, 3600 checked invokes, 0
    stale incl. 1000 no-fini invokes, consumer sums equal in all modes):** read-out + consumer at 256 tokens copy
    0.60 ms, zero-copy 0.45-0.51, no-fini 0.30-0.32; one token (A53) 0.020 / 0.012 / 0.008. Zero-copy saves less
    than the copy pass because the 786 KB still has to come from memory once (now in the consumer); only the copy's
    write and its re-read go (~0.09 ms, ~3% of the 256-token step). Leaving out CPU_FINI saves ~0.19 ms more: the
    kernel's clean pass after a read-only access (arm64 cleans for every direction). That is a kernel-interface
    question, not something Mesa or our runtime should skip. Verdict: the engine may use the zero-copy hook (small
    gain, test tree); the remaining read-out cost is the kernel's cache maintenance.
  - plan_cost.py now uses the measured numbers (CPU_SLOW 0.60, NPU_SLOW 0.57, BW_CAP 7.4, SPLIT_SYNC 0.25): every
    split plan is x1.00 except SmolLM3 x1.02-1.03 (x1.05-1.06 even with the read-out fixed). **Verdict: for
    one-token decode the CPU alone is the plan; the NPU cannot add memory bandwidth.** What is left for the NPU is
    M > 1 work, where one weight read serves many tokens: reading the prompt in chunks and checking several drafted
    tokens at once (both still need the 8-bit activation fix on 1.7B), or running beside a CPU that is busy with
    something other than streaming weights.

## L3 results (2026-11-19, runs/llm-l3-ladder-20261119-110448, Jordan's OK, FAIL at rung 5 = NPU hang)
| rung | model | KPC_CAP / TILE_H | max diff | ms (median) | L1c |
|---|---|---|---|---|---|
| 1 | smol_gateup_1x1_P | default | 0 | 0.86 | 0.66 |
| 2 | smol_gateup_4x4_P | default | 1 | 3.36 | 3.29 |
| 3 | smol_gateup_16x16_P | default | 1 | 28.82 | 31.17 |
| 4 | qwen_gateup_1x1_P | default | 0 | 1.65 | 1.62 |
| 5 | smol_gateup_1x1_P | 24 / - | - | HUNG | |
- Rung 5's NN config differs from rung 1 only in kernels_per_core 21 -> 24 (offline dumps). The NPU hung (recover
  hung GPU x2, devcoredump: FE in its WAIT/LINK loop, core busy), kernel recovered, board up, STOP latch set.
- So the driver's cap for 1x1 kernels (accum_buffer_depth / 3 = 21 kernels per core) is a HARDWARE limit; the
  bound the hook kept (depth * interleave / tile_y = 64) is not the binding one. Probably the accumulation buffer
  holds 3 sets (pipelining) per kernel. Rules from now on: never KPC_CAP > 21 (ladder guard added); the hook in
  mesa-llm should clamp to <= 21 at the next rebuild.
- Rung 1 ran 0.86 ms vs 0.66 in L1c with an identical config: one run each, so treat as run-to-run spread until
  repeated (rungs 2-4 match within 2-8%).
- Consequence for the engine: the per-job tiling is already at the hardware limit for one token; the 1-token cost
  (0.30 ms fixed + 4.2 GB/s) can only drop by fewer, bigger jobs per layer (fused qkv / gateup already), by more
  jobs per submit (own job builder: one submit per layer or per block instead of one Teflon invoke per op), and by
  M > 1 (prompt chunks, speculative verify). TILE_H alone (lower tiles at 256 tokens) was never run; it only helps
  if it allows more kpc, which is capped, so it is dropped too.

## L2 on SmolLM2-1.7B-Instruct (2026-11-19, npu-llm/runs/l2-big-SmolLM2-1.7B-Instruct*.{log,json}, 48 min CPU)
| variant (fused jobs) | ppl GPL-3 | ppl README | top-1 = fp32 |
|---|---|---|---|
| fp32 | 2.18 (memorized text) | 24.49 | 100% |
| w8 (per-tensor int8 weights) | 3.57 | 30.79 | 79 / 74% |
| w8a8 static | 861.27 | 533.84 | 6 / 20% |
| w8a8, position 0 on CPU | 128.69 | 252.60 | 18 / 30% |
- The 1.7B model breaks the NPU format: per-tensor WEIGHTS already cost +26-64% ppl (135M +3%, 360M +10%), and
  8-bit activations stay catastrophic even with position 0 on the CPU, so it has big outliers on later tokens too.
  Needed before the NPU can run it: weight channel groups (several jobs with their own scale), an ablation of which
  layers' activations break (l2b-style, ~35 min on 1.7B), then outlier handling (SmoothQuant scaling, outlier
  channels on the CPU, or int16 activations: the NN unit has int16 cores, the driver does not use them yet).
- FFN sparsity holds: 29.4% of neurons carry 95% of the down_proj energy, 84.4% of it was active in t-1..t-4,
  oracle keep-50% ppl 2.17 (fp32 2.18).
- Caveat: the 1.7B head (tied embedding, 49152 x 2048) was 8-bit per-tensor in w8/w8a8; SmolLM3's head stayed fp32.
  Part of the 1.7B w8 loss may be the head; check with the head in fp32 before blaming the layers.

## L2 on SmolLM3-3B (2026-11-19, npu-llm/runs/l2-big-SmolLM3-3B-20261119-075707.json, 82 min CPU)
Head (128256 x 2048) kept fp32 in every variant (big vocabulary), so this measures the 36 layers only.
| variant (fused jobs) | ppl GPL-3 | ppl README | top-1 = fp32 |
|---|---|---|---|
| fp32 | 4.27 | 30.20 | 100% |
| w8 (per-tensor int8 weights) | 6.32 | 31.15 | 75 / 82% |
| w8a8 static | 15.28 | 73.32 | 47 / 42% |
| w8a8, position 0 on CPU | 6.95 | 36.93 | 69 / 78% |
- SmolLM3-3B survives the NPU format far better than SmolLM2-1.7B: with position 0 on the CPU, 8-bit activations
  add only +10% (GPL-3) / +19% (README) over w8. The weights are the bigger loss (+48% GPL-3, +3% README):
  weight channel groups (l2e_groups.py) are the next step for it too.
- FFN sparsity holds: 33.4% of neurons carry 95% of the down_proj energy, 87.7% of it was active in t-1..t-4,
  oracle keep-50% ppl 4.42 (fp32 4.27), top-1 88.3%.
- Size: 3B int8 = ~3.1 GB of weights > free RAM; with sparsity (~1/3 of FFN hot) the active set is ~1.6 GB.

## L2 on SmolLM2-360M-Instruct (2026-11-19, npu-llm/runs/l2-quality-360m-20261119.{log,json}, CPU only)
| variant | ppl GPL-3 | ppl README | top-1 = fp32 |
|---|---|---|---|
| fp32 | 10.93 | 39.12 | 100% |
| w8 (per-tensor int8 weights) | 12.03 | 45.14 | 85 / 73% |
| w8a8 static (NPU format) | 40.64 | 284.20 | 47 / 31% |
| w8a8 fused, position 0 on CPU | 13.58 | 58.14 | 73 / 59% |
| same + down_proj on CPU | 13.56 | 58.79 | 79 / 62% |
- Same picture as 135M: the position-0 outliers cause the collapse; with position 0 on the CPU most of the gap to
  fp32 is the per-tensor int8 WEIGHTS (12.03), and down_proj no longer needs the CPU. Greedy chat answer in the
  NPU format is coherent and close to fp32 ("Create a Dedicated Study Space ... Set Clear Goals").
- Per-tensor weights cost more here (+10% ppl vs +3% on 135M): channel grouping (several jobs with their own
  weight scale) is the next quality lever for bigger models.
- FFN sparsity is the same as 135M: 30.6% of neurons carry 95%; oracle keep 50% = 11.18 (near fp32), 30% = 12.15;
  86% of a token's set was active in t-1..t-4.
- 1.7B / SmolLM3-3B: smollm.py now reads weights memory-mapped from the safetensors (one layer expanded at a time)
  and knows SmolLM3's no-RoPE layers, so they can run, but slowly (1.7B bf16 = 3.4 GB pages from the SD card on
  every forward). Not run yet.

## L2 results (2026-11-19, npu-llm/runs/l2-quality-20261119*.{log,json}, SmolLM2-135M-Instruct, CPU only)
Quality, teacher-forced 256 tokens (GPL-3 / project README), perplexity and top-1 agreement with fp32:
| variant | ppl GPL-3 | ppl README | top-1 = fp32 |
|---|---|---|---|
| fp32 | 15.71 | 47.38 | 100% |
| w8 (per-tensor int8 weights, fp32 activations) | 16.16 | 51.18 | 84 / 82% |
| w8a8 static (the NPU format) | 50.31 | 166.84 | 43 / 37% |
| w8a8 static fused (qkv, gateup as one job) | 51.30 | 173.48 | 40 / 38% |
| w8a8 dynamic fused (scales per call) | 52.83 | 146.65 | 43 / 39% |
| w8a8 static fused, head fp32 | 47.77 | 163.14 | 44 / 39% |
- Per-tensor int8 WEIGHTS are fine (+3% ppl). Per-tensor 8-bit ACTIVATIONS are not (3x ppl), and choosing the scale
  per call does not fix it, so it is outliers inside a tensor, not calibration. Greedy chat answer degrades
  visibly (repeats itself). Fix candidates: SmoothQuant-style scaling folded into the previous norm / v / up rows,
  the worst layers on the CPU, int16 activations (hardware has int16 NN cores; driver does not use them).
  l2b_ablation.py finds the layers.
FFN sparsity (fp32, down_proj output energy per token):
| granularity | for 90% | 95% | 99% | 95% set already in t-1 | in t-1..t-4 |
|---|---|---|---|---|---|
| 1 neuron | 20.7% | 30.3% | 51.0% | 54.7% | 85.8% |
| 16 neurons | 63.6% | 76.3% | 92.0% | 83.3% | 99.1% |
| 64 neurons | 77.3% | 87.3% | 98.2% | 90.1% | 99.8% |
Oracle skipping (exact activations known): keep 50% of single neurons -> ppl 15.81 (= fp32), 30% -> 17.37, 20% ->
19.88; but keep 50% of 16-blocks -> 34.0, of 64-blocks -> 99.5 (75% of 64-blocks -> 25.2).
- Neuron-level sparsity is real (half the FFN can go almost for free), block-level is not: the active neurons are
  scattered. Next CPU idea: reorder the FFN neurons (a free permutation of gate/up rows + down columns) so that
  neurons that fire together share a block, then re-measure blocks of 16 / 64.
- History helps the predictor: 86% of a token's important-neuron energy was active in one of the 4 previous tokens.

L2b ablation (npu-llm/runs/l2b-ablation-20261119.log, GPL-3 ppl; w8 = 16.16): 8-bit activations on any ONE of
q/k/v/o/gate/up/head cost little (16.1-17.0), on all but down 17.51, on DOWN alone 45.24 (in side 25.5, out 34.1).
down_proj ranges: in up to 3164, out up to 25937 (median over layers 17 / 48). These "massive activations" sit at
sequence POSITION 0 (attention sink): down in 3211 at position 0 vs 116 max later, out 25939 vs 1056.
L2d (npu-llm/runs/l2d-fpfirst-20261119.log): position 0 in full precision (the CPU's job, once per conversation),
calibration without it:
| variant | ppl GPL-3 | ppl README | top-1 = fp32 |
|---|---|---|---|
| w8a8 static, position 0 on CPU | 20.07 | 66.55 | 65 / 62% |
| w8a8 static fused, position 0 on CPU | 20.28 | 78.05 | 64 / 63% |
| same + down_proj on the CPU (w8) | 17.47 | 59.47 | 76 / 73% |
| w8a8 dynamic fused, position 0 on CPU | 20.63 | 78.80 | 72 / 59% |
Greedy chat answer with fused + position 0 on CPU is coherent again. Plan for now: position 0 on the CPU always;
down_proj either on the CPU (w8, 25% of the layer weights) or on the NPU after more work on its later-token
outliers (out up to 1056: SmoothQuant into the up rows, split the few outlier output channels into their own job).

L2c neuron reordering (npu-llm/runs/l2c-reorder-20261119.log; order from 967 calibration tokens, tested on the
eval texts; oracle block skipping, ppl GPL-3 / README):
| blocks | order | need for 95% | keep 50% | keep 75% |
|---|---|---|---|---|
| 16 | original | 76% | 34.0 / 107.2 | 17.2 / 56.3 |
| 16 | reordered | 69% | 23.1 / 72.1 | 17.3 / 57.1 |
| 64 | original | 87% | 99.5 / 392.2 | 25.2 / 79.3 |
| 64 | reordered | 79% | 30.6 / 93.6 | 18.8 / 59.6 |
Only 0.1% of neurons are "always hot". Reordering helps 64-blocks a lot but block skipping still costs quality;
realistic NPU saving ~25% of the FFN at +20% ppl. Neuron-level skipping stays far better (keep 50% = lossless).
Concrete win: when down_proj runs on the CPU, its input h is already known, so the CPU can read only the columns
of the top-50% neurons of h with no loss (that is exactly the oracle keep-50% case, ppl 15.81 vs 15.71): half
the down_proj bytes, no predictor needed. gate/up skipping needs the predictor (L2b TAGE/perceptron idea).

## Known risks
- Any new job shape can hang the NPU (then npu-run refuses NPU jobs until a reboot). Ladders go smallest first.
- deeplabv3's 960-channel 1x1 convs were wrong from row 12 (HANDOFF of the compute port): L1's 576x960 rung and
  the wide Qwen rungs test that region.
- Per-tensor int8 may not hold LLM quality: L2 answers it before the engine depends on it.

## Log
- 2026-11-18: L0/L0b done, L1 ladder written (lib b6c2a28d = mesa-upstream-fixes, 9 patches).
- 2026-11-18: L1 PASS 18/18 (correct; M=1 4.2 GB/s + 0.30 ms; M=16 slow in this layout). SmolLM2-135M-Instruct
  downloaded to npu-llm/models/ (Jordan's choice of model). L2 code: ref/smollm.py (numpy model + NPU 8-bit
  simulation), ref/l2_quality.py.
- 2026-11-18: PAUSED by Jordan before the full L2 run. Smoke test: the numpy model is right (ppl 18.9 on GPL-3,
  greedy "The capital of France is Paris."); per-tensor w8a8 (dynamic, fused) gives ppl 53 = a large loss, the
  full run shows which layers cause it. RESUME HERE: from ~/Documents/AI_Projects run
  `OPENBLAS_NUM_THREADS=4 .venv/bin/python -I -u ~/alta-npu/llm/ref/l2_quality.py` (CPU only, ~10 min).
  Then L1b (layouts for M > 1, needs Jordan's OK).
- 2026-11-19: G1 DONE (Mali int8 dot 2.6-3.0x, correct). 1.7B L2: 8-bit format breaks (weights +26-64%, activations broken even with pos 0 on CPU) -> next quality work: l2e_groups.py (weight channel groups), 1.7B ablation. L3a hooks built + offline-verified; NPU ladder waits for OK. SmolLM3 L2 running (unit alta-llm-smollm3).
- 2026-11-19: SmolLM3-3B L2 done (8-bit OK with pos 0 on CPU: 6.95 vs w8 6.32 vs fp32 4.27; weights the bigger loss). L3 NPU ladder (Jordan's OK): baselines PASS, KPC 24 HUNG the NPU -> 1x1 cap 21 = hardware limit, lever closed; incident 2026-11-19-1106-LLM3-k24; STOP set, NPU needs a reboot + Jordan's review. Caveat: 1.7B head was 8-bit, SmolLM3's fp32.
- 2026-11-19: L2e weight channel groups running (llm/l2e-chain.sh, unit alta-llm-l2e, ~3.5 h, CPU only): 135M, 360M, SmolLM3-3B, 1.7B; G = 1/4/16/per-channel w8 + w8a8 pos0 CPU at 16/per-channel, head fp32 in every row; logs npu-llm/runs/l2e-groups-<model>.log. Smoke test (135M, 24 tokens) OK.
- 2026-11-19: L3b job builder started on the fake NPU (llm/l3/JOB-BUILDER.md): jb_run calls ml_subgraph_create/invoke directly; its gateup job is byte-identical to Teflon's (config, weights, streams, buffers); 4/16 weight groups and a gateup->down chain each go out as one submit (kpc 12-21); CPU side per invoke 0.19 ms vs Teflon 0.47 on the shim. Teflon creates+destroys a context per invoke and keeps extra weight copies. NPU ladder llm/l3/jb-ladder.sh (7 rungs) waits for reboot + STOP review + OK.
- 2026-11-19 14:00: jb ladder 7/7 PASS (Jordan's OK, boot e01fb03e): direct 0.50 vs Teflon 0.82 ms on the same job (max diff 0); fixed per submit ~0.16 ms, 5.26 GB/s inside a submit; 4/16 groups +35/14 us per extra job; gateup->down chain in one submit saves ~0.19 ms. 135M estimate 40 ms/token NPU-only (Teflon-like 57; CPU 21): submit count is now the main cost.
- 2026-11-19 16:35: L2e done on all four models: 16 weight groups ~ lossless weights everywhere (1.7B 3.56 -> 2.27, SmolLM3 6.32 -> 4.40); SmolLM3 + 8-bit activations 5.32 (usable); 1.7B activations still broken (27) -> outlier handling needed for 1.7B only.
- 2026-11-19 17:45: L3c fewer submits: cost model says split big steps (x1.1-1.7 from 360M up), NPU-only always loses, 135M decode gains nothing; gateup on the NPU ~lossless on 135M (16.05 vs 15.71). jb_run ca878c69 + split packs offline OK; split-ladder.sh (10 rungs) needs OK + quiet CPU. l2f quality on 360M/1.7B/SmolLM3 running.
- 2026-11-19 18:51: L3c PAUSED by Jordan before the split ladder (not run). l2f quality left running (CPU only); 135M/360M/1.7B done, SmolLM3 running; 360M+ results to be logged on resume.
- 2026-11-19 18:59: l2f quality run STOPPED by Jordan during SmolLM3-3B (fp32 rows only); 135M/360M/1.7B complete. Whole L3c line on hold.
- 2026-11-20: L3c resumed (boot c0a2b9d9, audit clean). l2f quality: NPU gateup = 135M ~lossless, 360M +5-12%, 1.7B broken (2.24 -> 5.7-5.9; gateup+down 19-23): NPU activations need outlier handling before the 1.7B split pays. SmolLM3 l2f rerun started; split ladder still needs OK.
- 2026-11-20 00:57: split ladder 10/10 correct (Jordan's OK). Polling wait -0.06 ms per submit, NPU 5.10 GB/s with ~0.12 ms fixed, CPU GEMV 6.6-7.5 GB/s. But side by side the split step was slower than CPU-only (360M 1.06 vs 0.75 ms, 1.7B 5.99 vs 4.93): NPU x1.8, CPU x1.3-1.7, submit 0.3-0.8 ms. Suspect the unpinned submit/poll thread on an A73 next to a worker (L1c had it on the A53s); follow-up ladder prepared.
- 2026-11-20 01:14: split ladder 2 8/8 correct: thread placement ruled out (submit 0.06-0.09 ms, pinning changes nothing). Memory bus is the limit (NPU + CPU together 7.1-7.4 GB/s vs CPU 6.7 alone; L1c's ~11 not reproduced). plan_cost.py with measured numbers: no split pays. L3c CPU+NPU split for one-token decode CLOSED; NPU use = M > 1 work. Also found: Mesa reads int8 outputs byte by byte from write-combined memory (~150 ns/byte on an A53).
- 2026-11-20 01:45: L2g outlier fixes (CPU only, Jordan: "go ahead with the outlier fix"): l2g_outliers.py; 135M smoke test: rotation exact in fp32, KL vs fp32 ~10x lower, + per-job output scales (jobs by max |w|) ~15x lower. Full runs on all four models queued after l2f (unit alta-llm-l2g, ~6-8 h).
- 2026-11-20 02:10: Mesa read-out fix (Jordan: "go ahead with the read-out fix") + KPC clamp built in mesa-llm (lib a26f0852, jb_run baca6a17); offline: jobs byte-identical to the ones that ran (incl. L1c 64/256-token jobs), clamp verified; readout-ladder.sh (12 rungs) waits for OK. Changes uncommitted (Mesa needs a Generated-by: LLM trailer on any commit).
- 2026-11-20 02:17: read-out ladder 12/12 PASS (Jordan's OK): read-out x9-12 faster; Teflon 1-token invoke 0.85 -> 0.63 ms, 256-token 29.48 -> 7.21 ms; jb_run 256-token gateup 25.6 -> 5.6 ms. Remaining limit: uncached bulk read ~0.32 GB/s -> cached output buffers next for M > 1.
- 2026-11-20 02:39: l2f SmolLM3 done (NPU gateup +2-6% ppl, gateup + down 4.30-5.51 / 33); L2g 135M full run: rotation + per-job output scales 16.06 / 47.67 vs fp32 15.71 / 47.38 (baseline 19.48 / 57.55), KL 15x lower. 1.7B running.
- 2026-11-20 03:10: (Jordan: "go ahead with the commit and the upstream patch") mesa-llm commits 1c441f99 (read-out) + 2cd9c87d (KPC clamp), trailer Generated-by: LLM. Upstream series: patch 10 on alta-etnaviv-ml-fixes (tip 328286da6e0, lib 9574fdd1, comment rewrapped to 78 columns), series regenerated as 0001-0010 (contents unchanged), README + MR-DESCRIPTION updated, validate_offline 34/34. Not sent: Jordan sends.
- 2026-11-20 02:58: test_teflon with all ten patches (Jordan's OK): 1503 passed / 34 failed, the same 34 tests as with nine, no hang. Rewrite-list artifact (claude.ai/artifact/3vuFTmWChu7HGLKXKnne3m) updated for patch 10 (version 5).
- 2026-11-20 03:40: cached output buffers built in mesa-llm (lib 6caca26e) + stale-read test (jb_pack altinput, jb_run --alt, ad393a6f) + fake-NPU buffer log (shim f1cf88d5, NPU_SHIM_BO_LOG); offline PASS; cached-ladder.sh (14 rungs) waits for OK.
- 2026-11-20 03:20: cached-ladder 14/14 PASS (Jordan's OK): cached outputs correct (2800 alternating invokes, 0 stale), big read-outs x1.7-1.8 faster (256 tokens 2.61 -> 1.48 ms), Teflon 256-token 7.19 -> 6.29 ms, small outputs unchanged. Committed later as 415be4bf.
- 2026-11-20 03:35: (Jordan: "go ahead with the commit and patch 11") mesa-llm commit 415be4bf (cached graph outputs, lines wrapped to 78 columns; wrapped build a77ae542 = tested build offline). Upstream patch 11 on alta-etnaviv-ml-fixes (tip 470cdf1729d, lib 6a3290ab), validate_offline 34/34, series regenerated 0001-0011, README / MR-DESCRIPTION / rewrite-list artifact (v6) updated. test_teflon with 11 patches not run (needs OK).
- 2026-11-20 03:39: test_teflon with all eleven patches (Jordan's OK): 1503 / 34, the same failed tests, no kernel warning. README, MR-DESCRIPTION, rewrite-list artifact (v7) updated.
- 2026-11-20 03:47: perf installed with sudo (root step 230, Jordan's OK in chat; password not stored). Read-out profile: -128 byte loop ~0.9 ms of 1.56, memcpy ~0.25, kernel invalidate + clean ~0.4; a fused 64-bit conversion would roughly halve it.
- 2026-11-20 04:05: fused int8 conversion built in mesa-llm (lib fa818da9, jb_run a070d63e), offline + unit test PASS; fused-ladder.sh (14 rungs) waits for OK.
- 2026-11-20 04:00: fused-ladder 14/14 PASS (Jordan's OK): read-out x2.9 (256 tokens 1.55 -> 0.54 ms), input x2.6, Teflon 256-token invoke 6.28 -> 4.04 ms; no stale output in 3600 checked invokes. Uncommitted in mesa-llm.
- 2026-11-20 04:10: (Jordan: "go ahead with the commit and patch 12") mesa-llm commit 792a8e97; upstream patch 12 (tip 940b610ac79, lib 623a3f1e; the conflict was only patch 10's rewrapped comment, which patch 12 replaces; files byte-identical to mesa-llm), validate_offline 34/34, series 0001-0012, README / MR-DESCRIPTION / rewrite-list artifact (v8) updated. test_teflon with 12 patches needs OK.
- 2026-11-20 04:17: test_teflon with all twelve patches (Jordan's OK): 1503 / 34, the same failed tests, no kernel warning. README, MR-DESCRIPTION, rewrite-list artifact (v9) updated. The series has hardware evidence for every patch.
- 2026-11-20 04:30: zero-copy read-out hook in mesa-llm (uncommitted, test tree) + jb_run a79b7b5c; offline PASS; zero-ladder.sh (14 rungs) waits for OK.
- 2026-11-20 04:30: zero-ladder 14/14 PASS (Jordan's OK): zero-copy read-out + consumer 0.60 -> ~0.48 ms at 256 tokens (the data still comes from DRAM once), the no-fini experiment ~0.30 ms (kernel clean pass ~0.19 ms); no stale output. Remaining read-out cost = kernel cache maintenance.
- 2026-11-20 05:31: L2g 1.7B done: rotation + per-job output scales 2.63 / 25.43 vs fp32 2.18 / 24.49 (current format 27.20 / 416.23), 4 groups 2.52 / 25.38; rotation or per-job scales alone are not enough (4.58 / 30.40, 13.37 / 163). The 1.7B 8-bit break is fixed. 360M + SmolLM3 running.
- 2026-11-20 12:00: L2g chain finished on all four models (360M 05:45, SmolLM3 09:58): rotation + per-job output scales, every step 8-bit, within 0-4% ppl of fp32 (1.7B GPL-3 +21%), KL 10-38x lower than the current format; 4 groups ~ 16. Board idle.
- 2026-11-20 12:30: R2 started (Jordan: "go ahead with the real-weight NPU packs"): llm/l3/real_packs.py builds 240 packs of SmolLM2-135M (30 layers x 4 steps x 1 / 256 tokens; rotation folded in, 4 jobs per step, per-job output scales, real inputs); offline all 240 PASS (one submit each, kpc <= 20); llm/l3/real-ladder.sh waits for OK.
- 2026-11-20 12:35: R2 real-weight ladder 240/240 PASS (Jordan's OK): every layer step of SmolLM2-135M on the NPU within 1 level of the reference; all 120 steps per token: 1 token 30.6 ms, 256-token chunk 124.1 ms = 0.485 ms per token (llama.cpp pp128: 9.6 ms per token whole model). Next: the CPU glue and a full chunk end to end. Session ends here (Jordan switches to a remote session).
- 2026-11-21 15:50: R2 step 1 (Jordan: "go ahead with the engine process"): llm/l3/jb_engine.c compiles all 120 steps in one process and runs them back to back; offline PASS (8-9.5 s compile, peak 119 MB at 1 token = the weights once, one submit per step, streams as alone). llm/l3/engine-ladder.sh (2 rungs) waits for OK.
- 2026-11-21 15:40: engine ladder 2/2 PASS (Jordan's OK): all 120 steps in one process, back to back; 256-token chunk 140.0 ms (0.547 ms/token), 1 token 32.7 ms, compile 2.5 s. Next: the CPU glue.
- 2026-11-21 16:20: L2h groups test 135M + 360M: 4 jobs per step best, 2 jobs ~1.5x the KL, 1 job ~2-4x (360M README +17% over fp32); 1.7B running. Root step 240 (hw_job_limit=4) written, not run.
- 2026-11-21 16:40: handoff (context window full). Engine process PASS on the NPU; groups test 135M/360M done, 1.7B partial; root step 240 superseded by step 250 (job-queue driver). Resume from "Resume here".

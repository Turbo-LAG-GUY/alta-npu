# NN job config: the caching / pipelining fields Mesa never sets (idea 3, research started 2026-11-22)

Jordan's hypothesis (2026-11-22): the NPU's cache is under-used / misused; data jumps around instead of being
pipelined and prefetched. Two findings the same day back it up: Mesa dropped the input image cache whenever the
weights didn't fit (fixed: ETNA_ML_KEEP_IMAGE_CACHE, 1-job chunk 211 -> 104 ms), and a 16x16 tile made the 1.7B
input too big for the cache (ETNA_ML_V7_TILE_FIT / TILE_H: 2.3x). This note lists the remaining fields of
`struct etna_nn_params` (mesa-llm etnaviv_ml_nn.c) that look like caching or pipelining controls and that Mesa leaves
at zero or hard-codes, what each might do, and how to test it.

**Sources.** Field names come from Tomeu Vizoso's reverse engineering of the vendor blob's command streams
(blog.tomeuvizoso.net, "Etnaviv NPU update 4", 2023), which lists them without meaning. No public documentation was
found (web search 2026-11-22). Vendor VIPLite documents exist on a forum but are marked "CONFIDENTIAL - NOT FOR
REDISTRIBUTION": not read, not used (Jordan's call). So: hypotheses + experiments, offline first, then careful NPU
rungs with Jordan's OK (a wrong config can hang the NPU; npu-run then refuses jobs until a reboot).

## Where the time is (for scale)
- 8-job 1.7B-shaped step (2048 -> 8 x 224, 256 tokens) with tile 16x8 + keep-image: 2.81 ms = **0.67 TOPS**, vs
  the best 1x1 rate ever measured 0.79 TOPS (3x3: 1.73). The best case is now near the 1x1 compute ceiling.
- 135M 256-token chunk, 1 job per step: 104 ms = NPU ~55 + read-out ~30 + 120 submits ~19 (CPU glue ~10 s in numpy).

## Result 2026-11-22 15:26: nn_layer_flush = 0 HANGS THE NPU
Plan step 1 ran (tile-ladder S2, Jordan's OK): job 0 of 2 with nn_layer_flush = 0 (only difference) -> every invoke
"MMU 0 fault (page not present) addr 0x00000000", 20 recovered hangs, STOP latched (incidents/2026-11-22-1526-LLMT-S2-
noflush). So the bit is not a plain "wait for writes": without it the NN unit fetches something at address 0, most
likely the next layer of a chained layer list it expects in the descriptor. Mesa's hook is now refused unless
ETNA_ML_NO_FLUSH_INDEPENDENT=I-know-it-hangs. Any further work on job chaining needs a theory of that descriptor first.

## Candidates

| field(s) | Mesa today | hypothesis | why it could matter here |
|---|---|---|---|
| `nn_layer_flush` (1), `no_flush` (v8 only) | flush after every NN job | a flush makes the next job wait for this one's writes | a step's G jobs read the same input and write separate outputs: **no dependency**, so no flush is needed between them. Dropping it may let job k+1's weight/input fetch overlap job k's compute (pipelining across jobs). Cheapest test of the hypothesis. |
| `kernel_pattern_msb/low/high` | 5 hard-coded patterns chosen by out_z | in PARTIAL kernel cache: a repeating bit pattern (msb+1 bits) of which kernels are kept in the cache | with tile-fit there are 2+ tiles per job, so uncached kernels are re-read per tile; a pattern computed to exactly fill the kernel area (instead of the blob's fixed guesses) could cut that. The hardware coped with patterns asking for more than the area (down 2/4 jobs). |
| `partial_cache_data_unit` (2 bits) | 0 | granularity of the partial cache | goes with the pattern |
| `kernel_direct_stream_from_VIP_sram` | 0 | the kernels are read from the VIP SRAM instead of DDR, i.e. placed there beforehand | if a job (or an earlier one) can leave weights in the VIP SRAM, the next job could stream them from there: weight prefetch. Who writes them there is unknown (a TP job? the previous job's kernel cache?). |
| `in/out_image_circular_buf_size`, `*_end_addr_plus_1` | 0 / all ones (off) | the input / output is a ring buffer of that size | producer -> consumer pipelining through a small ring (e.g. gateup output feeding down) without the whole tensor ever existing; the vendor's layer-fusion ("tiling" in its tools) likely uses this together with strips. Needs strip-interleaved jobs (as mesa-strips patch 0005 does for 3x3 chains). |
| `brick_mode`, `brick_distance` | 0 | an input fetch / layout mode for tiles ("bricks") | unknown; maybe overlapping tile reads for KxK convs; likely irrelevant for 1x1 |
| `in_image_x/y_stride`, `out_image_y_stride` | = sizes | windows into bigger tensors | known from the strips work: in y_stride works as rows per plane, out y_stride is ignored; x-stride windows + x offsets HUNG the NPU (incident 2026-11-12-1003-STRIPS-C64): avoid |

Also not a field but the same theme: **the 1x1 ceiling itself** (0.79 vs 1.73 TOPS for 3x3). A matmul has no input
reuse across kernel positions; whether a 2x2 / 3x3 reformulation (channels laid out spatially) can use the MAC array
better is an open question (Mesa turns stride-2 convs back into 1x1 via its reshuffle, so it needs care).

## Plan (each step offline on the fake NPU first: decode the configs, check they are what was meant)
1. **No flush between independent jobs** (hook `ETNA_ML_NO_FLUSH_INDEPENDENT=1`: `nn_layer_flush = 0` on every job of
   a step except the last). Rungs: 2 jobs, then 4, then 8 on the axi-1p7b packs; every output checked. Expected: a
   gain only if the hardware overlaps jobs; risk: wrong outputs (caught) or a hang (unlikely: no shared writes).
2. **Computed kernel patterns**: offline first, a table of (out_z, kernel area, coef size) -> the blob patterns, to
   see whether they are fractions (cached kernels / all) and of what; then one computed pattern on one pack.
3. **kernel_direct_stream_from_VIP_sram**: only after 1-2, and only with a clear theory of who fills the SRAM.
4. **Circular buffers + strips for gateup -> down**: depends on idea 2 (the FFN glue on the NPU; quality simulation
   llm/ref/l2i_npu_mlp.py) and on mesa-strips experience; largest and riskiest.

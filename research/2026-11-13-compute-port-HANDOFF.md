# Handoff: compute on the NPU shader core, port to current Mesa (written 2026-11-13, boot d5fe88d6)

Read first: `research/2026-11-13-etnaviv-compute-port-survey.md` (what to port and the plan), memory note
`npu-shader-core.md`, and the last lines of `PROGRESS.md`. Rules as always: audit first, NPU jobs only with Jordan's
explicit OK for that step and only via `scripts/npu-attempt`, builds at -j2, never rebuild the kit library
(`mesa-axi-sram/build`, sha 41b08ede).

## START HERE (updated at the end of the second session, 2026-11-14 ~00:00, boot 8f7f01ab)

Second session in one line: **6d global AVGPOOL on the shader core** (MobileNet in one NPU submit, 6.77 vs 7.54 ms,
byte-identical), **6e loops/branches work** (S6E-LOOP*), **6e MEAN on the shader core** (channel + spatial, exact to the
CPU's XNNPACK, `ETNA_ML_SHADER=mean`). Read the "Step 6d" and "Step 6e" bullets below (end of this START HERE block),
then the **NEXT (d)** entry and the two **open findings** there. Patches added this session: research/port/
step6d-avgpool.diff (base research/port/base-step6c), step6e-loop-harness.diff, step6e-mean.diff (base
research/port/base-step6d); tools in research/port/tools/ (reduce_models.py, mean_fit_xnnpack.py, make_mean_tests.py,
teflon_check.py copy). Switches can be combined: `ETNA_ML_SHADER=transpose,softmax,avgpool,mean`. Lesson from 6e:
judge a shader op by ON vs the CPU reference (exact), not only ON vs OFF - OFF can be wrong (finding 1).

### State at the end of the first long session (2026-11-13 22:20, boot 8f7f01ab)

- Port tree `~/Documents/AI_Projects/mesa-compute` (never meson-regenerate its build; add sources with
  research/port/ninja_add_source.py). Patches in order: research/port/step2-driver.diff, step3-compiler.diff,
  step5-shared-barrier.diff, step6-ml-shader.diff, step6c-softmax.diff (p_state.h + tfl_device.c are in 6c).
  Kit library mesa-axi-sram 41b08ede untouched.
- Proven on the NPU: compute dispatch (copy, fma 6.15 GFLOPS), shared memory + barriers (workgroups up to 64),
  2D grids, float ops (exp/rcp ~1 ulp), Teflon layout transposes on the shader core inside the NN submit
  (`ETNA_ML_SHADER=transpose`, conv3x1 103 -> 46 ms, byte-identical), softmax on the shader core
  (`ETNA_ML_SHADER=softmax`, MobileNet's SQUEEZE + SOFTMAX on the NPU, 50/50 outputs byte-identical). Switches can
  be combined ("transpose,softmax"); with none set the port tree's Teflon = kit library (identical streams).
- Hard limit found: a workgroup with barriers needs all its threads resident; threads x temps above 512 vec4
  registers hangs the core (576 hung twice: incidents 2026-11-13-2109-S6C-SM1001, 2026-11-13-2143-S6C-REGP6).
  Guard `etna_compute_barrier_reg_budget()` = 512 (env ETNA_CS_BARRIER_REG_BUDGET); Teflon ops whose kernel
  doesn't fit any of 64/32/16 threads stay on the CPU automatically.
- Tools: npu-shader/cs_run (kernels in cs_kernels.h + the driver's own builders; `--offline` with the shim),
  cs_compile (offline compiler, no device), teflon_check.py (every invoke vs first / CPU / saved output),
  research/port/cmdstream_decode.py, shader_diff.sh. Offline Teflon: run the Python under
  `bwrap --dev-bind / / --dev /dev` + `strace -E LD_PRELOAD=<ABSOLUTE path>/shim/libnpu_etna_shim.so` with
  ETNA_ML_DISK_CACHE_DIR in scratch. Ladder scripts: research/port/step6c-ladder.sh (run detached with
  systemd-run --user). After an NPU hang npu-run refuses NPU jobs until a reboot; Jordan clears runs/STOP.
- Next candidates (Jordan to choose): (1) global AVGPOOL on the shader core - the op that splits MobileNet into two
  NPU submits, no barriers needed (design notes in step 6c below); (2) layernorm-style reductions for transformer
  graphs; (3) transpose tuning: 16x4 workgroups, the ETNA_ML_SHADER_MIN_CHANNELS crossover vs the TP.
- **Step 6d IN PROGRESS (second session, started 2026-11-13 22:25, boot 8f7f01ab): global AVGPOOL**, Jordan's order
  "avgpool, then layernorm-style reductions, then transpose tuning". Base snapshot of the step-6c files:
  research/port/base-step6c/ (patch step6d = diff against it). Done so far (CPU only): `ETNA_ML_SHADER=avgpool`;
  kernel `etna_ml_shader_build_avgpool(opts, hw, channels)` in etnaviv_ml_shader.c (one thread per 4 channel planes,
  unrolled word loads, 16-bit lane sums, (sum + hw/2)/hw as a multiply-shift checked exhaustively at build time; no
  barriers); support = global pool, batch 1, uint8 same quant in/out, no fused activation (new p_state.h field
  `pooling.fused_activation`, set in tfl_device.c), C % 4 == 0, hw <= 256; etnaviv_ml.c: POOLING supported/lowered/
  counted, its input needs NCHW (transpose added otherwise). CPU facts: MobileNet node 27 = 7x7x1024, scale 0.0235 zp 0
  in and out, CPU output = (sum + 24) // 49 on all 1024 channels; algorithm model exact for hw 1..256.
  Harness: `cs_run --kernel avgpool --hwc H W C -w WG`, `cs_compile --kernel avgpool --hw HW --ch C`.
  **CPU part DONE ~22:55** (patch research/port/step6d-avgpool.diff, 9 files, against base-step6c; build clean -j2,
  201 steps because p_state.h changed; kit lib 41b08ede untouched). Offline: kernel 7x7x1024 = 392 instructions,
  5 temps, 17 vec4 uniforms (15x15: 1668 / 61, 16x16: 1883 / 67 = longest programs yet); shim streams differ from the
  proven copy256 only in program length/address, temps, uniforms (runs/cs-offline/step6d-shim). MobileNet on the shim
  (runs/cs-offline/step6d-teflon): port tree switch off = kit library, 3 submits byte-identical incl. BO lists;
  `avgpool` -> the pool joins partition 1 (one inference submit; SQUEEZE+SOFTMAX stay on the CPU);
  `avgpool,softmax` -> all 31 nodes one partition, one submit: NN ... NN, flush+stall, SH pool (tensor 30 -> 0),
  flush+stall, NN conv, flush+stall, SH softmax; pool VAs in rw BOs with room.
  **6d DONE 2026-11-13 22:46 (boot 8f7f01ab, Jordan's OK "You have the OK, run the ladder"): ladder all 8 PASS, 0 NPU
  kernel lines, no STOP** (RESULTS.md S6D-*, runs/cs6d-ladder-20261113-224334). Pool kernel exact, 0 stray bytes:
  7x7x1024 0.28 ms/launch, 1x1x256 0.12, 15x15x64 0.21, 16x16x32 0.20 (1883 instructions / 67 vec4 uniforms run fine,
  the longest programs yet). MobileNet: OFF 7.54 ms (2 partitions), `avgpool` 7.16 (10/10 byte-identical),
  `avgpool,softmax` = the whole model in ONE NPU submit, 6.77 ms (10/10) and 7.04 over 50 invokes (50/50
  byte-identical); about 0.4-0.8 ms saved per invoke (single runs each).
- **Step 6e IN PROGRESS (started 2026-11-13 ~23:00, same session, Jordan: "Start the layernorm reductions")**; then
  transpose tuning (16x4 workgroups, ETNA_ML_SHADER_MIN_CHANNELS crossover vs the TP). Findings so far (CPU only):
  - TFLite has no layernorm op: converters emit MEAN(axis -1) -> SQUARED_DIFFERENCE -> MEAN -> ADD(eps) -> RSQRT ->
    MUL/SUB/ADD. Gallium/Teflon have no MEAN (needs a new PIPE_ML_OPERATION_TYPE + tfl_device.c mapping with the
    axes from the constant input 1). Teflon right-aligns dims to 4 ([1,T,C] -> [1,1,T,C]), so remap axes. MEAN over
    axes 1,2 is also how Keras models (MobileNetV2/V3, EfficientNet) do global pooling.
  - **Reference = XNNPACK, not TFLite's builtin kernels**: with Teflon loaded the leftover CPU ops run on XNNPACK
    (shim test, 109/109 discriminating inputs). XNNPACK's quantized MEAN (source read via GitHub, then fitted):
    `q = clamp(rne(fma(S, s_eff, zp')))`, `zp' = fma(-(N*zp_in), s_eff, zp_out)` (float), `S` = raw int32 sum of
    the stored values, `s_eff = fl(fl(s_in/s_out) * fl(1/N))`: exact on 205200/205200 outputs (11 cases: u8/i8,
    channel/spatial, same/different quant). Builtin TFLite differs (integer QuantizeMultiplier path).
    Tools: npu-shader/models/reduce_models.py (`make` MEAN test models, optional 1x1 CONV in front), mean_fit_xnnpack.py.
  - Kernel plan for MEAN: sum the stored bytes (etnaviv stores int8 +128, so the sum is unsigned in both cases), an
    integer estimate q0 = (S*M + B) >> sh, then q = q0 + (S >= T[q0+1]) - (S < T[q0]) from a 257-entry threshold table
    the driver computes on the CPU from the XNNPACK formula (fmaf + nearbyintf) over every possible S, verifying
    |q0 - q| <= 1 at the same time: exact by construction, independent of the shader core's float rounding.
    Channel MEAN on NCHW planes, P = H*W positions, P % 4 == 0: one thread per 4 positions, one word per plane,
    SWAR lanes (widen every 256 channels). Spatial MEAN: the avgpool kernel with the table requant.
  - **Loops**: C = 384..768 unrolled would be ~5000 instructions, so a loop probe: cs_kernels.h `cs_build_loopsum`
    (15 instructions: compare, conditional branch, body, branch back; runtime trip count capped at 4096), compiles,
    shim clean (runs/cs-offline/step6e, step6e-shim). **Loop ladder DONE 2026-11-13 23:22 (boot 8f7f01ab, Jordan's
    OK): LOOP1 1x1, LOOP64 64x100, LOOP256 256x1000 all exact, 0 NPU kernel lines, words past the output untouched;
    256x1000 = 2.80 ms/launch (~365 MB/s strided). Loops and branches work on the shader core** (RESULTS.md S6E-LOOP*,
    runs/cs6e-loop-20261113-232209). The harness pattern: nir_local_variable_create + nir_push_loop / nir_push_if +
    nir_jump(break); keep a compile-time cap on the trip count inside the kernel.
  - **MEAN CPU part DONE 2026-11-13 ~23:50** (Jordan: "Start the MEAN kernel"; patch research/port/step6e-mean.diff
    against research/port/base-step6d; build clean -j2; kit lib 41b08ede untouched): (a)-(c) below are implemented.
    `ETNA_ML_SHADER=mean`; channel MEAN = etna_ml_shader_build_mean_channels (loop over channels, 89 instructions,
    6 temps, kernel input out, in, table, channels, plane bytes, last thread); spatial MEAN = build_pool with the
    table (avgpool's binary stays byte-identical to 6d); etna_ml_mean_requant builds + checks table and estimate
    over every S; table = etna_vip_instruction.table, a third grid buffer, freed in etna_ml_shader_instruction_fini.
    Harness `cs_run/cs_compile --kernel mean --axes ch|sp --hwc H W C [--int8] [--q IS IZ OS OZ]` (cs_run's
    reference is an independent copy of the formula). Test models npu-shader/models/mean/ (make_mean_tests.py, copy in
    research/port/tools): mean_ch384_u8, mean_ch768_i8, mean_sp7x7_u8 (1x1 CONV in front), mean_ch64_in_u8 (MEAN on
    the graph input). Offline: switch off = kit streams; on = TP, NN conv, SH mean in one submit.
    **Ladder run 1 2026-11-13 23:38 (Jordan's OK; runs/cs6e-mean-20261113-233843): kernel rungs PASS exact, 0 stray,
    0 NPU kernel lines** (MCH384 64/64 0.33 ms, MCH768I8 32/32 0.56, MSP7 1024/1024 0.27, MCH128x768 128/128 0.71):
    the MEAN kernels, signed shift/selects and the table as third buffer work on the NPU. **Stopped at S6E-T384-OFF**
    (rc 3 = teflon_check mismatch, no STOP): the synthetic identity-like 1x1 CONV in front of MEAN differs on the NN
    from the CPU (MEAN after it off by max 22 / mean 6.8 levels vs the 2.0 limit) - an NN-side deviation for that
    weight pattern, worth a look later, not a shader issue. Fix: teflon_check --cpu-mean-limit (default 2.0 unchanged);
    the ladder's conv-model rungs pass 32 and are judged by ON == OFF byte for byte.
    **Run 2 23:42 (Jordan's OK, runs/cs6e-mean-20261113-234220): T384-OFF PASS (limit 32); T384-ON output EXACT vs
    the CPU reference (diff 0) but != OFF - and OFF is the wrong one.** Correction of the note above: the NN conv is
    fine; **etnaviv bug (kit library too): lower_operations adds the partition-output detranspose only if is_3d (H>1 &&
    W>1 && C>1), so a [1,1,W,C] / [1,H,1,C] NN output reaches the CPU still in channel planes** (confirmed byte-exact on
    the CPU with a numpy model of the misread). Affects transformer-shaped [1,1,T,C] partition outputs, not MobileNet.
    Possible fix (needs Jordan's decision, it changes switch-off streams for those shapes): detranspose when C > 1 and
    H*W > 1. Ladder revised: ON rungs must equal the CPU exactly (--cpu-mean-limit 0), OFF of [1,1,T,C] models only
    documents the bug. **Run 3 23:45 (runs/cs6e-mean-20261113-234507): S6E-T384-ON PASS - the shader MEAN inside
    Teflon is EXACT vs the CPU (10/10, 2.42 ms).** Stopped at the T768-OFF documentation rung: int8 [1,1,32,768] OFF
    is off by mean 151 levels and neither the planar-misread model nor planar+128 reproduces it (read_output does
    subtract 128), so the int8 OFF cause is open; T768-ON (exact vs CPU) will show whether the NN int8 conv is right.
    OFF documentation rungs are now non-gating (`doc`). **Run 4 23:47 (runs/cs6e-mean-20261113-234733): T384-ON PASS
    exact again; T768-ON rc 3: int8 OFF and ON are both all 127 - the NN's int8 conv saturates on this synthetic model**
    (second NN-side finding, worth a look later with a real int8 model; explains the unmatched int8 OFF). int8 inside
    Teflon is now tested without the NN: new model mean_ch64_in_i8 (int8 MEAN on the graph input, TP + SH, offline OK);
    T768-ON checks ON == OFF only.
  - **MEAN DONE 2026-11-13 23:53 (run 5, Jordan's OK, runs/cs6e-mean-20261113-235031 + kernel rungs of run 1): the
    shader-core MEAN works inside Teflon.** Every ON rung exact vs the CPU (XNNPACK) where checkable: T384 (64 tokens x
    384, uint8) 10/10 and 50/50, TSP7 (7x7x1024 spatial) 1.10 ms vs 1.73 OFF, TIN64 uint8 and TIN64I8 int8 on the graph
    input; ON == OFF byte for byte wherever OFF is valid. 0 NPU kernel lines, no STOP. RESULTS.md S6E-M*/T*.
  - **Open findings (not shader issues, need Jordan's decision):** (1) etnaviv skips the partition-output detranspose
    when H or W is 1 (is_3d in lower_operations), so [1,1,T,C] NN outputs reach the CPU in channel planes - kit library
    too; fix = detranspose when C > 1 && H*W > 1 (changes switch-off streams for those shapes). (2) The NN's int8 conv
    saturates (all 127) on the synthetic identity-like int8 1x1 conv of mean_ch768_i8 - check with a real int8 model.
  - **NEXT (d): the rest of a decomposed layernorm** - SQUARED_DIFFERENCE, RSQRT, MUL (elementwise, exact to XNNPACK
    like MEAN: fit each formula against the interpreter first, research/port/tools/mean_fit_xnnpack.py shows how), then
    a test model with the whole MEAN -> SQUARED_DIFFERENCE -> MEAN -> ADD -> RSQRT -> MUL/SUB/ADD chain. Fixing
    finding (1) first would make the [1,1,T,C] OFF baselines meaningful. Then transpose tuning.
  - (plan, kept for reference) (a) gallium `PIPE_ML_OPERATION_TYPE_MEAN` (axes bitmask over
    the 4 right-aligned dims + keep_dims) and tfl_device.c mapping of kTfLiteBuiltinMean (axes from the constant
    input 1, negative axes, rank remap); (b) etnaviv_ml_shader.c: threshold table builder (C, fmaf/nearbyintf, the
    formula above, verify |q0 - q| <= 1 over every S) + channel-MEAN kernel with a loop over C (thread per 4
    positions, NCHW, P % 4 == 0; MEAN input needs NCHW like the pool) + spatial MEAN via the avgpool kernel with the
    table; table in a BO bound as a third buffer; (c) test models from reduce_models.py make (--conv puts an NN op in
    front), offline shim (switch off = kit streams), CPU reference = XNNPACK (teflon_check.py's CPU run), then an NPU
    ladder; (d) after MEAN: SQUARED_DIFFERENCE, RSQRT, MUL for a whole decomposed layernorm.

## What works today (proven on this board, RESULTS.md 2026-11-13)

- **Compute on the shader core via the 2023 Collabora branch**, unmodified, in `~/Documents/AI_Projects/mesa-cl2023`
  (git archive of `refs/remotes/cl/tomeu-etnaviv-opencl` from `~/Documents/AI_Projects/mesa`; etnaviv-only meson
  build, configure log `../mesa-cl2023-configure.log`).
  Harness `~/Documents/AI_Projects/npu-shader/cl2023_copy` (source `cl2023_copy.c`, build `build_cl2023.sh`, flags
  `cl2023_flags.txt`): builds the kernel as NIR in rusticl's form and runs it through the branch's compiler + launch_grid.
  - copy kernel: `cl2023_copy -n 256 -w 64` (also 1/1, 64/64) - PASS
  - float multiply-add: `cl2023_copy -n 65536 -w 64 --fma 256 --repeat 10` - PASS, 6.14 GFLOPS fp32
  - compile only: add `--compile-only` with `ETNA_MESA_DEBUG=shaderdb,dump_shaders`; `--out DIR` collects dumps
  - Lesson: this branch's load/store_global need DEST_TYPE/SRC_TYPE, else the compiler asserts "Unhandled base type".
  - Reference dumps of a working compiled shader + command stream: `runs/cl2023-180102/`, `runs/fma-180621/`.
- **Hand-built bring-up** `npu-shader/hello` (+ `hello-mesa`, softpin): stores work; loads built from Mesa's decoder
  test words do not (the 2023 compiler's loads do). Kept for reference only; the 2023 path superseded it.
  One hang from it: incident `incidents/2026-11-13-0535-SHADER-HELLO` (256-thread indexed copy, MMU fault).
- **Planar setting** (separate topic, done): `npu-bench/npu_planar.py` (NPU_PLANAR=auto|on|off) and
  `npu-upscale/worker.py` (auto swaps in a planar copy in npunet mode). NPU: 3.6x on conv3x1, worker 14.7 vs 24.2 ms.

## Next steps (in order)

1. DONE 2026-11-13: port tree `~/Documents/AI_Projects/mesa-compute` (cp -a of mesa-axi-sram; all 16 absolute paths
   in build/build.ninja retargeted; touch build.ninja). Its Teflon rebuild touched 16 steps, nothing refers to the kit
   tree; code differs from the kit library only through encode.h, whose generator orders fields differently each run.
   No git commits in it (none without Jordan asking): the port's patch = diff against the untouched `mesa-axi-sram`.
2. DONE 2026-11-13 (CPU only, builds clean at -j2, Teflon in the port tree relinks; kit library still 41b08ede):
   driver side. Patch: `research/port/step2-driver.diff` (diff against mesa-axi-sram, 10 files). New
   `etnaviv_compute.c`: compute CSO (compiled at create, one variant), set_global_binding (softpin VA into the
   handle), launch_grid, lazy compute reset (PA_SYSTEM_MODE, API_MODE=OPENCL, SH_CONFIG, emitted by the first grid
   after a flush, so Teflon's streams don't change), etna_emit_compute_state = the 2023 state sequence, compute
   uniform writer (LOAD_STATE per vec4 as in 2023), dumps (ETNA_MESA_DEBUG=dump_shaders writes the same file names as
   2023: mesa-shader-*.bin, base_submit_*_cmdstream). Screen: caps.compute + compute caps + compute shader caps, only
   for HALTI5 cores with COMPUTE_ONLY (the NPU); hooks only on compute-only contexts. Refused with BUG(): indirect
   dispatch, nonzero grid_base, images/samplers. Not yet: shared/scratch memory (step 5).
   **Contract for step 3 (what the compiler must fill in):** `variant->workitem_funcs[3]` (ID inputs in register
   order; current NIR name `nir_intrinsic_load_global_invocation_id` replaces 2023's `..._zero_base`, plus
   `load_local_invocation_id`, `load_workgroup_id`), `variant->has_barriers`, `num_temps` (driver adds 1 for the
   store scratch register). Kernel arguments: uniform contents `ETNA_UNIFORM_KERNEL_ARG` with data
   `ETNA_KERNEL_ARG(byte_offset, size_bytes)` (replaces 2023's `UNIFORM(bit_size, offset*4)` + `sizes[]` array, so
   the disk cache format is unchanged); `ETNA_UNIFORM_GROUP_COUNT` (data = dimension); `ETNA_UNIFORM_UNIFORM`
   (dword index into constant buffer 0) also works. The kernel input buffer is constant buffer 0's user_buffer:
   current gallium has no `req_input_mem` / grid `input`, so the harness must call set_constant_buffer (it does).
   Register renames since 2023 (same addresses): 0x958-0x960 CL_HALTI5_UNK* -> `VIVS_CL_GLOBAL_SCALE_X/Y/Z`;
   2023's PS_NEWRANGE_HIGH (0x1090) -> `VIVS_PS_HALTI5_RANGE_HIGH` (today's PS_NEWRANGE_HIGH is 0x880, a different
   register); VS_ICACHE_CONTROL (0x868) -> `VIVS_SH_ICACHE_CONTROL`; raw 0x109C -> `VIVS_PS_PSCS_THROTTLE`.
   **Build rule for this tree: never let meson regenerate `mesa-compute/build`** (its coredata points at
   subprojects in Jordan's `~/Documents/AI_Projects/mesa`). A new source file goes into meson.build *and* by hand
   into build.ninja with `research/port/ninja_add_source.py FILE.c` (copies the etnaviv_context.c.o edge, adds the .o to the libetnaviv.a line, touches build.ninja;
   step 2 did this by hand, backup `build/build.ninja.pre-compute-port`); `ninja -n` must show no
   "Regenerating". Build: `ninja -j2 src/gallium/drivers/etnaviv/libetnaviv.a src/gallium/targets/teflon/libteflon.so`.
   Tools: `research/port/cmdstream_decode.py DUMP [--diff OTHER]` names every state write (2023 baseline decoded
   cleanly); the branch's own driver diffs are in `research/port/branch-*.diff` (base commit c12f7f601c3).
3. DONE 2026-11-13 (CPU only, builds clean at -j2 with no warnings, Teflon in the port tree relinks; kit library
   still 41b08ede): compiler side. Patch: `research/port/step3-compiler.diff` (6 files, ~190 lines). All new
   behaviour is gated on compute stages or compute-only intrinsics, so graphics output is unchanged.
   - Work-item IDs (`load_global_invocation_id`, `load_local_invocation_id`, `load_workgroup_id`): live from the
     start, pinned by RA to t0/t1/t2 in first-read order, which is recorded in `workitem_funcs`.
   - `load_kernel_input` (constant offset, 32-bit) -> uniforms `ETNA_UNIFORM_KERNEL_ARG` (offset, 4 bytes).
   - `load_global`/`store_global` (32-bit components; store write mask must be .x/.xy/.xyz/.xyzw): LOAD/STORE
     u32 + denorm, offset operand = uniform 0. Store's don't-care dst bits get the 2023 scratch register
     (num_temps) and use bit, set after encoding. Store sources live to the end, and a load's destination is kept
     out of its address registers (both as in the proven 2023 output).
   - As in 2023, for compute only: no inline immediates; ALU/branch encoded with the pack bit (".pack") except
     type conversions; load/store without it. Compiler option `has_cs_global_id`.
   - Temp-limit check counts the extra scratch register for compute.
   **Offline compiler** `npu-shader/cs_compile` (+ `build_cs_compile.sh`, `cs_compile_flags.txt`, kernels in
   `cs_kernels.h`, today's NIR API): compiles with the port compiler for GC8000 rev 0x7120 from Mesa's hwdb
   entry, **never opens a device** (checked with strace), so it needs no OK. `--fma N`, `--out F.bin`, `--dump`.
   Disassemble/diff: `research/port/shader_diff.sh A.bin [B.bin]`. Results in `runs/cs-offline/`:
   copy = 5 instructions vs 2023's 7: same words except that 2023's two dead `mov`s are gone and the load lands in
   t0.z (2023: t0.w); uniform layout identical (u0 = 4, in-arg, 0, out-arg). fma 16/256 = 21/261 instructions,
   num_temps 2. No 2023 fma binary exists (runs/fma-180621 dumps are empty).
4. DONE 2026-11-13 (boot d5fe88d6, Jordan's OK "You have the OK to use the NPU"): **the port tree runs compute on
   the shader core.** Ladder all PASS, 0 NPU kernel lines: CS-COMPILE, CS-COPY1/64/256, CS-FMA-SMALL, CS-FMA-BIG
   = 6.15 GFLOPS fp32 (2023 branch: 6.14), per-launch overhead 0.108 ms without dumps (2023: 0.165). Dumps in
   `runs/cs-193005/`. No change to the mesa-compute tree was needed in this step.
   - Device harness `npu-shader/cs_run` (source `cs_run.c`, build `build_cs_run.sh`, kernels from `cs_kernels.h`):
     same options as cl2023_copy plus `--offline`. Kernel input = constant buffer 0 user_buffer (out VA, in VA),
     buffers bound with set_global_binding. `--out DIR` must have an existing parent (it does a plain mkdir).
   - **Offline check before every NPU step:** `shim/libnpu_etna_shim.so` (source `shim/etnaviv_npu_shim.c`, build
     `shim/build_shim.sh`; Mesa's drm-shim core compiled outside the Mesa build) fakes the NPU (GC8000 rev 0x7120,
     DRM 1.4, the board's product/customer/eco IDs, so the driver uses the same hwdb entry), answers pipe 0 only,
     refuses perfmon queries (they would loop forever), and writes every submit as `shim_submit_NNNN.stream` +
     `.bos` (handles, flags, softpin VAs, sizes) into `$NPU_SHIM_DUMP_DIR`. Nothing executes. Run:
     `NPU_SHIM_DUMP_DIR=D ETNA_MESA_DEBUG=dump_shaders strace -f -E LD_PRELOAD=$PWD/shim/libnpu_etna_shim.so
     -e trace=openat,open,ioctl ./cs_run --offline ... --out D` (give LD_PRELOAD with `strace -E`, not to strace
     itself). cs_run refuses `--offline` without the shim and a device run with it (dlsym of
     `npu_etna_shim_active`); strace showed no /dev/dri open and no DRM ioctl. The hardware command streams of
     COPY1/64/256 were **identical** to the shim's, so a shim run predicts the submit exactly (also the buffer
     addresses: out 0xffffa000, in 0xffff9000, shader 0xffef8000 in both, and in 2023).
   - Versus 2023 (`runs/cl2023-180102`): streams differ only in PS_HALTI5_RANGE_HIGH / PS_ICACHE_COUNT (5 vs 7
     instructions: the two dead movs). The shader on hardware = cs_compile's offline output.
   - Timing note: ETNA_MESA_DEBUG=dump_shaders writes two files per launch; with it FMA-SMALL took 2.66 ms per
     launch, without it 0.108. Time only without dumps.
5. DONE 2026-11-13 ~19:59 (boot d5fe88d6, Jordan's OK "Run the ladder you have the OK"): **shared memory + barriers
   work on the shader core.** NPU ladder all PASS, 0 NPU kernel lines, no STOP: S5-IDS, S5-SELF, S5-REV4/16/64,
   S5-RED16/64 (RESULTS.md rows S5-*, dumps `runs/cs5-195609/`). Workgroups of 64 with barriers work (thread allocation 16,
   above the 2023 code's warning at 4). Hardware command streams byte-identical to the shim's again. 1D grids: the
   unused workgroup-ID components read 0. Details of the design and the offline checks follow (shared memory +
   barriers, needed for softmax/layernorm). Design = the 2023 branch's final code (mesa-cl2023: etnaviv_compiler_nir.c get_shared_slot /
   lower_shared_mem ~line 1280, barrier ~line 864, etnaviv_emit.c shared_bo ~line 297, thread_allocation ~line 401):
   - Shared memory is NOT the core's local storage (hwdb LocalStorageSize 0x10 is unused by the branch): it is a DDR
     buffer with one slice per workgroup. Slice = shared base + linear workgroup id x per-group size; load/store_shared
     become the proven load/store_global on that address. Port plan: an early NIR pass (compute only) builds the slice
     address once at the start of the shader from `load_shared_base_ptr` (-> new uniform ETNA_UNIFORM_SHARED_ADDR),
     `load_workgroup_id`, `load_num_workgroups` (-> ETNA_UNIFORM_GROUP_COUNT, already in the driver) and the per-group
     size baked in at compile time (ALIGN(shared_size, 16)); variable shared memory (grid info) refused with BUG().
   - Barrier: NIR `barrier` with execution scope >= workgroup -> one BARRIER instruction (words 0x2a,0,0,0 = Mesa's ISA
     test vector; the 2023 assembler emitted the same), memory-only barriers emit nothing (as 2023); sets
     `has_barriers`. Driver already sets CL_THREAD_ALLOCATION = ceil(workgroup size / 4) when has_barriers (2023 printed
     "not supported" above 4, i.e. workgroups over 16 threads with barriers: untested, ladder goes 4 -> 16 -> 64).
   - Driver: per-context shared buffer (grow only), referenced in the stream, uniform = its GPU VA; caps max_local_size.
   - Kernels to add to cs_kernels.h: `ids` (writes global/local/group IDs, addresses from the global ID only, proves the
     value order of 3 IDs before any shared address depends on the group ID), `reverse` (shared[lid] = in[gid];
     barrier; out[gid] = shared[wg-1-lid]), `reduce` (tree sum in shared memory, one barrier per level).
   - NPU ladder (each rung needs Jordan's yes, shim first): ids, reverse wg 4/16/64, reduce wg 16/64.
   **CPU part DONE 2026-11-13 ~20:05** (patch `research/port/step5-shared-barrier.diff`, 6 files, diff against the
   step-4 state = mesa-axi-sram + step2 + step3 patches; builds clean -j2, Teflon relinks, kit lib 41b08ede untouched):
   new pass `etna_lower_shared_to_global` (etnaviv_compiler_nir.c, compute only, right after nir_lower_io),
   `nir_intrinsic_barrier` -> BARRIER (pmode 1), uniforms `ETNA_UNIFORM_SHARED_ADDR` + GROUP_COUNT from
   load_shared_base_ptr / load_num_workgroups in emit_shader, `variant->shared_size` (disk-cached region);
   driver: `ctx->compute_shared` (grow-only pipe buffer), `etna_compute_uniforms_check` refuses a grid whose uniforms
   can't be filled (before, a missing kernel arg was written as 0 = an MMU fault), variable shared mem refused,
   caps.max_local_size 32 KiB. Harness: cs_kernels.h (ids, reverse, reverse-nobarrier, reduce; CS_SHARED_BYTES 2048),
   `cs_compile --kernel K [-w WG]`, `cs_run --kernel K [--self]` (kernel input 16 bytes: out, in, A, B).
   Offline results: copy/fma binaries + copy256 shim stream byte-identical to step 4; disassembly
   `runs/cs-offline/step5/*.disasm.txt` (barrier = 0000002a 00000000 00000000 00000000; reduce uses an inline -1
   immediate in its integer compare, inline immediates already ran on this core in SHADER-STORE1); shim runs
   `runs/cs-offline/step5-shim/` (run inside `bwrap --dev-bind / / --dev /dev` + strace: no /dev/dri, no DRM ioctl):
   value order 0x4 (group, global, local) in all new kernels, thread allocation 1 / 1 / 4 / 16 for no barrier / wg 4 /
   16 / 64, shared buffer = workgroups x 2 KiB.
6. IN PROGRESS (started 2026-11-13 ~20:10): Teflon + shader core. Survey (CPU only, read-only, mesa-compute):
   - Teflon (frontends/teflon/tfl_device.c): ops go through ml_dev->ml_operation_supported; supported TFLite ops =
     CONV_2D, DEPTHWISE_CONV_2D, MAX/AVERAGE_POOL_2D, ADD, CONCATENATION, SPLIT, PAD, FULLY_CONNECTED, RESHAPE, RELU,
     ABS, LOGISTIC, SUB, TRANSPOSE, STRIDED_SLICE, RESIZE_NEAREST_NEIGHBOR. SOFTMAX, MUL, MEAN, BATCH_MATMUL, etc. stay
     on the CPU. Each invoke creates a PIPE_CONTEXT_COMPUTE_ONLY context, runs ml_subgraph_invoke + read_output, and
     destroys the context: the same kind of context launch_grid runs on.
   - etnaviv_ml.c: a subgraph = list of etna_vip_instruction (ETNA_JOB_TYPE_NN / TP) emitted into ONE command stream
     (batched, close_batch flushes caches once); init_npu sets GL_API_MODE_OPENCL like etna_compute_reset_state.
     Tensors are pipe_resources (+offset), stored as uint8 (int8 inputs get +128 on upload).
   - Layout: NN jobs want channel planes (NCHW); Teflon adds an implicit TP TRANSPOSE (NHWC -> NCHW) in front of the
     graph and a TP DETRANSPOSE behind it (etnaviv_ml.c ~line 360 / 526, configs etnaviv_ml_tp.c
     create_transpose_config / create_detranspose_config: a plain permutation, out[c][h][w] = in[h][w][c],
     H = dims[1], W = dims[2]). These cost ~82 ms of a 99 ms single-layer invoke at 112x112x256 (~78 MB/s,
     npu-conv-throughput memory) and ~10 ms of the upscaler's 24 ms (planar copy: 14.7 ms).
   - So a shader-core op can be a new job type in the same list/stream, with tensors bound by GPU address.
   Jordan's choice 2026-11-13: "Transposes, then softmax" (go on to softmax/layernorm after the transposes without
   a new decision). Order: (a) shader-core transpose/detranspose kernels in the harness, measured
   against the TP numbers; (b) if faster, ETNA_JOB_TYPE_SHADER in etnaviv_ml replacing qualifying implicit
   transposes (opt-in env switch first); (c) then softmax/layernorm etc. for transformer-style graphs.
   6a CPU part DONE 2026-11-13 ~20:40 (no tree change; harness only): cs_kernels.h `ids2d` (2D grid IDs, addresses
   clamped with umin so wrong IDs can't leave the buffer), `t4x4` (4x4 byte-block transpose, one kernel for both
   directions via 6 stride args, C % 4 == 0 and H*W % 4 == 0, 57 instructions per 16 bytes), `tgather` (NHWC -> NCHW
   for any C, 4 word loads + shifts per output word; the upscaler input has C = 5). `cs_run --kernel t4x4|tgather
   --hwc H W C --dir in|out [-w BX --wy BY] [--repeat R]` checks byte-exact against a CPU reference and prints
   mb_per_s; `--kernel ids2d -n NX --ny NY -w BX --wy BY`. Offline compile + shim clean (runs/cs-offline/step6,
   step6-shim); 2D streams: CL_CONFIG dims 2, counts/sizes minus one, Z all ones (as 2023).
   Proposed NPU ladder (needs Jordan's yes): S6-IDS2D, S6-T4IN-256, S6-T4OUT-256 (112x112x256, the conv benchmark's
   tensors), S6-TGIN-5 (540x960x5 in), S6-T4OUT-4 (540x960x4 out), S6-TGIN-256; compare with the TP: ~41 ms each way
   at 112x112x256 (~78 MB/s).
   **6a NPU ladder DONE 2026-11-13 20:12-20:15 (Jordan's OK "You have the OK, run it"): all 6 PASS byte-exact, 0 NPU
   kernel lines** (RESULTS.md S6-*, runs/cs6-201216). Medians of 10 launches, no dumps:
   | kernel | tensor | ms | MB/s | TP for comparison |
   | t4x4 in | 112x112x256 | 20.04 | 160 | ~41 ms (78 MB/s) |
   | t4x4 out | 112x112x256 | 15.81 | 203 | ~41 ms |
   | tgather in | 540x960x5 | 33.39 | 78 | upscaler in+out ~9.5 ms together (~460 MB/s) |
   | t4x4 out | 540x960x4 | 9.49 | 218 | (same) |
   | tgather in | 112x112x256 | 50.02 | 64 | ~41 ms |
   So: 2.3x faster than the TP on wide tensors, slower on narrow ones, and slower than the planar wrapper's tiled CPU
   reorder (~13 ms for both directions at 112x112x256). t4x4 runs ~2.9 thread-instructions per clock vs ~1 for the
   vec4 fma test, so it is likely near an instruction-issue limit, not memory. Found: etnaviv sets
   `lower_bitops = true`, so NIR turns every shift by a constant into imul (nir_opt_algebraic.py:186).
   Probe variants written (CPU only, shim clean): `t4x4b` (butterfly shuffle, 48 instructions, 2 temps instead of 6;
   CPU-checked on 10000 random blocks), `t4x4copy` (same loads/stores, no shuffle = memory floor, 22 instructions),
   and workgroup shape 16x4 instead of 64x1. Next: probe rungs (need a yes), then decide what 6b integrates.
   **Probes DONE 20:20-20:21 (Jordan's OK): t4x4copy memory floor 13.41 ms; t4x4b in 14.66 / out 13.18 ms (in+out
   27.8 vs TP ~82); t4x4 with 16x4 workgroups 16.96 (64x1: 20.04).** Butterfly is near the floor of 4-byte accesses.
   **6b CPU part DONE ~21:00** (patch `research/port/step6-ml-shader.diff`, 8 files, against the step-5 state):
   - `etnaviv_ml_shader.c/.h` (new; added to meson.build and by ninja_add_source.py): ETNA_JOB_TYPE_SHADER. Opt-in
     `ETNA_ML_SHADER=transpose`; `etna_ml_shader_wants_transpose` picks implicit TP transposes/detransposes with
     C % 4 == 0, H*W % 4 == 0, C >= ETNA_ML_SHADER_MIN_CHANNELS (default 16), not in NPU_PARALLEL mode. The t4x4b
     kernel is built as NIR in the driver and compiled once per screen (etna_ml_device.transpose_kernel, freed in
     etna_screen_destroy). Compile falls back to the TP if a tensor is misaligned or in the AXI SRAM (tensors of
     non-NN/TP jobs are kept out of the SRAM anyway). Emission: flush caches (close_batch bits) + stall FE/PE,
     the grid, flush + stall again.
   - `etnaviv_compute.c`: screen-level `etna_compute_shader_create/get_variant/destroy` and
     `etna_compute_emit_grid(ctx, v, info, input, input_size, buffers, n, lazy_reset)` shared by launch_grid and the
     ML path (launch_grid's shim streams byte-identical to before for copy/reverse/reduce/t4x4b).
   - Offline Teflon on the shim works: run the Python test under `bwrap --dev-bind / / --dev /dev` with
     `strace -E LD_PRELOAD=<abs path>/shim/libnpu_etna_shim.so` and ETNA_ML_DISK_CACHE_DIR in scratch. conv3x1:
     port tree with the switch off = kit library 41b08ede streams byte-identical; switch on = SH transpose, NN
     conv, SH detranspose in one submit.
   - Checker `npu-shader/teflon_check.py -m M -r REF.npz -e LIB -n N [--save F.npy] [--compare F.npy]`: every
     invoke vs the first (race check), vs the CPU reference, and byte-for-byte vs a saved output.
   **6b NPU rungs DONE 20:35-20:36 (boot d5fe88d6, Jordan's OK): all PASS, 0 NPU kernel lines** (RESULTS.md S6B-*,
   runs/cs6b-203507): S6B-OFF (port-tree Teflon, switch unset) 103.35 ms median, within 1 level of the CPU;
   S6B-ON (ETNA_ML_SHADER=transpose) 47.59 ms, 10/10 outputs byte-identical to OFF; S6B-ON-50 46.34 ms, 50/50
   byte-identical. So NN jobs and shader-core grids share one submit correctly with the flush+stall fences, and
   conv3x1 runs 2.2x faster end to end. Not yet: other models (MobileNet's 224x224x3 input stays on the TP: C=3),
   ETNA_ML_SHADER_MIN_CHANNELS crossover vs the TP (unmeasured between C=5 and 256), 16x4 workgroups.
   NEXT (Jordan: "Transposes, then softmax"): softmax/layernorm on the shader core.
   6c softmax, IN PROGRESS (~20:45): target = MobileNet (kit model models/mobilenet_v1_mesa.tflite) whose graph ends
   CONV_2D -> SQUEEZE -> SOFTMAX over [1, 1001] uint8 (input scale 0.16609922 zp 66, output scale 1/256 zp 0); Teflon
   supports neither SQUEEZE nor SOFTMAX, so both run on the CPU after the NPU part. Plan: (1) float-ops probe kernel
   `fops` (exp2, exp via exp2*log2e, rcp, div = mul by rcp, fmax/fmin = select, f2u, floor; CPU-checked per result)
   - written, compiled offline (24 instructions), shim clean; NPU rung S6C-FOPS needs a yes; (2) softmax kernel in
   the harness: one workgroup per row, branch-free tree reductions in shared memory for max and sum (as `reduce`),
   exp, output = round(e / sum * 256) clamped, vs a CPU reference within 1 level; (3) Teflon: SQUEEZE as a reshape
   (bypass), new PIPE_ML_OPERATION_TYPE_SOFTMAX (gallium p_state.h, tfl_device.c fill_operation + supported check),
   etnaviv lowering to an ETNA_JOB_TYPE_SHADER job with the row length baked into a per-operation kernel; test on
   MobileNet with teflon_check.py (byte-compare vs the CPU softmax output, all invokes identical).
   **Status 2026-11-13 ~21:40 (Jordan away, blanket OK "I give you the OK for all steps"):**
   - S6C-FOPS PASS (float ops ~1 ulp), S6C-IDSGL PASS (value order 0x1), S6C-RED64 PASS (shared slot now clamps the
     workgroup index).
   - **S6C-SM1001 HUNG THE NPU at 21:09** (incident incidents/2026-11-13-2109-S6C-SM1001, STOP set, board up, no MMU
     fault, devcoredump root-only so not saved). Cause found offline: 16 temps x 64 threads with barriers; all
     threads of a barrier workgroup must be resident, and the register file can't hold them (64 x 6 proven fine).
     **NPU jobs are refused until a reboot; Jordan clears runs/STOP after review.**
   - CPU fixes since (patch research/port/step6c-softmax.diff, 9 files incl. p_state.h + tfl_device.c): softmax v4
     = 9 temps (see the incident NOTES); register guard `etna_compute_barrier_reg_budget()` (default 384, env
     ETNA_CS_BARRIER_REG_BUDGET); probe kernels `cs_run --kernel regpK`; Teflon SOFTMAX support:
     PIPE_ML_OPERATION_TYPE_SOFTMAX (beta) + reshape.squeeze in p_state.h, tfl_device.c maps SOFTMAX and SQUEEZE,
     etnaviv claims SQUEEZE (as a bypass) and SOFTMAX only with ETNA_ML_SHADER=softmax and a kernel inside the
     budget, lowered to a per-operation shader job (owned_shader, freed with the subgraph), rows [N,1,1,C] 8-bit with
     output scale 1/256. The formula equals TFLite's uint8 softmax exactly on MobileNet's CPU logits (1001/1001).
     Offline (shim): port tree with the switch off = kit library streams byte-identical for conv3x1 and MobileNet;
     ETNA_ML_SHADER=softmax ETNA_CS_BARRIER_REG_BUDGET=576 puts the softmax grid into MobileNet's tail NN submit.
     Reference npu-shader/models/mobilenet_v1_mesa_reference.npz (grace_hopper input, CPU output 653 @ 180/256).
   - **Update 21:50 (boot b8c53cd6, after Jordan's reboot): the ladder's REGP4 (64 threads x 7 temps = 448) PASSED,
     REGP6 (64 x 9 = 576) HUNG the NPU** (incident incidents/2026-11-13-2143-S6C-REGP6, STOP set again). So a barrier
     workgroup's limit is between 448 and 576 registers (512 untested). Fixes (CPU only): default budget 448; the
     softmax builder takes a thread count and the driver picks 64/32/16 to fit -> MobileNet's row uses 32 threads
     (605 instructions, 11 temps, 352 registers; Teflon picks it offline). The ladder script is revised (no 576
     probe): SM1001-W32, SM4x1000-W32, MN-OFF, MN-SM, MN-SM50; `--map` adds REGP5 (512). Needs another reboot.
   - **6c DONE 2026-11-13 22:12 (boot 8f7f01ab): ladder --map all PASS, 0 NPU kernel lines** (RESULTS.md S6C-*,
     runs/cs6c-ladder-20261113-221056): softmax kernel 32 threads 1x1001 and 4x1000 exact; MobileNet with
     ETNA_ML_SHADER=softmax 7.58 ms (off 7.63), 10/10 and 50/50 outputs byte-identical to off: SQUEEZE + SOFTMAX now
     run on the NPU; REGP5 (64 x 8 = 512) PASS -> limit between 512 and 576, **default budget raised to 512**.
   - NEXT candidates: global AVGPOOL on the shader core (puts MobileNet in one submit; see below), layernorm-style
     reductions for transformer graphs, the 16x4 workgroup shape for transposes, ETNA_ML_SHADER_MIN_CHANNELS
     crossover measurement.
   - (older) **NEXT after the reboot: `research/port/step6c-ladder.sh`** (REGP4 448, REGP6 576, SM1001, SM4x1000,
     MobileNet OFF / softmax / softmax x50; `--map` adds REGP8/10/12, which may hang). If REGP6 passes, raise the
     default budget to the largest passing size.
   - Found offline (TEFLON_DEBUG=verbose on the shim): MobileNet runs as two NPU partitions because AVGPOOL (node 27,
     global 7x7x1024 -> 1x1x1024) is unsupported in etnaviv; partition 2 = the last CONV (+ SQUEEZE + SOFTMAX with the
     switch). Candidate next shader op (not started, beyond Jordan's "transposes, then softmax"): global average pool
     on the shader core, no barriers needed: input is the NN output in channel planes (NCHW), 4 consecutive planes
     are 4*HW contiguous bytes, so one thread per 4 channels loads HW words and sums bytes at compile-time positions;
     out = round(((sum / HW) - in_zp) * in_scale / out_scale + out_zp). Would put all of MobileNet in one submit.
   - Verified after the 6c changes: the 6b transpose path's offline streams are byte-identical to the proven S6B-ON.
   Older notes on the remaining open questions:
   which ops/models need it (Teflon today runs int8 NN/TP only, so a float op on the shader core needs
   dequantize/requantize around it, or an int8 kernel), how Teflon hands a tensor's buffer to a compute grid in the
   same context (both are compute-only contexts on the same screen), and whether the grid can go into the same
   submit as the NN jobs. Kernels need loops (control flow is untested on hardware: the step 5 kernels are
   branch-free) and float ops beyond fma (exp, rcp: transcendental encodings untested). Each new instruction class
   gets an offline shim check, then its own NPU rung.

## Caution

Replies that went deep into individual instruction bit fields were repeatedly cut off by a safety filter in the
first session (false positives on legitimate driver work). Prefer the branch's compiler and Mesa's encoder/decoder
over hand-assembled words, and keep write-ups at the level of findings.

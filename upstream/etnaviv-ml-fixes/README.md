# Sending the etnaviv ML (Teflon) fixes to Mesa

Twelve patches against Mesa `main` (commit 8a9009e0482, 2026-11-14) for the etnaviv NPU driver, found while
porting compute to the NPU's shader core and building an LLM engine on the Alta (Amlogic A311D, VIPNano-QI.7120).
Patches 1-9 fix bugs, patches 10-12 make moving the int8 data in and out faster; all twelve problems are
in upstream `main` today. **Nothing has been sent.** You review, decide and send.

| Patch | What it fixes | Seen on the board as |
|---|---|---|
| 0001 | Graph outputs with one row or one column (`[1,1,W,C]`, `[1,H,1,C]`) were never converted back from channel planes | conv on `[1,1,64,384]` about 100 levels off the CPU |
| 0002 | V7 tiling used a tile wider than the image when H > W and W < 64 | NPU hang on every run of `[1,64,1,384]`; `[1,64,8,64]` has the same wrong tile but was only run after the fix |
| 0003 | V7 coefficient writers read kernels past the end of the weight and bias tensors when the channels don't divide evenly between the NN cores | out-of-bounds read |
| 0004 | V7 wrote int8 weights unshifted next to a zero point shifted by 128 | every int8 convolution wrong (identity conv gave all 127) |
| 0005 | V7 addition bias used the raw int8 zero points | int8 ADD all -128 |
| 0006 | V7 addition bias correction summed 256 bytes of an 8-byte weight tensor | ADD bias changed from run to run, uint8 too |
| 0007 | Addition of an NHWC operand and an NCHW operand, output not converted back | uint8 ADD up to 154 levels off |
| 0008 | V7 accepted SUBTRACT but can only lower ADD | assertion (abort) in `etna_ml_lower_add_v7`; found by Mesa's own `test_teflon` (6 tests) |
| 0009 | The tensor of an addition's first operand got the size of both operands | input upload reads past the caller's buffer and can overwrite the second operand: `micronet_vww.whole` wrong (found by AddressSanitizer in `test_teflon`) |
| 0010 | Signed (int8) outputs were converted byte by byte straight from the write-combined tensor buffer: one uncached load per byte | reading back a 786 KB int8 output took 22.5 ms (2.5 ms with the patch); a one-token LLM step through Teflon 0.85 -> 0.63 ms, a 256-token one 29.5 -> 7.2 ms |
| 0011 | Graph outputs live in write-combined memory, so even one bulk copy reads them slowly | reading back a 786 KB int8 output 2.6 ms (1.5 ms with the patch); a 256-token LLM step through Teflon 7.19 -> 6.29 ms; small outputs unchanged |
| 0012 | Signed tensors were converted (+-128) one byte at a time, in a loop GCC does not vectorize | reading back the 786 KB output 1.5 -> 0.54 ms, writing a 147 KB input 0.26 -> 0.10 ms; a 256-token LLM step through Teflon 6.28 -> 4.04 ms |

## What was tested, and what was not
- **Patch 12 (added 2026-11-20):** branch tip 940b610ac79, library 623a3f1e (kept as
  `libteflon.so.623a3f1e-12patches`). flip_sign_bits() flips the top bit of each byte eight bytes at a time; cached
  outputs convert straight from the buffer, write-combined ones after the bulk copy, signed inputs on the way in;
  etna_buffer_cpu_read_cached() (etnaviv_resource.h) is shared by the allocator and the read-out. Found with a perf
  profile of the read-out (`runs/llm-readout-profile-20261120-034535`: the byte loop was ~0.9 of 1.56 ms). The four
  changed files are byte-identical to the `mesa-llm` tree that was tested on the NPU (library fa818da9):
  `runs/llm-fused-ladder-20261120-035655`, 14 rungs, patch 11 vs patch 12 back to back, all passed, 3600 invokes
  with alternating inputs checked, none stale, no kernel warning:
  | job | read-out 11 -> 12 | input write 11 -> 12 | whole step 11 -> 12 |
  |---|---|---|---|
  | one token, 3 KB out, Cortex-A53 | 0.017 -> 0.011 ms | 0.007 -> 0.005 ms | 0.438 -> 0.390 ms |
  | 64 tokens, 196 KB out | 0.394 -> 0.133 ms | 0.069 -> 0.027 ms | 1.385 -> 1.063 ms |
  | 256 tokens, 786 KB out | 1.549 -> 0.538 ms | 0.259 -> 0.099 ms | 4.649 -> 3.481 ms |
  | Teflon whole invoke, 1 token / 256 tokens | | | 0.64 -> 0.62 / 6.28 -> 4.04 ms |
  Offline: NN configs, weights, the input buffer bytes, streams and buffer lists unchanged; a unit test
  (`llm/l3/bench/flip_test.c`) checks the helper against the old loops on 17 sizes and 3 alignments.
  `validate_offline.py` passes with the 12-patch library (34 of 34; the 11-patch results are in
  `validation/suite-11patches-6a3290ab`). `test_teflon` on the NPU with
  all twelve patches (Jordan's OK, `runs/teflon-suite-20261120-040956`, lib 623a3f1e): 1503 passed, 34 failed, the
  same 34 tests as with nine to eleven patches, no hang, no kernel warning.
- **Patch 11 (added 2026-11-20):** branch tip 470cdf1729d, library 6a3290ab (kept as
  `libteflon.so.6a3290ab-11patches`; the 10-patch library 9574fdd1 is kept as
  `~/Documents/AI_Projects/npu-llm/bin/libteflon-9574fdd1.so`). Graph outputs (and a buffer they alias through a
  bypass or split) are allocated cached on arm64 through `ETNA_RESOURCE_FLAG_CPU_READ`, which buffers now honour as
  textures already did; the zero fill at creation is wrapped in CPU_PREP / CPU_FINI. Tested on the NPU in `mesa-llm`
  before its lines were wrapped to 78 columns (library 6caca26e): `runs/llm-cached-ladder-20261120-031603`, 14 rungs,
  write-combined (the 10-patch code) vs cached back to back, all passed:
  | job | read-out write-combined -> cached |
  |---|---|
  | one token, 3 KB out, Cortex-A53 | 0.032 -> 0.022 ms |
  | 4 jobs, 6.6 KB out, Cortex-A53 | 0.069 -> 0.054 ms |
  | 64 tokens, 196 KB out | 0.58 -> 0.34 ms |
  | 256 tokens, 786 KB out | 2.61 -> 1.48 ms |
  | Teflon whole invoke, 1 token / 256 tokens | 0.66 -> 0.67 / 7.19 -> 6.29 ms |
  Stale-read check: two inputs with the same weights alternate (96-98% of the output bytes differ between them)
  and every output is compared with its reference: 2800 invokes, none stale, no kernel warning. With a buffer log
  in the fake NPU: exactly the graph outputs are cached, and no CPU_FINI comes without a CPU_PREP. The wrapped
  build compiles the same jobs, buffers and prep / fini sequence offline
  (`npu-llm/runs/offline-jb-20261120-cached-wrapped`). `validate_offline.py` passes with the 11-patch library (34 of
  34; the 10-patch results are in `validation/suite-10patches-9574fdd1`). `test_teflon` on the NPU
  with all eleven patches (Jordan's OK, `runs/teflon-suite-20261120-033148`, lib 6a3290ab): 1503 passed, 34 failed,
  the same 34 tests as with nine and ten patches, no hang, no kernel warning.
- **Patch 10 (added 2026-11-20):** branch tip 328286da6e0, library 9574fdd1 (9-patch library kept as
  `libteflon.so.b6c2a28d-9patches`). Tested on the NPU in the LLM test tree `mesa-llm` (this branch's nine patches
  + test-only tiling hooks that do nothing unless their environment variables are set + this change, library
  a26f0852; the patch's code is identical, only its comment is rewrapped to 78 columns here):
  `runs/llm-readout-ladder-20261120-021528`, 12 rungs, old vs new library and binary back to back, outputs
  identical (within 1 level of the CPU reference, the same as before):
  | job | read-out old -> new | whole step old -> new |
  |---|---|---|
  | 1x1 conv 576 -> 3072, one token (3 KB out), Cortex-A53 | 0.360 -> 0.032 ms | 0.743 -> 0.443 ms |
  | 4 jobs 2048 -> 1664 (6.6 KB out), Cortex-A53 | 0.817 -> 0.069 ms | 3.455 -> 2.709 ms |
  | 64 tokens (196 KB out) | 5.57 -> 0.58 ms | 6.53 -> 1.50 ms |
  | 256 tokens (786 KB out) | 22.5 -> 2.5 ms | 25.6 -> 5.6 ms |
  | Teflon (`teflon_check.py`), whole invoke, 1 token / 256 tokens | | 0.85 -> 0.63 / 29.48 -> 7.21 ms |
  The NPU's own time is unchanged; the submits, NN configurations and weights are byte-identical offline
  (`npu-llm/runs/offline-jb-20261120-readout`). `validate_offline.py` passes with the 10-patch library (34 of 34,
  `validation/suite`; the 9-patch results are kept in `validation/suite-9patches-b6c2a28d`). Mesa's
  `test_teflon` on the NPU with all ten patches (Jordan's OK, 2026-11-20, `runs/teflon-suite-20261120-025106`, lib
  9574fdd1): 1503 passed, 34 failed, exactly the same 34 tests as with nine patches, no hang, no kernel message.
  Even the bulk copy reads the uncached buffer at only ~0.3 GB/s; cached output buffers would be a further step
  (not in this series).
- **On the NPU:** the same code changes in the port tree (`mesa-compute`, based on Mesa main of 2026-04-10).
  Ladders: `runs/cs6f-ladder-20261114-012542` (0001), `runs/cs6h-ladder-20261114-015507` (0002),
  `runs/cs6i-ladder-20261114-150119` (0004 to 0007). 0003 is in the kit library (41b08ede).
- **Upstream `main` plus these patches:** builds without new warnings (`mesa-upstream-fixes/build`, library
  549408ab). Run against the fake NPU, it compiles 20 models (the int8 models and their uint8 twins, the layout
  models, MobileNetV1, conv3x1) to exactly the configurations, coefficient buffers and command streams of the
  port build that passed on the NPU (`runs/cs-offline/step6i-upstream`).
- **Upstream build on the NPU (2026-11-15, library 549408ab):** `runs/cs6i-upstream-ladder-20261115-033157`
  (14 rungs, patches 3 to 7) and `runs/cs-layout-upstream-ladder-20261115-033648` (7 rungs, patches 1 and 2), all
  passed with no kernel messages. Every output is byte-identical to what the port build produced on the NPU.
- **Address sanitizer (2026-11-15, fake NPU, no hardware):** a sanitizer build of the branch
  (`~/Documents/AI_Projects/mesa-upstream-asan`) reports both out-of-bounds reads when the fix is left out and
  nothing with all seven patches (`runs/cs-offline/patch3-asan`):
  - without patch 3, MobileNetV1: heap-buffer-overflow READ in `write_core_6` (`etnaviv_ml_nn_v7.c:245`), 0 bytes
    after the 1025024-byte weight tensor of the 1001-channel convolution;
  - without patch 6, `add_u8`: heap-buffer-overflow READ in `calculate_bias_correction`
    (`etnaviv_ml_nn_v7.c:126`), 0 bytes after the 8-byte addition weights;
  - with all seven: MobileNetV1 and six of the test models compile with no report.
  You can quote these in the messages of patches 3 and 6 when you rewrite them.
- **Patch 8 (added 2026-11-15):** branch tip 5b20ad12973, library a05298e9. Under the fake NPU the six
  `test_teflon` cases that aborted (movenet `0095`, `0136`, `whole`) no longer abort, and 19 of the proven models
  compile to byte-identical streams as with the seven-patch library 549408ab (kept beside the new one as
  `libteflon.so.549408ab-7patches`). The two ladders above ran with the seven-patch library.
- **Patch 9 (added 2026-11-15):** branch tip a08a8763308, library b6c2a28d. Found by the sanitizer in
  `test_teflon` (`micronet_vww.whole`: heap-buffer-overflow READ in `etna_ml_subgraph_invoke`); with the patch the
  sanitizer is silent and the test passes on the NPU (it failed with 8 patches and hung the NPU with unpatched
  main). `validate_offline.py` passes with the 9-patch library.
  Sanitizer pass over all 1568 `test_teflon` cases with the 9-patch tree under the fake NPU
  (`runs/cs-offline/teflon-suite-asan-9patches`, 2026-11-15): no report in the driver and no abort; the 40
  remaining reports are all in the test program (`test_teflon.cpp`, 32-bit inputs). Command streams are identical
  to the 7-patch run except for the 12 movenet cases with a SUBTRACT (patch 8). The same change also stops the
  output read-back from copying both operands' size when that tensor is a graph output (by reading the code;
  no test hits it).
- **Mesa's own Teflon test suite (`test_teflon`) on the NPU, 2026-11-15, main 8a9009e0482 + 9 patches:**
  1503 tests passed, 34 failed, no hang (`evidence/teflon-suite-20261115/README.md` has the table). A second full run
  in another boot (b43143e4, 18:02, `runs/teflon-suite-20261115-180223`) gave the same 1503 / 34 with the same
  failed tests and no kernel message. Left out:
  Mesa's skip list for this NPU (32 tests) and the two ds_cnn tests with an int8 FULLY_CONNECTED, which hung the
  NPU in the first run (`incidents/2026-11-15-0745-TFS-ds_cnn`). Unpatched main hangs on `ds_cnn.0014` as well
  (`incidents/2026-11-15-1502-FCB-ds_cnn-0014`, with the hang dump), so that hang is not from the series: a
  FULLY_CONNECTED hangs this NPU in upstream. Not fixed here.
  - 8 of the 34 failures are in Mesa's expected-failure list (`src/etnaviv/ci/etnaviv-vipnano-fails.txt`).
  - The other 26 were rerun with unpatched main (lib 385f828f): all 26 fail there too (the two movenet whole
    models abort there). None of the failures comes from the series.
  - Unpatched main HUNG the NPU in `micronet_vww.whole` (`incidents/2026-11-15-1438-TFB-micronet_vww`, with the
    kernel's hang dump); the series does not.
  - Seen in the failures, not fixed by this series: stride-2 depthwise convolutions on odd input sizes
    (ssd_mobilenet_v2 `0011` `0052`, same kind as Mesa's expected posenet failures); 1x1 convolutions with 960
    input channels on 33x33 (deeplabv3 `0052` `0056` `0060`: wrong from row 12 on).

## Before you send
1. **Read all twelve patches.** Each message says what was wrong, why, and what was tested. You will be the one
   answering review questions.
2. **Mesa's rules on AI** (`docs/submittingpatches.rst` and `AGENTS.md` at 8a9009e0482, read 2026-11-15):
   - Commits must carry `Generated-by: LLM`; no `Co-authored-by` for tools, no model name. The commits now
     follow this (trailer changed 2026-11-15; code unchanged).
   - **Commit messages, code comments and GitLab text must be your own words**, not AI-written. The commit
     messages, the code comments the patches add (the rewrite-list page lists each) and `MR-DESCRIPTION.md` were written by Claude. Treat them
     as notes and rewrite them yourself (`git rebase -i 8a9009e0482`, reword each) before sending.
   - You must be able to explain and justify every change yourself.
   - AI agents may not open merge requests or comment upstream; you send it.
   Mesa does not require `Signed-off-by`, but you may add it.
3. **Author line.** The commits use your git identity (your real name and Gmail address), the same one this
   public repo's commits already carry. Change it if you'd rather send under another name.
4. Rebase onto current `main` just before pushing (the base is from 2026-11-14) and rebuild once.

## How to send (Mesa uses GitLab merge requests, not email)
1. Get an account on https://gitlab.freedesktop.org and fork `mesa/mesa`. New accounts may need to ask for fork
   permission first; the site explains how when you try.
2. Push the branch from the worktree to your fork:
   ```
   cd ~/Documents/AI_Projects/mesa-upstream-fixes
   git remote add mine git@gitlab.freedesktop.org:<your-user>/mesa.git
   git push mine alta-etnaviv-ml-fixes
   ```
3. Open a merge request against `mesa/mesa` `main`, title `etnaviv/ml: V7 int8, addition and layout fixes, faster output read-back`,
   with the text of `MR-DESCRIPTION.md`. The etnaviv ML code's author is Tomeu Vizoso; GitLab will suggest
   reviewers.

## Files
- `0001` to `0012`: the patches (`git am` applies them to main 8a9009e0482). Regenerated 2026-11-20 as a
  twelve-patch series (patches 1-11 unchanged apart from the "n/12" numbering); the earlier exports are kept in
  `previous/` (9, 10 and 11 patches).
- `MR-DESCRIPTION.md`: text for the merge request.
- `make_series.py`: how the first seven commits were made (patches 8 and 9 were committed by hand on top; do not rerun it) (edits plus messages), for regenerating on a newer `main`.
- `models/` (18 small models with CPU references, `manifest.json` has the hashes), `reproduce.py` (checks a model
  against its reference on the CPU, or on hardware with `--delegate`; on this board only through
  `scripts/npu-attempt` with an OK) and `validate_offline.py`: merged 2026-11-15 from the second package that
  another session made on 2026-11-14.
- `evidence/`: that session's records. Offline comparison of 30 archived plus 4 fresh port runs against the
  upstream build, all 34 match; CPU check of the 18 models, all exact; hardware results of the port ladder as
  JSON; the upstream configure and build logs. They describe commit 5384f808, which has the same source tree as
  the branch now.
- The second package itself (`~/Documents/AI_Projects/upstream/etnaviv-ml-fixes`) is superseded by this folder.
  Only its raw offline dumps (`validation/suite/`, 226 MB) were not copied.
- The branch `alta-etnaviv-ml-fixes` (tip 940b610ac79) lives in the worktree `~/Documents/AI_Projects/mesa-upstream-fixes`
  (a worktree of `~/Documents/AI_Projects/mesa`, kept outside this repo).

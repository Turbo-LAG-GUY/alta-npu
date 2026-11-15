# Sending the etnaviv ML (Teflon) fixes to Mesa

Seven patches against Mesa `main` (commit 8a9009e0482, 2026-11-14) for the etnaviv NPU driver, found while
porting compute to the NPU's shader core on the Alta (Amlogic A311D, VIPNano-QI.7120). All seven bugs are in
upstream `main` today. **Nothing has been sent.** You review, decide and send.

| Patch | What it fixes | Seen on the board as |
|---|---|---|
| 0001 | Graph outputs with one row or one column (`[1,1,W,C]`, `[1,H,1,C]`) were never converted back from channel planes | conv on `[1,1,64,384]` about 100 levels off the CPU |
| 0002 | V7 tiling used a tile wider than the image when H > W and W < 64 | NPU hang on every run of `[1,64,1,384]`; `[1,64,8,64]` has the same wrong tile but was only run after the fix |
| 0003 | V7 coefficient writers read kernels past the end of the weight and bias tensors when the channels don't divide evenly between the NN cores | out-of-bounds read |
| 0004 | V7 wrote int8 weights unshifted next to a zero point shifted by 128 | every int8 convolution wrong (identity conv gave all 127) |
| 0005 | V7 addition bias used the raw int8 zero points | int8 ADD all -128 |
| 0006 | V7 addition bias correction summed 256 bytes of an 8-byte weight tensor | ADD bias changed from run to run, uint8 too |
| 0007 | Addition of an NHWC operand and an NCHW operand, output not converted back | uint8 ADD up to 154 levels off |

## What was tested, and what was not
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
- **Not done:** Mesa's own Teflon test suite has not been run (it needs gtest, a C++ TensorFlow Lite library and
  a downloaded model zoo, none of which are on the board).
  Nothing was tested on V8 hardware; the patches touch V7 code and the shared layout pass (0001, 0007).
- **Not included:** per-channel int8 convolutions are still refused by the driver.

## Before you send
1. **Read all seven patches.** Each message says what was wrong, why, and what was tested. You will be the one
   answering review questions.
2. **Mesa's rules on AI** (`docs/submittingpatches.rst` and `AGENTS.md` at 8a9009e0482, read 2026-11-15):
   - Commits must carry `Generated-by: LLM`; no `Co-authored-by` for tools, no model name. The commits now
     follow this (trailer changed 2026-11-15; code unchanged).
   - **Commit messages, code comments and GitLab text must be your own words**, not AI-written. The commit
     messages, the two code comments the patches add and `MR-DESCRIPTION.md` were written by Claude. Treat them
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
3. Open a merge request against `mesa/mesa` `main`, title `etnaviv/ml: V7 int8, addition and layout fixes`,
   with the text of `MR-DESCRIPTION.md`. The etnaviv ML code's author is Tomeu Vizoso; GitLab will suggest
   reviewers.

## Files
- `0001` to `0007`: the patches (`git am` applies them to main 8a9009e0482).
- `MR-DESCRIPTION.md`: text for the merge request.
- `make_series.py`: how the commits were made (edits plus messages), for regenerating on a newer `main`.
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
- The branch `alta-etnaviv-ml-fixes` (tip f3ac5799d2d) lives in the worktree `~/Documents/AI_Projects/mesa-upstream-fixes`
  (a worktree of `~/Documents/AI_Projects/mesa`, kept outside this repo).

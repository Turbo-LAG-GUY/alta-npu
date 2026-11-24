# Incident 2026-11-23 15:37: prefill2 ladder, the 1.7B jb_prefill run stalled the board in swap (no NPU hang)

Ladder: llm/l3/prefill2-ladder.sh (Jordan's OK), boot b887d66c, driver f317fc8f, jb_prefill 6de63573. Two runs.
Sources: Jordan's desktop-session report (jordan-desktop-session-report.md, his original, kept as written), PROGRESS.md,
the npu-attempt captures (run1-R1-observe.txt / -kmsg.txt) and run 2's npu-run record (run2-R1-*.txt).

## Run 1: stalled 1 h 47 min, stopped by Jordan (runs/llm-prefill2-ladder-20261123-153222)
- 15:32 start. Q1, Q2, Q3 (135M, last token) passed, rc 0, no kernel warnings, ~1.5 min each.
- 15:37:37 R1 (the first whole 1.7B chunk, --all) started, then **no output at all for 1 h 47 min**.
- npu-observe (1 s samples, run1-R1-observe.txt): MemAvailable 1076 MB at R1's start -> 890 MB after 70 s -> 239 MB at
  15:50; `npu_irqs` stayed at 10723 the whole time: **no NPU job was ever submitted** (jb_prefill was still loading and
  compiling the 96 packs). The observer itself got 117 of its 780 samples: starved like everything else. npu-run never
  even wrote R1's record directory, and its 600 s deadline never fired (its own loop was blocked).
- Jordan, 17:20: load ~33, 193 MB of 3.7 GB available, 4 GB of swap in use, 2.2 GB "shared" (the job's NPU / DRM
  buffers: /dev/shm and the tmpfs mounts were empty); **25 tasks in D state waiting on block I/O** (kswapd0, Xorg, both
  gnome-shells, every claude process, the jb_prefill worker) = swap thrashing, not an NPU hang. systemd: 8 min 53 s CPU,
  1.5 GB memory peak, 407.8 MB swap peak.
- Jordan (approved by him): 17:24:21 `systemctl --user stop alta-llm-prefill2` (clean, 17:24:52); SIGTERM to three
  duplicate `claude -r` processes idle for ~17 h (186020, 186099, 186166). After: 1.2 GB available, shared 708 MB, swap
  down ~1 GB. No STOP latch from run 1 (stopped from outside, nothing in the kernel log).

## Run 2: npu-run's deadline (runs/llm-prefill2-ladder-20261123-172638)
- 17:26:38 restart (from this LLM session). Q1-Q3 passed again (Q1: 120 steps compiled in 6989 ms, 787.3 ms per chunk).
- 17:32:08 R1 started; 17:43:36 **npu-run's 600 s deadline -> rc 124 -> STOP latch** (runs/STOP 17:43:43). jb_prefill's
  stdout was empty: it never finished load + compile this time either. Kernel log (run2-R1-kernel.txt): only the Mali
  driver's shrinker ("panfrost_gem_shrinker_scan ... Purging") = memory pressure; no etnaviv line, hw_job_trips 0, no
  devcoredump, the process gone after the kill.

## Cause
The 1.7B prompt chunk in jb_prefill keeps every compiled step on the NPU: ~1.8 GB of GEM buffers, pinned once the NPU
maps them, plus the process (~0.3 GB) and Mesa's transient compile memory. The board has 3.7 GB, and at the start had
~1.1 GB available: two GNOME sessions (the desk + Jordan's xrdp), Xorg, and six Claude sessions (~1.3 GB) were resident.
The kernel pushed everything else to swap on the eMMC until every process waited on block I/O. A fake-NPU run of the
same load (no pinning; npu-llm/runs/offline-prefill-1p7b) finished load + compile in 108 s and still pushed ~1 GB to
swap. The NPU driver was not involved: no NPU job was submitted in either R1.

## Lessons
- **A swap stall is silence, not a failure.** npu-run's deadline only works while its own shell can run; under thrashing
  it didn't fire for 1 h 47 min. Check `free -m` and the D-state count (`ps -eo stat | grep -c '^D'`) before blaming
  the NPU driver.
- **Memory per run** (jb_prefill, 256-token chunk): 135M ~0.4 GB total (NPU buffers ~0.17 GB); 1.7B ~1.8 GB of NPU
  buffers + ~0.3 GB process + compile transients -> needs >= ~2.3 GB available before it starts.
- Run the 1.7B rungs without a desktop session logged in (or close one of the two GNOME sessions), and don't keep extra
  `claude -r` resumes open on the board.
- Have the ladders refuse to start a 1.7B rung below the memory it needs, and have jb_prefill print (and flush) its
  load progress, so a stall is visible.

## Follow-up
- prefill2-ladder.sh: a MemAvailable guard before each 1.7B rung (and the 1500 s deadline + SKIP_135M already added).
- jb_prefill: load / compile progress lines (flushed) every 8 steps.
- Proposal for Jordan (kit script, not changed): npu-run could also fail a job whose stdout and the kernel log stay
  silent for N minutes, and check MemAvailable / D-state tasks in its loop.
- runs/STOP from run 2: left for Jordan's review.

## Resolution (2026-11-24)
- **The memory figure above was an estimate, and too low.** Measured from the fake-NPU run's BO lists (Mesa's
  calculate_weight_bo_size reproduces every recorded coefficient BO size): the 1.7B steps pinned **2127 MiB**, of which
  299 MiB was libdrm-etnaviv's BO cache rounding each BO up to the next quarter power of two (a gateup job's 8.03 MiB
  of weights in a 10 MiB BO), plus ~192 MiB of per-pack read-back buffers in the process: **~2.5 GB needed**, so the
  2300 MB guard (follow-up above) would not have been safe either. After the reboot (boot cbbc9ac1) a watcher waited an
  hour for 2300 MB (peak 2259 MB with the gdm greeter + Jordan's xrdp session) and never started the run.
- Fixes: jb_prefill 36703f9e (one set of read-back buffers per step kind, -184 MiB) and the mesa-llm test-tree hook
  ETNA_ML_EXACT_BO=1 (exact-size coefficient BOs, -299 MiB pinned: **1828 MiB**).
- llm/l3/prefill3-ladder.sh (Jordan's OK, memory guard off at his request; runs/llm-prefill3-ladder-20261124-021642):
  **5/5 PASS**, trips 0, no kernel warnings. 135M bucket vs exact BOs: logits bit-identical; the 1.7B chunk completed
  three times (once starting at 1865 MB available, down to 404 MB at the end of the load: slower, no stall), quality =
  the --ref / simulator (KL .154 vs fp32), 3.66 s per 256-token chunk. Details: llm/DESIGN.md "Resume here (2026-11-24
  02:30)".

# hw_job_limit: from a boot-time constant to a runtime / automatic setting

2026-11-21. Status: **patches 0006 + 0007 built offline, app written and tested on a fake /sys, root step 250
written. Nothing installed, nothing run on the NPU.**

- Patches: `~/alta-npu/patches/etnaviv-job-limit/0006-drm-etnaviv-alta-adjustable-hw-job-limit.patch` (8041959938ce),
  `0007-drm-etnaviv-alta-program-priority-by-name.patch` (307eb9ba5782).
- Module to install: `~/alta-npu/build/etnaviv-6.12.73-g6b2d05eda916-20261121-163831-job-prio/etnaviv.ko`
  (sha256 f317fc8f4d6e..., 0001-0007; gcc-14 against the stock headers like 140; no warnings; #0x7120 compare and
  the reset / AXI SRAM markers present; vermagic matches; every import exported). The 0006-only build
  (…-161304-job-limit, 66d8d55b) is kept as an intermediate.
- Install: `root-steps/250-install-etnaviv-job-queue.sh` (not run). Revert: same script with `--revert`.
- App: `~/Documents/AI_Projects/npu-status/` (live; copies in `~/alta-npu/indicator/`): `npu_queue.py` (logic),
  `npu_control.py` (window), `npu_status.py` (indicator: "Job queue…" menu item, menu row, trip warning,
  saved settings at login). Tests: `tests/` (23 pass, fake /sys, window drawn off-screen). Screenshots:
  `screenshots/` (fake driver states).
- The running Image is the night-light build (~/nightlight-fix, GCC 13, same release string and stock config), so
  build-etnaviv-oot.sh now takes the compiler from the headers' include/config/auto.conf, not /proc/version.

Jordan's decisions (2026-11-21): hw_job_auto on by default; a trip must warn the user and offer reset or reboot;
rule D (turns) only when asked for (programs share the ring by default, as in a graphics driver); program priority
by name with a little multitasking; a GUI rather than terminal only.

## What the setting does on this chip

- `hw_job_limit` (etnaviv_sched.c) is the drm scheduler's credit limit: how many jobs may be in the NPU's
  hardware ring at once. Stock default 4; ours is 1 since root step 080 (2026-11-07 freeze hunt; the freezes
  went on with 1 and ended with the reset build, so the original reason is gone).
- It does **not** run jobs in parallel. The front end runs the ring in order. A limit above 1 only lets the next
  job already wait in the ring, so the core does not sit idle while the CPU takes the interrupt, signals the
  fence and the scheduler queues the next job.
- So it matters only when a program **submits ahead** (or two programs share the NPU). Our programs that wait
  for each job (jb_run `--wait`, teflon_check invokes) hand the kernel one job at a time: no change for them.
- The parameter is 0444 (read-only) and is passed to `drm_sched_init()` once, so today it can only change at a
  reboot (or a module reload, which needs every NPU user to close the node).
- The kernel ring is one page (4 KiB) and wraps without checking how far the front end has read
  (etnaviv_buffer_reserve), so the limit also keeps queued jobs from being overwritten. Stay at or below 4.

What it could win: per hidden job transition, the kernel round trip. Measured fixed cost per submit in jb_run
is ~0.12-0.16 ms (that includes the user-side part, which does not go away). Examples:
- upscaler, 1 job per frame, two frames in flight: ~0.1 ms per 14-20 ms frame, under 1%.
- LLM 256-token prefill if split into sub-chunks so CPU glue of chunk B overlaps the NPU on chunk A:
  ~120 jobs per pass, so the hidden gap could add up. Needs an engine change first.
- Back-to-back jobs with no CPU work between them: chaining them in one submit (jb_run already does) saves
  more than any queue depth.

## Ideas (A-E), and the recommendation

| | Idea | Needs | Dynamic? |
|---|---|---|---|
| A | Pick the boot value automatically: 1 if the last boot died during an NPU job (PROGRESS.md STARTING without DONE, or a new devcoredump), else 4 | root step, no kernel change | per boot |
| B | Make `hw_job_limit` writable at runtime (sysfs), implemented with per-job credits | kernel patch | yes, by hand or by scripts |
| C | Automatic trip: a recovered hang / MMU fault drops the limit to 1 until written again | kernel patch (on top of B) | yes, automatic |
| D | Programs take turns: jobs from different DRM files never share the ring (fairness: the upscaler never waits behind 4 queued LLM jobs, and the mixed case stays as tested today) | kernel patch (on top of B) | yes, automatic |
| E | Leave the kernel at 4, let each program choose its own depth (frames in flight, prefill sub-chunks) | userspace only | per program |

There is no speed reason to *lower* the limit: a deeper ring never slows the NPU down. The reasons for 1 are
safety and fairness. So "automatic" here means: **run at 4, drop to 1 on trouble, keep programs from piling up
on each other.** Recommended: B + C + D in one small patch (drafted), A later as the boot-time half (covers the
case where the board dies and C never gets to run), E as the actual speed work.

## Patch 0006 as built

- `hw_job_limit` 0644, values 1-4 (others clamped with a warning). modprobe.d still sets the boot value, so with
  `alta-etnaviv-debug.conf` (=1) the board boots exactly as today.
- Credit pool fixed at 12; each job asks for 12 / limit via `update_job_credits`. Lowering never trips the
  `drm_WARN_ON` in `drm_sched_available_credits()`; it applies as the ring drains.
- `hw_job_auto` (0644, default **on**): on the recovery path of `etnaviv_sched_timedout_job()` (MMU faults reach it
  through `drm_sched_fault()`): `hw_job_trip_from` = old limit, `hw_job_trips` += 1, `hw_job_limit` = 1, log
  `alta: hw_job_limit N -> 1 after a recovered hang (hw_job_auto, trip T)`. The kernel cannot ask the user; the
  desktop tool reads these and offers "put back to N" or "reboot".
- `hw_job_turns` (0644, default **off**): off = stock behaviour, programs share the ring (the kernel writes an MMU
  context switch + flush between two programs' jobs, etnaviv_buffer_queue). On = jobs from different DRM files
  never share the ring (wait until it is empty; FIFO stops at a job that does not fit, so no starvation).
- Load marker: `alta: hw_job_limit 1 (writable, 1-4), hw_job_auto on, hw_job_turns off`.
- Safety: run_job / update_job_credits / free_job run on the scheduler's ordered workqueue; recovery stops it first;
  a running job keeps the credits it started with. The callback runs under the run-queue spinlock and only does
  READ_ONCE / atomic_read.

## Patch 0007: program priority by name

- `hw_job_prio` (0644): up to 16 rules `name=high|normal|low` (process name as in /proc/<pid>/comm, 15 chars).
  Applied in `etnaviv_open()` (the entity is created at that priority) and, on every write, to the programs that
  already have the NPU open: `drm_sched_entity_set_priority()` + `drm_sched_entity_modify_sched()` under
  `gpu->sched_lock` (the lock etnaviv holds for job arm/push, as modify_sched requires). drm_sched only moves an
  entity to another run queue in `drm_sched_entity_select_rq()`, which returns early for single-scheduler entities
  (sched_list NULL); modify_sched sets the list again, so the move happens at the program's next quiet moment (no
  job queued, last one done). A program that keeps jobs queued all the time moves when it pauses; the window shows
  it as "moves when idle". Unnamed programs stay normal. Bad rules are skipped with a warning (never fail the load).
- `hw_job_clients` (0444): `pid name priority applied|pending` per program with the NPU open.
- `hw_job_prio_guard_ms` (0644, default 100, 0 off): the "slight multitasking" rule. drm_sched always takes the
  next job from the highest priority that has one ready, so lower priorities run in the gaps. The NPU cannot
  pre-empt; so while a higher-priority job ran within the guard time, a lower-priority job enters only an empty
  ring. A higher-priority job then waits for at most the one lower-priority job already running.
- Lock order: kernel_param_lock -> etnaviv_prio_lock -> filelist_mutex -> sched_lock. The drm device pointer for
  table writes is set after drm_dev_register and cleared first thing in unbind, under etnaviv_prio_lock.
- Limits: all Python scripts are "python3"/"python", so they share one setting; a program that opens the NPU
  during a table write can miss that one write.

## The app

- Window `npu_control.py` ("Job queue…" in the indicator menu, or the queue row): limit 1-4, automatic drop,
  turns, priority guard, status, "apply at login"; program list (saved names + programs running on the NPU) with a
  priority per name, add/remove. Changes apply at once and are saved in ~/.config/npu-status/queue.json. Without
  root step 250 (or with the current driver) it is read-only and says why.
- Trip: the indicator polls hw_job_trips each second; a new trip gives one critical notification and opens the
  window with a banner: **Put back to N** / **Keep 1 for now** / **Reboot…** (confirmation, then
  `systemctl reboot`). The answer is saved per boot, so a restarted indicator doesn't ask again.
- Login: the indicator applies the saved settings once per boot. The saved limit is held back (with a
  notification) while runs/STOP exists or after a trip in this boot; priorities and switches still apply.
- The kit's runs/STOP latch and npu-run's rules are unchanged.

## Steps to carry it out (each one needs Jordan's OK)

1. ~~Review~~ DONE (decisions above).
2. ~~Offline build~~ DONE 2026-11-21 (see top).
3. ~~Root step 250~~ WRITTEN (not run): installs the 0001-0007 module (checks sha256, vermagic, current = 140
   build, no etnaviv in the initramfs), plus a udev rule + helper that give the user's group write access to the
   five writable hw_job_* files. Keeps `hw_job_limit=1` in modprobe.d; 240 not needed.
4. ~~Tools~~ DONE: app (see above); npu-run (2026-11-21, backup npu-run.pre-20261121) records job_queue_before /
   job_queue_after (limit, auto, turns, guard, trips, prio, clients) in meta.txt, refuses to start after a trip in
   this boot (rc 77), and sets the STOP latch (rc 90) when hw_job_trips rises during a job. Tested with a fake
   parameter folder (NPU_RUN_QUEUE_PARAMS, tests only) and on the real driver with a no-op job.
5. ~~Ladder~~ DONE 2026-11-21, L0-L4 all PASS:
   - L0: verify the load marker, `modinfo -n`, parameter permissions, hw_job_clients empty, window not read-only.
   - L0 PASS 2026-11-21 (boot d6e90679): module, markers, permissions, app read/write.
   - L1 + L2 PASS 2026-11-21 (`ladder-L1L2.sh`, runs/hwjl-ladder-20261121-173959): upscaler hist2_n8_planar and
     MobileNet, 10 invokes each, at limit 1, then raised at runtime to 2 and 4: every output byte-identical to the
     limit-1 run (0 differing bytes, 40 invokes), CPU diff 1 / 4 as before, medians 12.6-13.8 ms / 8.2-8.7 ms
     (noise; these programs wait for each job), hw_job_trips 0, no kernel lines. Limit put back to 1.
   - L1: known-good workloads at limit 1: byte-identical, same timing as before (patch is neutral).
   - L3 PASS 2026-11-21 (`ladder-L3.sh` + `pipeline_check.py`, runs/hwjl-l3-20261121-174506): T worker threads,
     each its own Teflon delegate / NPU file, invoking back to back. All 9 rungs correct (0 bytes off, CPU diff 1/4),
     no trips, no kernel lines; 2 and 3 NPU clients interleaving at limit 4 work. Speed: **the limit makes no
     measurable difference.** MobileNet (clean numbers): 1 thread 8.45 ms/invoke; 2 threads 7.48 (limit 1) vs 7.50
     (limit 4), i.e. +13% from overlapping CPU-side work with the NPU, nothing from the ring. Upscaler numbers are
     dominated by the checker's own CPU compare (~36 ms per 960x540x4 output; 50 ms/invoke at 1 thread for a 13 ms
     job); 2 threads 44.7 / 42.6 / 43.8 ms at limits 1 / 2 / 4 = noise. Checker since fixed (byte compare per
     invoke, CPU check on warm-up + last only); an upscaler rerun would give clean numbers.
     Reading: the kernel round trip that a deeper ring hides is well under 1% of a 7-13 ms job.
   - L4 PASS 2026-11-21 (`ladder-L4.sh` + `prio_pair.sh`, runs/hwjl-l4-20261121-180227; checker fixed):
     upscaler clean rerun: 1 thread 16.5 ms/invoke, 2 threads 13.39 (limit 1) vs 13.22 (limit 4): no ring gain,
     +23% from two programs. Priority, MobileNet, limit 4, "hwjl_hi" 1 thread vs "hwjl_lo" 3 threads, 5 s:
       no priority:                 hi 36.7/s, median 27.1 ms | lo 108/s
       hi=high, lo=low, guard 100:  hi 61.0/s, median 15.1 ms | lo 80/s (not starved)
       hi=high, lo=low, guard 0:    hi 37.0/s, median 26.9 ms | lo 107/s   (priority alone does nothing here:
                                    lo keeps the ring full, so hi still waits behind its queued jobs)
       turns on, no priority:       hi 34.6/s, median 28.5 ms | lo 102/s
     hw_job_clients showed both programs with the right priority, "applied". All outputs byte-identical, no
     trips, no kernel lines; settings restored (limit 1, no table, guard 100, turns off).
   - L2: raise to 2, then 4 at runtime, same workloads: identical results (they do not submit ahead).
   - L3: a submit-ahead workload (upscaler hook, two frames in flight) at 1 / 2 / 4: FPS and kernel log.
   - L4: two programs at once: turns off vs on; then one at high, one at low (priority + guard): both correct,
     no hang, per-program latency; hw_job_clients shows applied/pending as expected.
   - No deliberate hang to test the trip; the next recovered hang tests it (the window's banner and buttons are
     tested on a fake /sys).
6. **Then** save 4 in the window (applied at every login, held back after a hang); the boot value can stay 1.
7. Later: A (boot-time choice from the last boot's outcome; partly covered: the saved limit is held back while runs/STOP exists) and E (prefill sub-chunk pipelining in the LLM
   engine), which is where a deeper ring can actually pay.

Optional measurement (any time after 3): the gpu_scheduler tracepoints `drm_run_job` and
`drm_sched_process_job` time each job's start and end, so the gap between back-to-back jobs at limit 1 vs 4 is
measured directly (needs tracefs, i.e. root).

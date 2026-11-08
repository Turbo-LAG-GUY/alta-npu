# Incident 2026-11-07 18:12 - T3 run 11 hard freeze

## Setup
- Test: T3, 20 separate runs via `scripts/npu-t3`, started detached as the user unit `alta-t3-20261107-180912`.
  Mesa-docs MobileNet v1 (sha256 1ce11b7e1895...), grace_hopper.bmp, `-n 5`, no TEFLON_DEBUG.
- Kernel 6.12.73-g6b2d05eda916, patched etnaviv (sha256 5717091b..., taint O), Mesa 26.1.0-devel c5423d3e273.
- `hw_job_limit=1` (root step 080), hardware watchdog armed at 60 s (root step 090), NTP synced.
- Tray indicator on (1 s updates), plus a recorder of the indicator (0.5 s updates) inside the unit.

## What happened
- Runs 01-10 all PASSED: 653 at 0.7188, top 5 identical to the CPU, median 7.10-8.05 ms per inference, no
  kernel messages. Each run started with the NPU powered down (runtime-suspended; the driver powers it off
  200 ms after the last job) and woke it once.
- Run 11 timeline, uptime in s, boot c40e6c74:
  - 6682.82  T3-11 STARTING. First 1 s sample: npu=suspended, 111 IRQs, active 3334 ms, CmaFree 1132 kB, MemAvailable 2147 MB.
  - 6688.74  classifier starts. load_delegate 6688.75-6688.78, Interpreter() (graph compile) 6688.78-6689.97,
             allocate_tensors done 6689.99, input set 6690.006.
  - 6689.31  last 1 s sample reached disk (NPU still suspended, 111 IRQs).
  - 6690.009 `invoke 1/5: begin`, the last thing on disk. The next 1 s sample (~6690.4) never arrived.
- **So the board froze within ~0.4 s of the run's first NPU job, while the NPU was being woken up and/or
  running that first job.**
- Kernel: kmsg-catch (fsync per record) caught nothing. The kernel log of boot -1 has nothing after 688.8 s
  (panfrost "Purging" lines, long before T3). The journal ends at 6682.77 s with T3-10's PASS line. There was no
  shutdown sequence, and the user journal file was left truncated.
- Recovery: Jordan reports it rebooted by itself. This kernel started at 18:13:48, ~80 s after the freeze
  (~18:12:28). That is 60 s for the watchdog plus firmware and boot time. `bootstatus` is 0 because the Meson driver
  doesn't report reset reasons, so "watchdog" is inferred, not proven. Nothing else restarts a frozen board here
  (kernel.panic=0).

## Same as the passing runs (not the cause)
- The same etnaviv.ko (sha256 5717091b...). "no marker" appears in every run record, T2c included: this build
  never prints the `alta: SH_EU` line that npu-run looks for.
- CmaFree ~1.1 MB in T3-10 (passed) and T3-11. Temperatures 33-41 C. No CREATE_DUMB/scanout errors.
  MemAvailable ~2.1 GB.

## Pattern
- T2b (14:05, hw_job_limit=4) froze 1-4 s into its run. The NPU was still powered down at the last sample, so it
  froze around the first NPU job, the same as here.
- With hw_job_limit=1: 11 runs passed (T2c + T3 01-10), then this one froze. Every passing and failing run
  woke the NPU from power-down exactly once, at its first job.
- Working hypothesis: an intermittent hang in the NPU wake-up path (power domain/clock/reset on runtime resume,
  hardware re-init, MMU restore) or in the first job straight after it. Not yet separated from "the first job of
  a new process".

## Files
- `capture/`: steps.txt (stage markers), kmsg.txt (live kernel log), observe.txt (1 s samples) for run 11.
- `run/`: npu-run record for run 11 (meta.txt; stdout/stderr were empty, lost with the page cache).
- `prev-T3-10-capture/`, `prev-T3-10-run/`: the last good run, for comparison.
- `T3-summary.txt`, `indicator-watch.txt` (indicator recorder; the last ~1 s before the freeze is missing).
- `journal-boot-1.txt`, `kernel-boot-1.txt` (boot c40e6c74), `kernel-boot0-first400.txt` (the boot after).

STOP latch set by Claude per the PROGRESS rule (2026-11-07 19:17).

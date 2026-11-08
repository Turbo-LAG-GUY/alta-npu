# Incident 2026-11-07 21:04 - wake-up test froze at its first classification

## Setup
- Test: `scripts/npu-wake 50 10`, detached as the user unit `alta-WAKE-20261107-210436`. One process, 50 classifications,
  each one after >= 10 s with the NPU powered down: 1 new-program wake-up, then 49 wake-ups inside the same
  program. Purpose: separate "the NPU waking up freezes the board" from "a new program starting on the NPU freezes it".
- Mesa-docs MobileNet v1, grace_hopper.bmp. Kernel 6.12.73-g6b2d05eda916, patched etnaviv (sha256 5717091b...),
  Mesa 26.1.0-devel c5423d3e273, hw_job_limit=1, watchdog 60 s, NPU power/control=auto, tray indicator on.
- Same boot (6c9cd78d) as the passing T4 soak, 25 minutes earlier. No NPU work in between.

## What happened (uptime, boot 6c9cd78d)
- 2755.16 WAKE STARTING. Baseline samples: NPU suspended, CmaFree 0 kB at 2758.5 and 2759.6 s, then 2064-2548 kB.
- 2761.33-2761.40 load_delegate. Meanwhile, at 2761.36-2761.54, panfrost logged 10 "Purging" lines (~7.5 MB of
  cached GPU buffers freed by reclaim). T4 had 5 such lines during its setup and passed.
- 2761.40-2762.62 Interpreter() (graph compile, 1.23 s). 2762.65 input set.
- 2761.83 last 1 s sample on disk.
- **2762.656 `invoke 1/50: begin`**, the last thing on disk. The NPU was suspended. The board froze before the next
  sample (~2762.9), i.e. within ~0.3 s of the first NPU job.
- Nothing from the NPU reached the kernel log. The journal ends at 2761.54 s. No shutdown sequence.
- Recovery: this kernel started 73 s after the freeze, consistent with the 60 s watchdog plus boot time.

## The pattern (all NPU compute runs since T2b)
| run | first job of the program | later jobs |
|---|---|---|
| T2b 14:05 (hw_job_limit=4) | FROZE | - |
| T2c 16:06 | ok | 4 ok |
| T3 runs 01-10 18:09-18:12 | 10 ok | 40 ok |
| T3 run 11 18:12 | FROZE | - |
| T4 20:38 | ok | 1999 ok |
| WAKE 21:04 | FROZE | - (none reached) |

3 freezes in 15 program starts, 0 in ~2,040 later jobs. Every freeze was at a program's first NPU job, with the NPU
powered down just before. (T3-01 at 16:19 also stopped right after "invoke 1/5: begin", but the system ran on: that
was the Claude Code session ending, not a freeze. Its CmaFree was 0 kB at the start.)
CmaFree at the start doesn't separate the two groups (freezes: ~20, 1132, 436 kB; passes: 972-1436, 9244, 50388 kB).

## Still open
The in-program wake-ups never ran, so "power-up" vs "new program" is still not separated.

## Files
`capture/` (steps, kmsg, observe), `run/` (npu-run meta; stdout/stderr empty), `WAKE-summary.txt`,
`indicator-watch.txt`, `journal-boot-6c9cd78d.txt`, `kernel-boot-6c9cd78d.txt`, `T4-capture-same-boot/` (the
passing first job, same boot, for comparison).

STOP latch set by Claude per the PROGRESS rule (2026-11-07 21:10).

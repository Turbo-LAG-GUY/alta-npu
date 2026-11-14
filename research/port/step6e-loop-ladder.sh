#!/usr/bin/env bash
# step6e-loop-ladder.sh - the first kernels with control flow (a loop with a branch out) on the NPU's shader core.
# Written 2026-11-13 (CPU only). Needs Jordan's OK before it runs. Step 6e (layernorm-style reductions) wants loops
# so a channel reduction over C = 384..768 doesn't have to be unrolled into ~5000 instructions.
#
# Kernel npu-shader cs_kernels.h cs_build_loopsum: thread i sums n words in a real loop (n is a kernel argument, capped
# at 4096 inside the kernel); 15 instructions, 3 temps, no barriers: compare, conditional branch, body (1 load, 3 adds),
# branch back. Branches have never executed on this core: if they are miscompiled the loop may not end and the NPU
# hangs (then npu-run refuses NPU jobs until a reboot), so the first rung is 1 thread, 1 iteration.
#   1 S6E-LOOP1     1 thread, 1 iteration
#   2 S6E-LOOP64    64 threads (1 workgroup), 100 iterations
#   3 S6E-LOOP256   256 threads (4 workgroups of 64), 1000 iterations, 10 launches (timing)
# Usage: research/port/step6e-loop-ladder.sh     (run detached: systemd-run --user --unit=alta-s6e-loop --collect)
set -u
K=/home/jordan/alta-npu
CS=/home/jordan/Documents/AI_Projects/npu-shader
RD=$K/runs/cs6e-loop-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi

rung() { # LABEL -- CMD...
   local label=$1; shift 2
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET \
      scripts/npu-attempt "$label" -- "$@" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc"
   grep -E '"result"' "$RD/$label.stdout" | head -3
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress compute-port-step6e-loop FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}

scripts/progress compute-port-step6e-loop STARTING "research/port/step6e-loop-ladder.sh: $RD"
rung S6E-LOOP1 -- $CS/cs_run --kernel loopsum -n 1 -w 1 --len 1 --out $RD/LOOP1
rung S6E-LOOP64 -- $CS/cs_run --kernel loopsum -n 64 -w 64 --len 100 --out $RD/LOOP64
rung S6E-LOOP256 -- $CS/cs_run --kernel loopsum -n 256 -w 64 --len 1000 --repeat 10 --out $RD/LOOP256
scripts/progress compute-port-step6e-loop DONE "all rungs passed: $RD"
echo "all rungs passed: $RD"

#!/usr/bin/env bash
# split-ladder2.sh - LLM engine step L3c follow-up on the NPU: WHY was the CPU + NPU split slower than the CPU alone in
# split-ladder.sh (2026-11-20, runs/llm-split-ladder-20261120-005522: NPU x1.8, CPU x1.3-2.1 side by side, the submit
# 0.3-0.8 ms)? Suspect: the unpinned main thread (submit + fence polling) sharing an A73 with one of the 4 pinned CPU
# workers. L1c, which saw only -6% / -8% side by side, kept the NPU side on the A53s (cpus 0-1).
# Same NPU jobs as split-ladder.sh: the submit streams and buffer lists are byte-identical (offline,
# npu-llm/runs/offline-jb-20261120-maincpu vs offline-jb-20261119/split-ladder-dry). Only host side changes:
# jb_run 3fa43e81 adds --main-cpu C (pin the main thread) and reports flush_ms (the submit ioctl alone) and
# main_on_a73 (share of the main thread's samples found on cpus 2-5).
# Written 2026-11-20 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS, and a QUIET CPU (pause other heavy jobs first).
# Stops at the first failure, STOP, an output more than 1 level off, or a wrong CPU result.
# Usage: systemd-run --user --unit=alta-llm-split2 --collect llm/l3/split-ladder2.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JB=$AI/npu-llm/bin/jb_run-3fa43e81   # frozen copy (source llm/l3/jb_run-3fa43e81.c)
P=$AI/npu-llm/models/jb
RD=$K/runs/llm-split-ladder2-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JB" | cut -c1-8)" = 3fa43e81 ] || { echo "jb_run is not 3fa43e81"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)

report() { # LABEL RC
   local label=$1 rc=$2 mx ok
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   ok=$(grep -oE '"cpu_check": (true|false)' "$RD/$label.stdout" | grep -oE '(true|false)$')
   echo "== $label rc=$rc max_diff=${mx:--} cpu_check=${ok:--} $(grep -oE '"(ms_median_after_first|npu_ms|flush_ms|spins_per_wait|cpu_ms|cpu_gbs|main_cpu|main_on_a73)": -?[0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   if [ "$rc" -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-0}" -gt 1 ] || [ "${ok:-true}" != true ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-split-ladder2 FAILED "stopped at $label rc=$rc max_diff=${mx:--} cpu_check=${ok:--}; $RD"
      exit 1
   fi
}
npu() { # LABEL PACK [jb_run options]
   local label=LLMSQ-$1 pack=$2; shift 2
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$JB" -n 200 "$@" "$P/$pack.jbp" > "$RD/$label.stdout" 2>&1
   report "$label" $?
}
cpu() { # LABEL ROWS IC [jb_run options]  (CPU only, no NPU)
   local label=LLMSQ-$1 rows=$2 ic=$3; shift 3
   "$JB" --cpu-only --cpu-rows "$rows" --cpu-ic "$ic" -n 200 "$@" > "$RD/$label.stdout" 2>&1
   report "$label" $?
}
S17=split_17b_gateup_npu6656_g4
scripts/progress llm-split-ladder2 STARTING "llm/l3/split-ladder2.sh: $RD"
# alone, main thread on an A53 (the baselines for the pinned rungs)
npu Q1-npu-17b-share-main0 $S17 --wait spin --main-cpu 0
cpu Q2-cpu-9728x2048-main0 9728 2048 --main-cpu 0
# side by side, unpinned as in split-ladder.sh (now with the submit time and where the main thread ran)
npu Q3-both-smol-unpinned smol_gateup_1x1 --wait spin --cpu-rows 3072 --cpu-ic 576
npu Q4-split-17b-unpinned $S17 --wait spin --cpu-rows 9728 --cpu-ic 2048
# side by side, main thread pinned to an A53
npu Q5-both-smol-main0 smol_gateup_1x1 --wait spin --cpu-rows 3072 --cpu-ic 576 --main-cpu 0
npu Q6-split-360m-main0 split_360m_gateup_npu1536_g4 --wait spin --cpu-rows 3584 --cpu-ic 960 --main-cpu 0
npu Q7-split-17b-main0 $S17 --wait spin --cpu-rows 9728 --cpu-ic 2048 --main-cpu 0
npu Q8-split-17b-main0-block $S17 --wait block --cpu-rows 9728 --cpu-ic 2048 --main-cpu 0
scripts/progress llm-split-ladder2 DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"

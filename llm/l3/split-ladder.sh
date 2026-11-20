#!/usr/bin/env bash
# split-ladder.sh - LLM engine step L3c on the NPU: fewer submits per token = the CPU takes the small steps and shares
# the big ones with the NPU (each side computes part of a step's output rows AT THE SAME TIME).
# Measures what the cost model (llm/l3/plan_cost.py) had to assume: (1) how much of the 0.16 ms fixed cost per submit
# is the CPU waking up (blocking vs polling fence wait), (2) how much the CPU and the NPU slow each other down when
# they stream weights together, (3) a real split step for SmolLM2-360M and 1.7B gateup shapes.
# Written 2026-11-19 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS, and a QUIET CPU (pause other heavy jobs first:
# the CPU share runs on 4 threads pinned to the A73s and waits for the slowest one).
# Every NPU job here was built offline first (npu-llm/runs/offline-jb-20261119, stock mesa-llm compiler, overrides
# unset): S1/S2/S4 = the J1 job (byte-identical to L1c); split packs = 4 jobs of 384 x 960 (kpc 16) and 1664 x 2048
# (kpc 21) in one submit, no driver complaint. The C* rungs use the CPU only (no NPU job, not through npu-run).
# Stops at the first failure, STOP, an output more than 1 level off, or a wrong CPU result.
# Usage: systemd-run --user --unit=alta-llm-split --collect llm/l3/split-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JB=$AI/npu-llm/bin/jb_run-ca878c69   # frozen copy of the binary this ladder ran with (source llm/l3/jb_run-ca878c69.c)
P=$AI/npu-llm/models/jb
RD=$K/runs/llm-split-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JB" | cut -c1-8)" = ca878c69 ] || { echo "jb_run is not ca878c69"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)

report() { # LABEL RC
   local label=$1 rc=$2 mx ok
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   ok=$(grep -oE '"cpu_check": (true|false)' "$RD/$label.stdout" | grep -oE '(true|false)$')
   echo "== $label rc=$rc max_diff=${mx:--} cpu_check=${ok:--} $(grep -oE '"(ms_median_after_first|npu_ms|read_ms|spins_per_wait|cpu_ms|cpu_gbs)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   if [ "$rc" -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-0}" -gt 1 ] || [ "${ok:-true}" != true ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-split-ladder FAILED "stopped at $label rc=$rc max_diff=${mx:--} cpu_check=${ok:--}; $RD"
      exit 1
   fi
}
npu() { # LABEL PACK [jb_run options]
   local label=LLMSP-$1 pack=$2; shift 2
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$JB" -n 200 "$@" "$P/$pack.jbp" > "$RD/$label.stdout" 2>&1
   report "$label" $?
}
cpu() { # LABEL ROWS IC  (CPU only, no NPU)
   local label=LLMSP-$1
   "$JB" --cpu-only --cpu-rows "$2" --cpu-ic "$3" -n 200 > "$RD/$label.stdout" 2>&1
   report "$label" $?
}
scripts/progress llm-split-ladder STARTING "llm/l3/split-ladder.sh: $RD"
# wake-up share of the fixed cost: the J1 job with an explicit flush + blocking wait, then + polling wait
npu S1-block smol_gateup_1x1 --wait block
npu S2-spin smol_gateup_1x1 --wait spin
# the CPU shares alone (no NPU)
cpu C1-cpu-3072x576 3072 576
cpu C2-cpu-3584x960 3584 960
cpu C3-cpu-9728x2048 9728 2048
# the NPU shares alone
npu S3b-npu-360m-share split_360m_gateup_npu1536_g4 --wait spin
npu S3c-npu-17b-share split_17b_gateup_npu6656_g4 --wait spin
# together: same-size halves (CPU 3072 x 576 next to the J1 job), then the real split steps
npu S4-both-smol smol_gateup_1x1 --wait spin --cpu-rows 3072 --cpu-ic 576
npu S5-split-360m-gateup split_360m_gateup_npu1536_g4 --wait spin --cpu-rows 3584 --cpu-ic 960
npu S6-split-17b-gateup split_17b_gateup_npu6656_g4 --wait spin --cpu-rows 9728 --cpu-ic 2048
scripts/progress llm-split-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"

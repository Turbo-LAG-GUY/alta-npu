#!/usr/bin/env bash
# jb-ladder.sh - LLM engine step L3b on the NPU: NN jobs through the Gallium ML API directly (npu-llm/bin/jb_run, no
# TFLite / Teflon delegate) vs Teflon, and several NN jobs per submit (weight groups, a dependent chain, batches).
# Written 2026-11-19 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS, and a fresh boot after the 2026-11-19 hang
# (npu-run refuses NPU jobs in a boot that logged one) with runs/STOP reviewed and removed by Jordan.
# Every job here was built offline first (llm/l3/jb_offline.sh, npu-llm/runs/offline-jb-20261119) by the stock
# mesa-llm compiler with the tiling overrides UNSET (kernels per core 12-21, never above the hardware cap of 21):
#   J1 is byte-identical to the job of L1c / L3 rung 1 (NN config, weights, command stream, buffer list);
#   multi-job submits are the pattern Teflon already uses for multi-op graphs (MobileNet: 28 ops, one submit).
# Stops at the first failure, STOP, or an output more than 1 level (chain: 2) from the reference.
# Usage: systemd-run --user --unit=alta-llm-jb --collect llm/l3/jb-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
JB=$AI/npu-llm/bin/jb_run-2b12a1ed   # frozen copy of the binary this ladder ran with (source llm/l3/jb_run-2b12a1ed.c)
LIB=$AI/mesa-llm/build/src/gallium/targets/teflon/libteflon.so
P=$AI/npu-llm/models/jb
M=$AI/npu-llm/models/sizes
RD=$K/runs/llm-jb-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JB" | cut -c1-8)" = 2b12a1ed ] || { echo "jb_run is not 2b12a1ed"; exit 1; }
[ "$(sha256sum "$LIB" | cut -c1-8)" = 678775c5 ] || { echo "mesa-llm lib is not 678775c5"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)

check() { # LABEL MAXDIFF KEY RC
   local label=$1 lim=$2 key=$3 rc=$4 mx
   mx=$(grep -oE "\"$key\": [0-9]+" "$RD/$label.stdout" | grep -oE '[0-9]+$')
   echo "== $label rc=$rc max_diff=${mx:-?} $(grep -oE '"(ms_median_after_first|invoke_ms_median_after_first|ctx_ms|invoke_call_ms|read_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   if [ "$rc" -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-99}" -gt "$lim" ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-jb-ladder FAILED "stopped at $label rc=$rc max_diff=${mx:-?}; $RD"
      exit 1
   fi
}
jb() { # LABEL PACK MAXDIFF [jb_run options]
   local label=LLMJB-$1 pack=$2 lim=$3; shift 3
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$JB" -n 200 "$@" "$P/$pack.jbp" > "$RD/$label.stdout" 2>&1
   check "$label" "$lim" diff_vs_expected_max $?
}
teflon() { # LABEL MODEL
   local label=LLMJB-$1
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- $PY -I $AI/npu-shader/teflon_check.py -m "$M/$2.tflite" \
      -r "$M/${2}_reference.npz" -e "$LIB" -n 200 > "$RD/$label.stdout" 2>&1
   check "$label" 1 diff_vs_cpu_max $?
}
scripts/progress llm-jb-ladder STARTING "llm/l3/jb-ladder.sh: $RD"
# J1/J2: the same job, direct vs Teflon (same boot, back to back); J3: a context per invoke like Teflon
jb J1-direct smol_gateup_1x1 1
teflon J2-teflon smol_gateup_1x1_P
jb J3-ctx-per-invoke smol_gateup_1x1 1 --ctx per-invoke
# J4: four copies of the job in one submit (cost of one more job inside a submit)
jb J4-batch4 smol_gateup_1x1 1 --batch 4
# J5/J6: weight channel groups, G jobs (own scale each) in one submit
jb J5-groups4 smol_gateup_1x1_g4 1
jb J6-groups16 smol_gateup_1x1_g16 1
# J7: gateup -> down, two dependent jobs in one submit
jb J7-chain smol_gateup_down_1x1 2
scripts/progress llm-jb-ladder DONE "all 7 rungs passed: $RD"
echo "ALL PASSED: $RD"

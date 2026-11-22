#!/usr/bin/env bash
# engine-ladder.sh - one process runs the whole model's NPU side (2026-11-21, LLM engine R2 step 1): jb_engine compiles
# all 120 layer steps of SmolLM2-135M (llm/l3/real_packs.py packs, the same jobs that passed 240/240 one by one in
# runs/llm-real-ladder-20261120-120736) and runs them back to back in one context, 20 rounds, every output checked in
# the first and last round. E1 = 1 token per step, E2 = 256 tokens per step (one prompt chunk).
# Offline (npu-llm/runs/offline-engine-20261121): 120 compiled per size, one submit per step, every step's stream size
# and buffer count the same as alone, no driver complaint; peak memory 119 MB at 1 token (the weights once).
# Written 2026-11-21 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS. Tell the other Claude session first.
# Stops at the first failure, STOP, an output more than 1 level from the reference, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-engine --collect llm/l3/engine-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JE=$AI/npu-llm/bin/jb_engine-f85c58fa
PD=$AI/npu-llm/models/jb/real-135m
RD=$K/runs/llm-engine-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JE" | cut -c1-8)" = f85c58fa ] || { echo "jb_engine is not f85c58fa"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

run() { # LABEL TOKENS
   local label=LLMEN-$1 rc mx bad w
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$JE" -n 20 --tokens "$2" "$PD" > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_packs": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $label rc=$rc max_diff=${mx:-?} bad_packs=${bad:-?} kernel_warnings=$w"
   grep -E '^compiled|^  [a-z]' "$RD/$label.stdout"
   grep -oE '"(round_ms_median|ms_per_token|compile_ms|peak_rss_mb|readout_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' '; echo
   if [ "$rc" -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-99}" -gt 1 ] || [ "${bad:-99}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-engine-ladder FAILED "stopped at $label rc=$rc max_diff=${mx:-?} bad_packs=${bad:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}
scripts/progress llm-engine-ladder STARTING "llm/l3/engine-ladder.sh: $RD"
run E1-all-steps-1-token 1
run E2-all-steps-256-tokens 256
scripts/progress llm-engine-ladder DONE "both rungs passed: $RD"
echo "ALL PASSED: $RD"

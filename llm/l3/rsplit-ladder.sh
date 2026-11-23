#!/usr/bin/env bash
# rsplit-ladder.sh (2026-11-23, LLM engine): the CALIBRATED 1.7B down split with range-grouped jobs on the NPU.
# NEEDS JORDAN'S OK BEFORE IT RUNS. Message the other NPU sessions first; Steam closed; quiet CPU.
#
# jb_run 3b8f7fdf / jb_engine 751f40b7 = mesa-llm 4eb662fc + keep-image + the GATED tile rule (ETNA_ML_V7_TILE_FIT only
# tiles when the untiled image can't be cached and a DDR estimate favours tiles; offline its configs are byte-identical
# to the best measured layouts: 1.7B 768/768 = tile-ladder B3, 135M 960/960 = keep-only R1) + the SPLIT op in the pack
# format (channel SPLIT = aliasing at offsets in the input buffer, no job; offline: pieces read base + k x 512 KiB).
# The no-flush hook is NOT used (it hung the NPU, incidents/2026-11-22-1526-LLMT-S2-noflush).
# Packs npu-llm/models/jb/real-1p7b-split4 (real_packs.py 4 jobs, DOWN_SPLIT=4range: down = 4 input pieces x 4 jobs
# grouped by calibrated output range, uneven sizes ~1000 / 630 / 300 / 40-100 rows; sim KL .145 / .086 vs .182 / .089
# unsplit). Offline: npu-llm/runs/offline-split4r-1p7b-* PASS.
#   S0 one down pack (jb_run): L00, 16 uneven jobs (smallest 44-85 rows), 256 tokens
#   S1 one down pack, 1 token (16 jobs on one token: the decode shape)
#   S2 1.7B first 4 layers, 256 tokens, gated rule (compare: unsplit 218.3 ms, equal-size split 174.6 ms)
#   S3 1.7B first 4 layers, 1 token (compare: unsplit 58.1 ms)
# Stops at the first failure, STOP, an output more than 1 level off, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-rsplit --collect llm/l3/rsplit-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JR=$AI/npu-llm/bin/jb_run-3b8f7fdf
JE=$AI/npu-llm/bin/jb_engine-751f40b7
J=$AI/npu-llm/models/jb
RD=$K/runs/llm-rsplit-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JR" | cut -c1-8)" = 3b8f7fdf ] || { echo "jb_run is not 3b8f7fdf"; exit 1; }
[ "$(sha256sum "$JE" | cut -c1-8)" = 751f40b7 ] || { echo "jb_engine is not 751f40b7"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
ls -d $AI/npu-llm/runs/offline-split4r-1p7b-* >/dev/null 2>&1 || {
   echo "offline checks missing"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_KEEP_IMAGE_CACHE -u ETNA_ML_SHARE_INPUT_TRANSPOSE
       -u ETNA_ML_V7_TILE_FIT -u ETNA_ML_V7_TILE_FIT_MAX_IC -u ETNA_ML_V7_TILE_FIT_ALWAYS -u ETNA_ML_NO_FLUSH_INDEPENDENT)
RULE=(ETNA_ML_KEEP_IMAGE_CACHE=1 ETNA_ML_V7_TILE_FIT=262144)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL RC MAXDIFF BAD
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "${3:-99}" -gt 1 ] || [ "${4:-99}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-rsplit-ladder FAILED "stopped at $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}

eng() { # LABEL PACKDIR TOKENS LAYERS
   local label=LLMR-$1 rc mx bad
   "${CLEAN[@]}" "${RULE[@]}" scripts/npu-attempt "$label" -- "$JE" -n 20 --tokens "$3" --layers "$4" "$2" \
      > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_packs": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   grep -E '^compiled|^  [a-z]' "$RD/$label.stdout"
   echo "$label $(grep -oE '"(round_ms_median|ms_per_token|readout_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   check "$label" "$rc" "$mx" "$bad"
}

scripts/progress llm-rsplit-ladder STARTING "llm/l3/rsplit-ladder.sh: $RD"
one() { # LABEL PACK
   local label=LLMR-$1 rc mx bad
   "${CLEAN[@]}" "${RULE[@]}" scripts/npu-attempt "$label" -- "$JR" -n 20 --wait spin "$2" > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_invokes": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   echo "$label $(grep -oE '"(ms_median_after_first|npu_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   check "$label" "$rc" "$mx" "$bad"
}
P=$J/real-1p7b-split4
one S0-down-L00-256 "$P/L00_down_t256.jbp"
one S1-down-L00-1tok "$P/L00_down_t1.jbp"
eng S2-1p7b-4layers-256 "$P" 256 4
eng S3-1p7b-4layers-1tok "$P" 1 4
scripts/progress llm-rsplit-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"

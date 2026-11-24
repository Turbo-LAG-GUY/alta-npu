#!/usr/bin/env bash
# half-ladder.sh (2026-11-24, LLM engine, design (b) question; Jordan's OK: "You have my OK to run the NPU"): how fast
# is the NPU on 128-token images vs 256? (b) = two 128-token streams per prompt chunk, so the CPU glue of one runs while
# the NPU does the other; worth it only if a 128-token step costs well under ~70% of a 256-token one.
# Packs (x86 PC, real_packs.py JB_PACK_SIZES / JB_PACK_LAYERS=4, SmolLM2-1.7B split4 4range, same weights and groups
# for both sizes): real-1p7b-split4-L4-a = 128 as 8 x 16 + 256 as 16 x 16, real-1p7b-split4-L4-b = 128 as 16 x 8
# (H > W: the v7 tile fix 348aef38 covers it; last rung). jb_engine 751f40b7 + keep-image + the gated tile rule, as in
# rsplit-ladder S2 (4 layers, 256 tokens: 170.7 ms per round).
#   H1 4 layers 256 (16 x 16)    H2 4 layers 128 (8 x 16)    H3 4 layers 128 (16 x 8)
# Stops at the first failure, STOP, an output more than 1 level off, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-half --collect llm/l3/half-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JE=$AI/npu-llm/bin/jb_engine-751f40b7
J=$AI/npu-llm/models/jb
RD=$K/runs/llm-half-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JE" | cut -c1-8)" = 751f40b7 ] || { echo "jb_engine is not 751f40b7"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
for d in a b; do [ -s "$AI/npu-llm/runs/offline-half-$d/stdout.txt" ] || { echo "offline check $d missing"; exit 1; }; done
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_KEEP_IMAGE_CACHE -u ETNA_ML_SHARE_INPUT_TRANSPOSE
       -u ETNA_ML_V7_TILE_FIT -u ETNA_ML_V7_TILE_FIT_MAX_IC -u ETNA_ML_V7_TILE_FIT_ALWAYS -u ETNA_ML_NO_FLUSH_INDEPENDENT
       -u ETNA_ML_EXACT_BO -u ETNA_ML_CACHED_COEFFS)
RULE=(ETNA_ML_KEEP_IMAGE_CACHE=1 ETNA_ML_V7_TILE_FIT=262144)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL RC MAXDIFF BAD
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "${3:-99}" -gt 1 ] || [ "${4:-99}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-half-ladder FAILED "stopped at $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}

eng() { # LABEL PACKDIR TOKENS LAYERS
   local label=LLMH-$1 rc mx bad
   "${CLEAN[@]}" "${RULE[@]}" scripts/npu-attempt "$label" -- "$JE" -n 20 --tokens "$3" --layers "$4" "$2" \
      > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_packs": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   grep -E '^compiled|^  [a-z]' "$RD/$label.stdout"
   echo "$label $(grep -oE '"(round_ms_median|ms_per_token|readout_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   check "$label" "$rc" "$mx" "$bad"
}

scripts/progress llm-half-ladder STARTING "llm/l3/half-ladder.sh: $RD"
eng H1-1p7b-4layers-256 "$J/real-1p7b-split4-L4-a" 256 4
eng H2-1p7b-4layers-128-8x16 "$J/real-1p7b-split4-L4-a" 128 4
eng H3-1p7b-4layers-128-16x8 "$J/real-1p7b-split4-L4-b" 128 4
scripts/progress llm-half-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"

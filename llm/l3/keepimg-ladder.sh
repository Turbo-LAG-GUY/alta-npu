#!/usr/bin/env bash
# keepimg-ladder.sh (2026-11-22, LLM engine): fix 1 for "wider jobs are slower" ON THE NPU. Written CPU only.
# NEEDS JORDAN'S OK BEFORE IT RUNS. Message the other NPU session(s) first; Steam closed; quiet CPU.
#
# Cause (DESIGN.md "Resume here 2026-11-22"): when a job's compressed weights don't fit the 512 KiB VIP SRAM, Mesa's
# create_nn_config also drops the input image cache, so the input is re-read from DDR once per superblock (1-job gateup
# at 256 tokens: 48 x 147 KB). Hook ETNA_ML_KEEP_IMAGE_CACHE=1 (mesa-llm etnaviv_ml_nn.c; jb_run 84bd6ece, jb_engine
# 39c22a60; unset = upstream, byte-identical offline): image at the top of the SRAM, weights partially cached below it,
# only when that leaves >= 0x20000 for the weights (the smallest weight area that has run: down 2 / 4 jobs).
# Offline first: npu-llm/runs/offline-keepimg-* (llm/l3/keep_image_offline.sh) must have PASSED.
#
# K1 one pack, jb_run: 1-job down 256 tokens (weight area 0x800..0x20000 = the proven down 2/4-job layout, out_z 576)
# K2 one pack, jb_run: 1-job gateup 256 tokens (weight area 0x800..0x5c000, out_z 3072)
# K3 jb_engine all 120 steps, 1 job, 256 tokens, hook ON   (OFF was 211.3 ms per round; predicted ~100 with read-out)
# K4 jb_engine 2 jobs, 256 tokens, hook ON                 (OFF 177.7 ms)
# K5 jb_engine 4 jobs, 256 tokens, hook OFF = same-session baseline (137.8 ms yesterday)
# K6 jb_engine 1 job, 1 token, hook ON                     (OFF 27.2 ms)
# Every output checked against the int8 reference (first and last round). Stops at the first failure, STOP, an output
# more than 1 level off, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-keepimg-npu --collect llm/l3/keepimg-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JR=$AI/npu-llm/bin/jb_run-84bd6ece
JE=$AI/npu-llm/bin/jb_engine-39c22a60
J=$AI/npu-llm/models/jb
RD=$K/runs/llm-keepimg-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JR" | cut -c1-8)" = 84bd6ece ] || { echo "jb_run is not 84bd6ece"; exit 1; }
[ "$(sha256sum "$JE" | cut -c1-8)" = 39c22a60 ] || { echo "jb_engine is not 39c22a60"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
ls -d $AI/npu-llm/runs/offline-keepimg-* >/dev/null 2>&1 || { echo "no offline-keepimg run"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_KEEP_IMAGE_CACHE)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL RC MAXDIFF BAD
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "${3:-99}" -gt 1 ] || [ "${4:-99}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-keepimg-ladder FAILED "stopped at $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}

one() { # LABEL PACK   (jb_run, hook ON)
   local label=LLMK-$1 rc mx bad
   "${CLEAN[@]}" ETNA_ML_KEEP_IMAGE_CACHE=1 scripts/npu-attempt "$label" -- "$JR" -n 20 --wait spin "$2" \
      > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_invokes": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   grep -oE '"(ms_median_after_first|npu_ms|compile_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' '; echo
   check "$label" "$rc" "$mx" "$bad"
}

eng() { # LABEL PACKDIR TOKENS on|off
   local label=LLMK-$1 rc mx bad e=()
   [ "$4" = on ] && e=(ETNA_ML_KEEP_IMAGE_CACHE=1)
   "${CLEAN[@]}" "${e[@]}" scripts/npu-attempt "$label" -- "$JE" -n 20 --tokens "$3" "$2" > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_packs": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   grep -E '^compiled|^  [a-z]' "$RD/$label.stdout"
   grep -oE '"(round_ms_median|ms_per_token|compile_ms|readout_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' '; echo
   check "$label" "$rc" "$mx" "$bad"
}

scripts/progress llm-keepimg-ladder STARTING "llm/l3/keepimg-ladder.sh: $RD"
one K1-1job-down-256 "$J/real-135m-g1/L00_down_t256.jbp"
one K2-1job-gateup-256 "$J/real-135m-g1/L00_gateup_t256.jbp"
eng K3-1job-256-on "$J/real-135m-g1" 256 on
eng K4-2jobs-256-on "$J/real-135m-g2" 256 on
eng K5-4jobs-256-off "$J/real-135m" 256 off
eng K6-1job-1-on "$J/real-135m-g1" 1 on
scripts/progress llm-keepimg-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"
